from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.resnet import resnet50

from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers,\
    _mobilenet_extractor, resnet_fpn_backbone
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from rpn import RegionProposalNetwork, RPNHead, RPNHeadSNN
from roi_heads import RoIHeads, RoIHeadsSNN
from generalized_rcnn import GeneralizedRCNN

from norse.torch import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch import LICell
from norse.torch.functional.lif import lif_current_encoder



def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


class FasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(
            self,
            backbone,
            num_classes=None,
            # transform parameters
            min_size=768,  # 800
            max_size=1536,  # 1333
            image_mean=None,
            image_std=None,
            # RPN parameters
            rpn_anchor_generator=None,
            rpn_head=None,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,
            # Box parameters (Detection part, RoIHeads)
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            box_score_thresh=0.10,  # Minimum score to output the detection
            box_nms_thresh=0.45,  # Boxes with greater IoU than the thr and lower score are removed. 
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,  # Defines the final number of proposals per image selected inside ROI to apply pooling
            box_positive_fraction=0.25,
            bbox_reg_weights=None,
            # SNN usage
            rpn_snn=False,
            detector_snn=False,
            # For using only one BBOX prediction
            only_one_bbox=False,
            **kwargs,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
            
        if rpn_head is None:
            if rpn_snn:
                print('| --------------- |')
                print('|  Using RPN SNN  |')
                print('| --------------- |')
                rpn_head = RPNHeadSNN(
                    out_channels, rpn_anchor_generator.num_anchors_per_location()[0],
                    num_steps=12
                )
            else:
                rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0], )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            # Feature maps not indicated here are not used, even if proposals have been generated
            # Therefore the "4" ftmap is not used
            # https://erdem.pl/2020/02/understanding-region-of-interest-part-2-ro-i-align
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        if detector_snn:
            print('| ------------------- |')
            print('|  Using RoIHeadsSNN  |')
            print('| ------------------- |')
            box_head_and_predictor = FastRCNNPredictorSNNFull(
                out_channels * resolution ** 2, representation_size,
                num_classes, num_steps=16, only_one_bbox=only_one_bbox
            )
            roi_heads = RoIHeadsSNN(
                # Box
                box_roi_pool,
                box_head_and_predictor,
                box_fg_iou_thresh,
                box_bg_iou_thresh,
                box_batch_size_per_image,
                box_positive_fraction,
                bbox_reg_weights,
                box_score_thresh,
                box_nms_thresh,
                box_detections_per_img,
            )
        else:
            # Currently using RoIHeads from pytorch
            roi_heads = RoIHeads(
                # Box
                box_roi_pool,
                box_head,
                box_predictor,
                box_fg_iou_thresh,
                box_bg_iou_thresh,
                box_batch_size_per_image,
                box_positive_fraction,
                bbox_reg_weights,
                box_score_thresh,
                box_nms_thresh,
                box_detections_per_img,
            )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        print('Image mean:\t', image_mean)
        print('Image std:\t', image_std)
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        super().__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        # self.fc6 = nn.Linear(in_channels, representation_size, bias=False)
        # self.fc7 = nn.Linear(representation_size, representation_size, bias=False)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNConvFCHead(nn.Sequential):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        conv_layers: List[int],
        fc_layers: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        """
        Args:
            input_size (Tuple[int, int, int]): the input size in CHW format.
            conv_layers (list): feature dimensions of each Convolution layer
            fc_layers (list): feature dimensions of each FCN layer
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        in_channels, in_height, in_width = input_size

        blocks = []
        previous_channels = in_channels
        for current_channels in conv_layers:
            blocks.append(misc_nn_ops.Conv2dNormActivation(previous_channels, current_channels, norm_layer=norm_layer))
            previous_channels = current_channels
        blocks.append(nn.Flatten())
        previous_channels = previous_channels * in_height * in_width
        for current_channels in fc_layers:
            blocks.append(nn.Linear(previous_channels, current_channels))
            blocks.append(nn.ReLU(inplace=True))
            previous_channels = current_channels

        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        # self.cls_score = nn.Linear(in_channels, num_classes, bias=False)
        # self.bbox_pred = nn.Linear(in_channels, num_classes * 4, bias=False)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class FastRCNNPredictorSNNFull(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
        num_classes (int): number of output classes (including background)
        beta (float or torch.Tensor): membrane potential decay rate
        spike_grad: surrogate gradient function
        thresholds (list): list with the thresholds for the shared lifs
    """
    def __init__(
            self, in_channels, representation_size, num_classes, num_steps, only_one_bbox=False
    ):
        super().__init__()

        # Simulation time
        self.num_steps = num_steps
        
        # The time for each timestep
        self.dt = 0.001

        # Size of the layers
        self.in_channels = in_channels
        self.representation_size = representation_size
        self.num_classes = num_classes

        # Encoder
        self.p_enc = LIFParameters(v_th=torch.tensor(0.25))
        #self.encoder_lif = ConstantCurrentLIFEncoder(num_steps, p=LIFParameters(v_th=torch.tensor(0.5)))

        # Shared layers
        self.fc6 = nn.Linear(in_channels, representation_size, bias=False)
        self.lif6 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(0.1)), dt=self.dt)

        self.fc7 = nn.Linear(representation_size, representation_size, bias=False)
        self.lif7 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(0.1)), dt=self.dt)

        # Classification head
        self.cls_score = nn.Linear(representation_size, num_classes, bias=False)
        self.lif_cls = LICell(dt=self.dt)

        # Regression head
        print('----------------------------------')
        if only_one_bbox:
            self.only_one_bbox = only_one_bbox
            print('Using ONLY ONE Bbox for all classes for prediction')
            self.bbox_pred = nn.Linear(representation_size, 4, bias=False)
        else:
            self.only_one_bbox = only_one_bbox
            print('Using one Bbox PER CLASS for prediction')
            self.bbox_pred = nn.Linear(representation_size, num_classes * 4, bias=False)
        self.lif_bbox = LICell(dt=self.dt)

    def forward(self, x):
        # x = box_features -> Tensor[N*batch_size_per_image, ch_out, output_size, output_size]
        # batch_size_per_image is the number of proposals selected in the self.select_training_samples function,
        # from the 2000 that arrive
        x = x.flatten(start_dim=1)
        # The first dimension is kept, becoming the "batch size" for this linear operation
        # This way, we convert the dimensions [N*num_proposals_per_image, ch_out, H_out, W_out] to
        # [N*num_proposals_per_image], therefore making kind of GlobalAveragePooling
        # for every proposal of every image, as all ftmaps (the 256 ftmaps of shape (7,7), by default)
        # are shrinked to a vector of representation_size size.

        # x.shape = Tensor[N*batch_size_per_image, ch_out, output_size, output_size]

        # Initialize
        v = torch.zeros(*x.shape, device=x.device)

        # Initialize hidden states at t=0
        state_lif6 = state_lif7 = state_cls = state_bbox = None

        # Encoder
        # x = self.encoder_lif(x)

        for step in range(self.num_steps):
            # Constant current LIF encoder
            z, v = lif_current_encoder(input_current=x, voltage=v, p=self.p_enc, dt=self.dt)
            # print(f'Enc DET: {(z.count_nonzero()/z.nelement())*100:.3f}%')

            # Shared layers
            cur = self.fc6(z)
            spk_lif6, state_lif6 = self.lif6(cur, state_lif6)
            cur = self.fc7(spk_lif6)
            spk_lif7, state_lif7 = self.lif7(cur, state_lif7)
            # print(f'spk_7: {(spk_lif7.count_nonzero() / spk_lif7.nelement()) * 100:.3f}%')

            # cls layers
            cur = self.cls_score(spk_lif7)
            mem_cls, state_cls = self.lif_cls(cur, state_cls)

            # bbox regression layers
            cur = self.bbox_pred(spk_lif7)
            mem_bbox, state_bbox = self.lif_bbox(cur, state_bbox)

        # logits_cls = torch.max(torch.stack(mem_cls_rec, dim=0), dim=0)[0]
        logits_cls = mem_cls
        bbox_deltas = mem_bbox

        return logits_cls, bbox_deltas
    
    """ ### EXTRACT SPIKE RATES ### activate
    # Forward for extracting spike rates
    def forward(self, x):
        x = x.flatten(start_dim=1)
        v = torch.zeros(*x.shape, device=x.device)
        # Initialize hidden states at t=0
        state_lif6 = state_lif7 = state_cls = state_bbox = None

        batch_size = x.shape[0]
        device = x.device
        # encoder = torch.zeros(batch_size, self.in_channels, device=x.device, requires_grad=False)
        spk_counts_6 = torch.zeros(batch_size, self.representation_size, device=device, requires_grad=False)
        spk_counts_7 = torch.zeros(batch_size, self.representation_size, device=device, requires_grad=False)
        spk_counts_cls = torch.zeros(batch_size, self.num_classes, device=device, requires_grad=False)
        if self.only_one_bbox:
            spk_counts_bbox = torch.zeros(batch_size, 4, device=device, requires_grad=False)
        else:
            spk_counts_bbox = torch.zeros(batch_size, self.num_classes * 4, device=device, requires_grad=False)

        for step in range(self.num_steps):
            # Constant current LIF encoder
            z, v = lif_current_encoder(input_current=x, voltage=v, p=self.p_enc, dt=self.dt)

            # Shared layers
            cur = self.fc6(z)
            spk_lif6, state_lif6 = self.lif6(cur, state_lif6)
            cur = self.fc7(spk_lif6)
            spk_lif7, state_lif7 = self.lif7(cur, state_lif7)

            # cls layers
            cur = self.cls_score(spk_lif7)
            mem_cls, state_cls = self.lif_cls(cur, state_cls)

            # bbox regression layers
            cur = self.bbox_pred(spk_lif7)
            mem_bbox, state_bbox = self.lif_bbox(cur, state_bbox)

            # Spike counts
            # encoder += z
            spk_counts_6 += spk_lif6
            spk_counts_7 += spk_lif7
            spk_counts_cls += mem_cls
            spk_counts_bbox += mem_bbox

        # Spike rates
        # encoder = encoder / self.num_steps
        # spk_counts_6 = spk_counts_6 / self.num_steps
        # spk_counts_7 = spk_counts_7 / self.num_steps
        # spk_counts_cls = spk_counts_cls / self.num_steps
        # spk_counts_bbox = spk_counts_bbox / self.num_steps
        spk_counts_6 = (spk_counts_6 / self.num_steps).mean(dim=1, keepdim=True)
        spk_counts_7 = (spk_counts_7 / self.num_steps).mean(dim=1, keepdim=True)
        spk_counts_cls = (spk_counts_cls / self.num_steps).mean(dim=1, keepdim=True)
        spk_counts_bbox = (spk_counts_bbox / self.num_steps).mean(dim=1, keepdim=True)
        # spk_counts_6 = (spk_counts_6 / self.num_steps).sum(dim=1, keepdim=True)
        # spk_counts_7 = (spk_counts_7 / self.num_steps).sum(dim=1, keepdim=True)
        # spk_counts_cls = (spk_counts_cls / self.num_steps).sum(dim=1, keepdim=True)
        # spk_counts_bbox = (spk_counts_bbox / self.num_steps).sum(dim=1, keepdim=True)

        # FLOPS
        # flops_enc = torch.tensor([x.shape[1] * self.in_channels]).repeat(batch_size)
        flops_spk_6 = torch.tensor(
            [self.in_channels * self.representation_size], requires_grad=False, device=device
        ).repeat(batch_size, 1)
        flops_spk_7 = torch.tensor(
            [self.representation_size * self.representation_size], requires_grad=False, device=device
        ).repeat(batch_size, 1)
        flops_spk_cls = torch.tensor(
            [self.representation_size * self.num_classes], requires_grad=False, device=device
        ).repeat(batch_size, 1)
        if self.only_one_bbox:
            flops_spk_bbox = torch.tensor(
                [self.representation_size * self.num_classes], requires_grad=False, device=device
            ).repeat(batch_size, 1)
        else:
            flops_spk_bbox = torch.tensor(
                [self.representation_size * self.num_classes * 4], requires_grad=False, device=device
            ).repeat(batch_size, 1)

        spike_rates_and_flops = [
            # torch.hstack((encoder, flops_enc)),
            torch.hstack((spk_counts_6, flops_spk_6)),
            torch.hstack((spk_counts_7, flops_spk_7)),
            torch.hstack((spk_counts_cls, flops_spk_cls)),
            torch.hstack((spk_counts_bbox, flops_spk_bbox)),
        ]

        # spike_rates = {
        #     'spk_encoder': encoder,
        #     'spk6': spk_counts_6,
        #     'spk7': spk_counts_7,
        #     'spk_cls': spk_counts_cls,
        #     'spk_bbox': spk_counts_bbox
        # }

        # spk_counts_6 = (spk_counts_6 / self.num_steps).mean(dim=1)
        # spk_counts_7 = (spk_counts_7 / self.num_steps).mean(dim=1)
        # spk_counts_cls = (spk_counts_cls / self.num_steps).mean(dim=1)
        # spk_counts_bbox = (spk_counts_bbox / self.num_steps).mean(dim=1)

        return spike_rates_and_flops
        """

model_urls = {
    "fasterrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "fasterrcnn_mobilenet_v3_large_320_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
    "fasterrcnn_mobilenet_v3_large_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
}


def fasterrcnn_resnet50_fpn(
    pretrained=False, progress=True, num_classes=91,
    pretrained_backbone=True, trainable_backbone_layers=None,
    rpn_snn=False, detector_snn=False, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.
    Reference: `"Faster R-CNN: Towards Real-Time Object Detection with
    Region Proposal Networks" <https://arxiv.org/abs/1506.01497>`_.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection
    For more details on the output, you may refer to :ref:`instance_seg_output`.
    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = resnet50(pretrained=pretrained_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, rpn_snn=rpn_snn, detector_snn=detector_snn, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress)
        model.load_state_dict(state_dict, strict=False)
        overwrite_eps(model, 0.0)
    return model


def _fasterrcnn_mobilenet_v3_large_fpn(
    weights_name,
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    rpn_snn=False,
    detector_snn=False,
    **kwargs,
):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 3
    )

    if pretrained:
        pretrained_backbone = False

    backbone = mobilenet_v3_large(
        pretrained=pretrained_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d
    )
    backbone = _mobilenet_extractor(backbone, True, trainable_backbone_layers)

    anchor_sizes = (
        (
            32,
            64,
            128,
            256,
            512,
        ),
    ) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model = FasterRCNN(
        backbone, num_classes, rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        rpn_snn=rpn_snn, detector_snn=detector_snn,**kwargs
    )
    if pretrained:
        if model_urls.get(weights_name, None) is None:
            raise ValueError(f"No checkpoint is available for model {weights_name}")
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    return model


def fasterrcnn_mobilenet_v3_large_320_fpn(
    pretrained=False, progress=True, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs
):
    """
    Constructs a low resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone tunned for mobile use-cases.
    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    weights_name = "fasterrcnn_mobilenet_v3_large_320_fpn_coco"
    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )


def fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=False, progress=True, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None,
    rpn_snn=False, detector_snn=False, **kwargs
):
    """
    Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone.
    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    weights_name = "fasterrcnn_mobilenet_v3_large_fpn_coco"
    defaults = {
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        rpn_snn=False,
        detector_snn=False,
        **kwargs,
    )