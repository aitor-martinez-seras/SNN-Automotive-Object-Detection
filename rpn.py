from typing import Dict, List, Optional, Tuple, cast
from copy import deepcopy

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
from torchvision.ops import boxes as box_ops

from torchvision.models.detection import _utils as det_utils

# Import AnchorGenerator to keep compatibility.
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: 401
from torchvision.models.detection.image_list import ImageList

from norse.torch import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch import LICell
from norse.torch.functional.lif import lif_current_encoder


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob: Tensor, orig_pre_nms_top_n: int) -> Tuple[int, int]:
    from torch.onnx import operators

    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat((torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype), num_anchors), 0))

    # for mypy we cast at runtime
    return cast(int, num_anchors), cast(int, pre_nms_top_n)


class RPNHeadSNN(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """

    _version = 2

    # This class is instantiated inside FasterRCNN class
    def __init__(self, in_channels: int, num_anchors: int, num_steps) -> None:
        super().__init__()

        # TODO: The first approach will be done without BatchNorm.
        #  There are some implementations of spiking BatchNorm but may not work for all neurons and encoding methods.

        # Simulation time
        self.num_steps = num_steps

        # The time for each timestep
        self.dt = 0.001

        # Encoder
        self.p_enc = LIFParameters(v_th=torch.tensor(0.25))

        # Layer shapes
        self.in_channels = in_channels
        self.num_anchors = num_anchors

        # Shared layers
        self.shared_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(3-1)//2, bias=False)
        self.shared_lif = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(0.1)), dt=self.dt)

        # cls layers
        self.conv_cls = nn.Conv2d(in_channels, num_anchors, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.lif_obj = LICell(dt=self.dt)  # Leaky integrator

        # bbox regression layers
        self.conv_bbox = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.lif_bbox = LICell(dt=self.dt)

        # Initialize the layers
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        # X will be always a list of feature maps in different spatial locations. It may contain only the
        # last ftmap but also ftmaps from different depths. At each depth, at each position of the list, the Tensor
        # contains the feature maps from all the images of the batch
        for feature in x:

            # Initialize
            v = torch.zeros(*feature.shape, device=feature.device)

            # Initialize hidden states at t=0
            state_shared_lif = state_obj = state_bbox = None

            for step in range(self.num_steps):

                # Constant current LIF encoder
                z, v = lif_current_encoder(input_current=feature, voltage=v, p=self.p_enc, dt=self.dt)
                # print(f'Enc RPN: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Shared layers
                cur = self.shared_conv(z)
                spk_shared, state_shared_lif = self.shared_lif(cur, state_shared_lif)
                # print(f'Shared: {(spk_shared.count_nonzero() / spk_shared.nelement()) * 100:.3f}%')

                # cls layers
                cur = self.conv_cls(spk_shared)
                mem_obj, state_obj = self.lif_obj(cur, state_obj)

                # bbox regression layers
                cur = self.conv_bbox(spk_shared)
                mem_bbox, state_bbox = self.lif_bbox(cur, state_bbox)

            # For every spatial depth, append the logits and the bboxes
            logits.append(mem_obj)
            bbox_reg.append(mem_bbox)

        return logits, bbox_reg

    """

    # ### EXTRACT SPIKE RATES ### activate
    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        all_spike_rates_and_flops = []
        logits = []
        bbox_reg = []
        # X will be always a list of feature maps in different spatial locations. It may contain only the
        # last ftmap but also ftmaps from different depths. At each depth (at each position of the list) the Tensor
        # contains the feature maps from all the images of the batch
        for feature in x:
            device = feature.device
            v = torch.zeros(*feature.shape, device=device)
            # Initialize hidden states at t=0
            state_shared_lif = state_obj = state_bbox = None

            batch_size = feature.shape[0]
            # encoder = torch.zeros(batch_size, self.in_channels, device=x.device, requires_grad=False)
            spk_counts_shared = []
            spk_counts_obj = []
            spk_counts_bbox = []

            for step in range(self.num_steps):

                # Constant current LIF encoder
                z, v = lif_current_encoder(input_current=feature, voltage=v, p=self.p_enc, dt=self.dt)

                # Shared layers
                cur = self.shared_conv(z)
                spk_shared, state_shared_lif = self.shared_lif(cur, state_shared_lif)

                # cls layers
                cur = self.conv_cls(spk_shared)
                mem_obj, state_obj = self.lif_obj(cur, state_obj)

                # bbox regression layers
                cur = self.conv_bbox(spk_shared)
                mem_bbox, state_bbox = self.lif_bbox(cur, state_bbox)

                # Spike rates
                spk_counts_shared.append(spk_shared.flatten(start_dim=1))
                spk_counts_obj.append(mem_obj.flatten(start_dim=1))
                spk_counts_bbox.append(mem_bbox.flatten(start_dim=1))

            # For every spatial depth, append the logits and the bboxes
            logits.append(mem_obj)
            bbox_reg.append(mem_bbox)

            # Spike rates
            spk_counts_shared = (torch.stack(spk_counts_shared).sum(dim=0) / self.num_steps).mean(dim=1, keepdim=True)
            spk_counts_obj = (torch.stack(spk_counts_obj).sum(dim=0) / self.num_steps).mean(dim=1, keepdim=True)
            spk_counts_bbox = (torch.stack(spk_counts_bbox).sum(dim=0) / self.num_steps).mean(dim=1, keepdim=True)

            # FLOPs
            flops_spk_shared = torch.tensor(
                [(self.shared_conv.kernel_size[0]**2) * (spk_shared.shape[2] * spk_shared.shape[3]) * self.in_channels * self.in_channels],
                requires_grad=False, device=device
            ).repeat(batch_size, 1)
            flops_spk_obj = torch.tensor(
                [(self.conv_cls.kernel_size[0]**2) * (mem_obj.shape[2] * mem_obj.shape[3]) * self.in_channels * self.num_anchors * 4],
                requires_grad=False, device=device
            ).repeat(batch_size, 1)
            flops_spk_bbox = torch.tensor(
                [(self.conv_bbox.kernel_size[0] ** 2) * (mem_bbox.shape[2] * mem_bbox.shape[3]) * self.in_channels * self.num_anchors],
                requires_grad=False, device=device
            ).repeat(batch_size, 1)

            spike_rates_and_flops = [
                # torch.hstack((encoder, flops_enc)),
                torch.hstack((spk_counts_shared, flops_spk_shared)),
                torch.hstack((spk_counts_obj, flops_spk_obj)),
                torch.hstack((spk_counts_bbox, flops_spk_bbox)),
            ]

            # For every spatial depth, append the spike rates and flops
            all_spike_rates_and_flops.append(spike_rates_and_flops)

        return logits, bbox_reg, [item for sublist in all_spike_rates_and_flops for item in sublist]
        """

class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """
    # This class is instantiated inside FasterRCNN class
    def __init__(self, in_channels: int, num_anchors: int, conv_depth=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        # self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1, bias=False)
        # self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1, bias=False)

        # for layer in self.children():
        #     torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
        #     torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

        # Initialize the layers.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        # X will be always a list of feature maps in different spatial locations. It may contain only the
        # last ftmap but also ftmaps from different depths. At each depth, at each position of the list, the Tensor
        # contains the feature maps from all the images of the batch
        for feature in x:
            t = self.conv(feature)  # Process ftmaps
            # For every anchor a channel contains the cls probability generated (binary classification)
            logits.append(self.cls_logits(t))
            # The number of channels is 4 per anchor box, as a bbox is defined by 4 numbers.
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    # The -1 tells torch to fill that dimension. This is done to create a dimension for the bboxes in the Channels.
    # In the case of the class prediction or classification (cls), C=1 as the passed value in
    # concat_box_prediction_layers function when calling this one, C=1. Therefore, the .view divides de AxC dimension to
    # two dimensions A, C, where C will be 1 and will be the objectness score and A the number of anchor boxes.
    # .view just creates another dim
    # that will be equal to the number of anchors as the -1 fills the  as only one value
    # Initial shape = [N, AxC, H, W]
    layer = layer.view(N, -1, C, H, W)  # [N, A, C, H, W]
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, A, C]
    layer = layer.reshape(N, -1, C)  # [N, (H,W,A), C]
    return layer


def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    # This class flattens the List of tensors where each position is a spatial location and the tensors are
    # of the shape (N, AxC, H, W) being A number of anchors per location and C=1 in cls and C=4 in regresion.
    # C refers to the number of outputs that have to be predicted (1 in clasification and 4 in regression)
    # The final result is a Tensor with shape [(spatial_loc,N,H,W,A), C], that is, two dimensions. I understand
    # that the order of the operations is to arrange the flattened dimension (spatial_loc,N,H,W,A) the correct
    # way to continue the processing in further operations of the rpn.
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # N = batch, AxC = Anchors x Channels,
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4  # A=num_anchors_per_location (3 in the default case)
        C = AxC // A  # C=1

        # [N, (H,W,A), C], C=1. Then accumulate in a list per spatial location
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, (H,W,A), C], C=4 (4 points to define a bbox). Then accumulate in a list per spatial location.
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    # List per spatial location [(N, (H,W,A), C), (N, (H,W,A), C), ...] gets converted to [spatial_loc, N, (H,W,A), C]
    # and then flattened to [(spatial_loc,N,H,W,A), C] where C is 1 in the cls and 4 in the reg.
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).
    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n  # Is just a Dict with two keys (training: int, testing: int)
        self._post_nms_top_n = post_nms_top_n  # Same as previous
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        # targets = List[target]. Every position is an image
        # target["boxes"] = Tensor[num_boxes_in_the_image, 4]
        # target["labels"] = Tensor[num_boxes_in_the_image, 1]
        # target["image_id"] = Tensor[1]
        # target["area"] = Tensor[1]
        # target["iscrowd"] = Tensor[1]

        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]  # Tensor[num_boxes_in_the_image, 4]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # Performs IoU between gt and anchors (not final ones, the proposals, but rather the inital anchors)
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                # proposal_matcher = Matcher:
                #     This class assigns to each predicted "element" (e.g., a box) a ground-truth
                #     element. Each predicted element will have exactly zero or one matches; each
                #     ground-truth element may be assigned to zero or more predicted elements.
                #     Matching is based on the MxN match_quality_matrix, that characterizes how well
                #     each (ground-truth, predicted)-pair match. For example, if the elements are
                #     boxes, the matrix may contain box IoU overlap values.
                #     The matcher returns a tensor of size N containing the index of the ground-truth
                #     element m that matches to prediction n. If there is no match, a negative value
                #     is returned.
                # In this case, the predicted elements refers to the anchor, not the actual proposals
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor], Dict[str, Tensor]]:  # Changed signature

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)  # Tensor[num_images, num_boxes_per_image]

        # Generates a list of tensors, each tensor being filled with the index of the spatial location it represents,
        # as many times as the number of anchors in that spatial location are. Example, if there are 3 anchors per
        # spatial location and 5 spatial locations: List[tensor[0, 0 ,0], tensor[1, 1 ,1], ..., tensor[4, 4 ,4]]
        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        # Then concatenate all the list into one dimension tensor
        levels = torch.cat(levels, 0)
        # Finally the singleton dimensions (in this case the second one, as levels is reshaped to create a new
        # dimensions), are expanded to match the same as objectness, filling the singleton dimension by appending
        # the necessary times the not singleton dimension.
        # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
        # Levels is used to track the level of the proposals as they are unrolled
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms (non-max suppression)
        # Uses the variable self.pre_nms_top_n (that for test is 1000) an retrieves than number of boxes per lvl
        # If a lvl has less boxes than that, just returns the number of boxes of that lvl
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        # Like range function from python
        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        # Retrieve the top K indices from the objectness tensor. Same for levels and proposals. Mantains dimensions.
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        # SAVING PROPOSALS FOR NEW OBJ DISCOVERY  # OPT1
        proposals_and_obj_pre_nms = []
        for img_idx, prop in enumerate(proposals):
            proposals_and_obj_pre_nms.append({
                'proposals': prop,
                'objectness': objectness_prob[img_idx]
            })

        final_boxes = []
        final_scores = []
        # This loops images
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes. A low number of boxes are removed
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes. Usually score_thresh is 0, therefore no removal happens
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions  # TODO: Here we could implement the uncertainty estimation
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores, proposals_and_obj_pre_nms

    def compute_loss(
        self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # The ImageList contains the different sized images an to be able to have a Tensor where all images are,
        # the H and W dimensions are the dimensions of the biggest image and the rest
        # are places in the top left corner, leaving the rest of the image with 0s. The image sizes are kept
        # in a separate variable

        # Convention:
        # N = batch
        # A = number of anchors per spatial location
        # sp_loc = spatial location
        # C = number of outputs to define the prediction
        # H, W = Height and Width

        # For every point of the feature map is considered an anchor point and for each anchor point
        # k anchor boxes are generated, being k the number of sizes (per spatial location) * number of aspect ratios.

        # RPN uses all feature maps that are available (usually FPN is used to extract ftmaps at different layer
        # depths, at different spatial locations). Usually 5 are used.
        # A = number of anchors per spatial location and C is the number of values needed to obtain the prediction. For
        # classification (cls) is a 1, for predicting bboxes, regression (reg) is a 4, as a bbox is defined
        # by 4 points.
        features = list(features.values())  # List[[N, backbone_ch_out, H, W]]

        """
        # ### EXTRACT SPIKE RATES ### activate
        objectness, pred_bbox_deltas, spike_rates = self.head(features)
        """

        # ### EXTRACT SPIKE RATES ### deactivate
        objectness, pred_bbox_deltas = self.head(features)  # List[Tensor[[N, AxC, H, W]]]. C=1 objecness, C=4 bboxes
        # Lists of objectness and bboxes per ftmap. Inside each position of the list is the result for one depth or
        # spatial location for every image on the batch.
        # objectness and pred_bbox_deltas = [N, AxC, H, W]. A = n_anchors is the
        # number of anchors per spatial location and C is the number of values needed to obtain the prediction. For
        # classification (cls) is a one, for predicting bboxes, regression (reg) is a 4, as a bbox is defined
        # by 4 points.
        # In reality, we are predicting the transformations needed to apply to the anchor boxes in order to
        # correctly fit the bbox to the object. This transformations are also 4 (dx, dy, dw, dh), center's
        # x,y and width and height.
        
        # The prediction is made solely on the features as the size of the filters of the convolutions are defined
        # such that for every pixel of the ftmaps, same number of outputs as number of anchors per spatial location is
        # generated (in case of bbox predictions, 4 outputs per pixel times the number of anchors per spatial location)

        # AnchorGenerator generates the anchors for every image in the batch. Returns a List[Tensor].
        # Each position in the list is one image and the Tensor shape = [n_anchors_per_img, 4].
        # n_anchors_per_img = anchors_per_spatial_pos * n_windows_per_ftmap
        # The number of windows per ftmap is n_windows_per_ftmap = H_ftmap * W_ftmap = H * W. (Explanation in notebook)
        # the relation between the image and the feature map
        # (image_size/ftmap_size per dimension) are used in the image.
        # List[[n_anchors_per_img, 4]] = List[[(A * (H_ftmap * W_ftmap for every spatial pos), 4]]. A = 3 anchors per spatial pos
        # Same as making the summation of the list num_anchors_per_level
        anchors = self.anchor_generator(images, features)  # List[Tensor[n_anchors_per_img, 4]]

        num_images = len(anchors)  # Each position are the anchors of an image

        # Per level or depth or spatial location, the number of anchors is retrieved by using the objectness. It loops
        # every spatial position, and in each one a Tensor [batch, ch, h, w] or
        # following the convention in the code [N, AxC, H, W]
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # If we have 3 aspect ratios and 1 size per spatial location, the n_anchors = A = 3*1 = 3.
        # The result of this operation is that the list is concatenated and Tensors are flattened to obtain
        # a 2 dimensional Tensor [(sp_loc * N * H * W * A), C] where C refers to
        # the number of outputs needed for the prediction. So C=1 in cls as it is a binary clasification
        # and C=4 in regression as bboxes are defined by 4 values.
        # Tensor[(N * sp_loc * H_ftmap * W_ftmap * A), C] = Tensor [num_boxes, C]
        # sp_loc makes reference to the fact that we need to make the sum of the multiplication
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals.
        # This is the step where the final proposals are obtained, what is achieved by appliying the predicted
        # transformations (pred_bbox_deltas) to the anchor boxes.
        # pred_bbox = [(sp_loc,N,H,W,A), 4]. Basically all the bboxes concatenated to a single dimension, that is,
        # the shape is Tensor[num_bboxes, 4], being num_bboxes = (sp_loc,N,H,W,A) as it can be derived.
        # anchors = List[Tensor[(sp_loc * A * H * W), 4]]. So the dimension of N is in the list.
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)  # Tensor[num_bboxes, 1, 4]
        # Separates the batch dimension, but it comes already separated
        proposals = proposals.view(num_images, -1, 4)  # Tensor[N, num_boxes, 4]

        # Reduces the number of proposals (converted to the variable boxes) to the predefined number of 
        # proposals after nms
        # Tuple[List[Tensor[num_boxes_per_image, 4]], List[Tensor[num_boxes_per_image, 1]]]
        boxes, scores, prop_and_obj_pre_nms = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            # The assignment is done to the anchors (not the predictions). Here non-max suppression (nms) is used,
            # among other things
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)  # List[] per image
            # Encode the matched boxes. This means to create tx, ty, tw, th using the ground truth and the anchors.
            # This is explained in
            # https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/#bounding-box-regression
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            # Now compute the loss using the predicted bbox deltas and the created regression targets
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        else:
            losses = prop_and_obj_pre_nms

        # Activate the line below to return proposals and scores
        # boxes = (boxes, scores)

        """
        # ### EXTRACT SPIKE RATES ### activate
        losses = spike_rates
        """

        return boxes, losses