from faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictorSNNFull, TwoMLPHead, \
    FastRCNNPredictor, _default_anchorgen
from roi_heads import RoIHeadsSNN, RoIHeads
from rpn import RegionProposalNetwork, RPNHead, RPNHeadSNN


def create_model(dataset_name, num_classes, rpn_snn, detector_snn, trainable_backbone_layers,
                 pretrained_rpn_and_detector, pretrained_fpn, num_steps_rpn, num_steps_detector, only_one_bbox=False):

    # Workaround to a SLL certificate problem when downloading weigths
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    if rpn_snn or detector_snn:
        pretrained_rpn_and_detector = False
        trainable_backbone_layers = 0
        print('---------------------------------------------------------------')
        print('WARNING: As SNN is used, only BACKBONE and FPN pretrained weighs are'
              ' used and trainable backbone layers is set to 0')
        print('---------------------------------------------------------------')

    if dataset_name == 'cityscapes':
        image_mean = [0.2869, 0.3251, 0.2839]
        image_std = [0.1870, 0.1902, 0.1872]
    else:
        image_mean = image_std = None
    model = fasterrcnn_resnet50_fpn(
        pretrained=pretrained_fpn,
        progress=True,
        num_classes=91,
        pretrained_backbone=True,
        trainable_backbone_layers=trainable_backbone_layers,
        rpn_snn=False,
        detector_snn=False,
        image_mean=image_mean,
        image_std=image_std,
        only_one_bbox=only_one_bbox,
    )

    if not pretrained_rpn_and_detector:
        print('*********************************************************')
        print('--  NOT using pretrained weights for RPN nor Detector  --')
        print('*********************************************************')
        # --------
        # RPN
        # --------
        out_channels = model.backbone.out_channels
        rpn_anchor_generator = _default_anchorgen()
        # RPN parameters
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_score_thresh = 0.0

        if rpn_snn:
            print('| --------------- |')
            print('|  Using RPN SNN  |')
            print('| --------------- |')
            rpn_head = RPNHeadSNN(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0],
                num_steps=num_steps_rpn
            )
        else:
            print('')
            print('--  Loading standard RPN module  --')
            print('')
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

        # --------
        # Detector (RoIHeads)
        # --------
        # Box parameters (Detection part, RoIHeads)
        box_roi_pool = None
        box_head = None
        box_predictor = None
        box_score_thresh = 0.4  # Minimum score to output the detection  # ORIGINAL = 0.15
        box_nms_thresh = 0.5  # Boxes with greater IoU than the thr and lower score are removed. ORIGINAL = 0.5
        box_detections_per_img = 100
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        # Defines the final number of proposals per image selected inside ROI to apply pooling
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25  # ORIGINAL 0.25
        bbox_reg_weights = None

        if detector_snn:
            print('| ------------------- |')
            print('|  Using RoIHeadsSNN  |')
            print('| ------------------- |')

            if box_roi_pool is None:
                from torchvision.ops import MultiScaleRoIAlign
                # Feature maps not indicated here are not used, even if proposals have been generated
                # Therefore the "4" ftmap is not used
                # https://erdem.pl/2020/02/understanding-region-of-interest-part-2-ro-i-align
                box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

            out_channels = model.backbone.out_channels

            if box_head is None:
                resolution = box_roi_pool.output_size[0]
                representation_size = 1024
                box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

            box_head_and_predictor = FastRCNNPredictorSNNFull(
                out_channels * resolution ** 2, representation_size,
                num_classes, num_steps=num_steps_detector, only_one_bbox=only_one_bbox
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

            print('')
            print('--  Loading standard detector module  --')
            print('')

            if box_roi_pool is None:
                from torchvision.ops import MultiScaleRoIAlign
                # Feature maps not indicated here are not used, even if proposals have been generated
                # Therefore the "4" ftmap is not used
                # https://erdem.pl/2020/02/understanding-region-of-interest-part-2-ro-i-align
                box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

            out_channels = model.backbone.out_channels

            if box_head is None:
                resolution = box_roi_pool.output_size[0]
                representation_size = 1024
                box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

            if box_predictor is None:
                representation_size = 1024
                box_predictor = FastRCNNPredictor(representation_size, num_classes)

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

        # Replace the RPN and the Detector for the new instantiated modules
        model.rpn = rpn
        model.roi_heads = roi_heads

    return model
