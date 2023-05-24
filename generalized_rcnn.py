"""
Implements the Generalized R-CNN framework
"""

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from torchvision.utils import _log_api_usage_once


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.
    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    target["boxes"][bb_idx][2:] = target["boxes"][bb_idx][2:] + 2
                    print(f'The bounding box was found to be degenerate')

        with torch.no_grad():
            features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        """
        # ### EXTRACT SPIKE RATES ### activate
        proposals, rpn_spike_rates = self.rpn(images, features, targets)
        detector_spike_rates = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = rpn_spike_rates + detector_spike_rates  # Concat lists
        losses = None
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
        """

        # RPN -> Generates proposals for the Fast R-CNN, the detection part
        proposals, proposal_losses = self.rpn(images, features, targets)
        # detections = [{"boxes": p} for p in proposals]

        # Fast R-CNN -> Uses the proposals of the RPN to detect objects in the image
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # The post process takes the detections and in case of training does nothing and otherwise it changes the
        # sizes of the boxes (converts them back to fit the original images).
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        # If we are in inference, we want to extract proposals and all boxes
        if not self.training:
            # To save proposals for new object discovery
            for i in range(len(detections)):
                for k, v in proposal_losses[i].items():
                    detections[i][k] = v
            # This is done to transform all_boxes and proposals to the original image size
            if 'all_boxes' in detections[0].keys():
                detections = self.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

    def postprocess(self, result: List[Dict[str, Tensor]], image_shapes: List[Tuple[int, int]],
                    original_image_sizes: List[Tuple[int, int]], ) -> List[Dict[str, Tensor]]:
        if not self.training:
            for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
                boxes = pred["all_boxes"]

                # Transform [N, num_classes, coordinates] to [N * num_classes, coordinates] to obtain a tensor where
                # each row is just one bbox and then resize them
                boxes_shape = boxes.shape
                boxes = boxes.reshape(boxes_shape[0] * boxes_shape[1], -1)
                boxes = resize_boxes(boxes, im_s, o_im_s)

                # Transform boxes back to the original configuration
                boxes = boxes.reshape(*boxes_shape)

                # Assign new boxes to the corresponding place in the dict
                result[i]["all_boxes"] = boxes

                # Same with proposals
                if "proposals" in pred:
                    proposals = pred["proposals"]
                    proposals = resize_boxes(proposals, im_s, o_im_s)
                    result[i]["proposals"] = proposals

            return result


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)