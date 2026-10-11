"""COCO transfer learning; checkpoint inference never downloads initial weights."""

import torch
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_barnacle_maskrcnn(
    num_classes=2, freeze_backbone=True, pretrained=True, image_size=512
):
    """Build a two-class Mask R-CNN with fresh box and mask heads.

    Training starts from COCO weights and normally freezes the backbone.
    Use pretrained=False before loading a saved checkpoint to avoid downloading
    initial weights. image_size controls TorchVision resizing, not tile selection.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        min_size=image_size,
        max_size=image_size,
        box_detections_per_img=400,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=1000,
    )
    if not pretrained:
        # COCO uses frozen batch normalization (eps=0). Recreate it so a
        # checkpoint-only load has the same inference architecture.
        def replace_batchnorm(module):
            for name, child in list(module.named_children()):
                if isinstance(child, torch.nn.BatchNorm2d):
                    setattr(
                        module, name, FrozenBatchNorm2d(child.num_features, eps=0.0)
                    )
                else:
                    replace_batchnorm(child)

        replace_batchnorm(model.backbone)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, num_classes
    )
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes
    )
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    return model
