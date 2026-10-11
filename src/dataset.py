"""Lazy patch targets avoid storing thousands of full-resolution instance masks."""

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.masks import annotation_instances
from src.regions import keep_inside, load_region, polygon_mask, region_bbox, tile_boxes


def read_rgb(path):
    """Read an RGB uint8 image; OpenCV normally returns BGR."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Cannot read {path}. Copy the original Barnacles images into data/."
        )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Cannot read {path}. Copy the original Barnacles images into data/."
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_instance(instance, box):
    """Place the visible part of a compact instance into a tile-sized mask.

    Both boxes use original coordinates. The returned array uses tile coordinates.
    """
    x, y, r, b = box
    ix, iy, ir, ib = instance.box
    left, top, right, bottom = max(x, ix), max(y, iy), min(r, ir), min(b, ib)
    mask = np.zeros((b - y, r - x), np.uint8)
    if left < right and top < bottom:
        mask[top - y : bottom - y, left - x : right - x] = instance.mask[
            top - iy : bottom - iy, left - ix : right - ix
        ]
    return mask


def masks_to_target(masks, shape, index):
    """Build the TorchVision target for one tile, including empty targets.

    Reject masks too narrow to form valid detection boxes. Class 1 is barnacle;
    class 0 is background. Boxes use exclusive right/bottom coordinates.
    """
    height, width = shape[:2]
    boxes, valid = [], []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if len(xs) < 2:
            continue
        x1, x2, y1, y2 = (
            int(xs.min()),
            int(xs.max()) + 1,
            int(ys.min()),
            int(ys.max()) + 1,
        )
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append([x1, y1, x2, y2])
        valid.append(mask)
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return {
        "boxes": boxes,
        "labels": torch.ones(len(valid), dtype=torch.int64),
        "masks": torch.from_numpy(
            np.stack(valid) if valid else np.zeros((0, height, width), np.uint8)
        ),
        "image_id": torch.tensor(index),
        "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        "iscrowd": torch.zeros(len(valid), dtype=torch.int64),
    }


class BarnacleDataset(Dataset):
    """Load paired photos once and construct tile targets only when requested.

    Pass only the training photograph(s): splitting patches here would leak
    overlapping regions into evaluation. Empty tiles and visible fragments stay
    in the dataset. Pixels outside the selected frame are blacked out.
    """

    def __init__(
        self,
        image_paths,
        mask_paths,
        preprocess_fn=None,
        crop_to_annotation=False,
        patch_size=512,
        patch_stride=384,
        region_file="config/regions.json",
    ):
        if len(image_paths) != len(mask_paths) or not image_paths:
            raise ValueError("Provide a nonempty list of paired images and masks.")
        if crop_to_annotation:
            raise ValueError(
                "Annotation-derived crops are no longer used; supply a frame region instead."
            )
        self.preprocess_fn = preprocess_fn
        self.sources = []
        self.samples = []
        for image_path, mask_path in zip(image_paths, mask_paths):
            image, mask = read_rgb(image_path), read_rgb(mask_path)
            if image.shape != mask.shape:
                raise ValueError(
                    f"Image/mask dimensions differ: {image_path}, {mask_path}"
                )
            polygon = load_region(image_path, region_file)
            roi = polygon_mask(polygon, image.shape)
            instances = keep_inside(annotation_instances(mask), polygon)
            self.sources.append(
                {
                    "image": image,
                    "instances": instances,
                    "roi": roi,
                    "name": Path(image_path).name,
                }
            )
            x, y, r, b = region_bbox(polygon)
            boxes = (
                tile_boxes((b - y, r - x), patch_size, patch_stride or patch_size)
                if patch_size
                else [(0, 0, r - x, b - y)]
            )
            for px, py, pr, pb in boxes:
                self.samples.append(
                    (len(self.sources) - 1, (x + px, y + py, x + pr, y + pb))
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source_id, box = self.samples[index]
        source = self.sources[source_id]
        x, y, r, b = box
        image = source["image"][y:b, x:r].copy()
        if self.preprocess_fn:
            image = self.preprocess_fn(image)
        # Outside the scientist's region is excluded consistently in train/inference.
        image[source["roi"][y:b, x:r] == 0] = 0
        masks = []
        for instance in source["instances"]:
            ix, iy, ir, ib = instance.box
            if max(x, ix) >= min(r, ir) or max(y, iy) >= min(b, ib):
                continue
            mask = crop_instance(instance, box)
            mask[source["roi"][y:b, x:r] == 0] = 0
            if mask.any():
                masks.append(mask)
        target = masks_to_target(masks, image.shape, index)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0, target


def make_patches(image, instance_masks, patch_size=512, patch_stride=384):
    # Compatibility helper for small arrays and tests; include negative patches.
    """Small-array helper for coverage tests; full photos use BarnacleDataset."""
    return [
        {
            "image": image[y:b, x:r],
            "masks": [m[y:b, x:r].copy() for m in instance_masks if m[y:b, x:r].any()],
        }
        for x, y, r, b in tile_boxes(image.shape, patch_size, patch_stride)
    ]


def collate_fn(batch):
    """Keep detection targets as a list because each tile has a different count."""
    return tuple(zip(*batch))
