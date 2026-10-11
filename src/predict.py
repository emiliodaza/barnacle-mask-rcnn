"""Frame-constrained tiled inference, with sparse masks and duplicate removal."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from src.dataset import read_rgb
from src.instances import compact_mask, deduplicate, instance_record
from src.model import get_barnacle_maskrcnn
from src.regions import load_region, polygon_mask, region_bbox, tile_boxes, keep_inside
from src.train import PREPROCESSORS


def load_model(checkpoint_path, device="cpu"):
    """Restore weights and training settings without fetching pretrained weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    state = checkpoint.get("state_dict", checkpoint)
    model = get_barnacle_maskrcnn(
        pretrained=False, image_size=config.get("patch_size", 800)
    )
    model.load_state_dict(state)
    return model.to(device).eval(), config


def predict_instances(
    image,
    polygon,
    model,
    preprocess_name="raw",
    score_threshold=0.5,
    mask_threshold=0.5,
    tile_size=512,
    stride=384,
    device="cpu",
    nms_threshold=0.3,
):
    """Propose barnacles in a selected region without consulting any annotations.

    Mask the region exterior, run overlapping tiles, translate their masks to
    original coordinates, apply the centroid rule, and remove likely duplicates.
    Return compact instances and timing/candidate counts. The result is a set
    of proposals, not a verified scientific count.
    """
    if not 0 <= score_threshold <= 1 or not 0 <= mask_threshold <= 1:
        raise ValueError("Thresholds must be between zero and one.")
    roi = polygon_mask(polygon, image.shape)
    x, y, r, b = region_bbox(polygon)
    processed = PREPROCESSORS[preprocess_name](image[y:b, x:r].copy())
    processed[roi[y:b, x:r] == 0] = 0
    candidates = []
    start = time.perf_counter()
    boxes = tile_boxes(processed.shape, tile_size, stride)
    with torch.inference_mode():
        for tx, ty, tr, tb in boxes:
            patch = processed[ty:tb, tx:tr]
            tensor = torch.from_numpy(patch.copy()).permute(2, 0, 1).float() / 255.0
            prediction = model([tensor.to(device)])[0]
            keep = (prediction["scores"] >= score_threshold) & (
                prediction["labels"] == 1
            )
            scores = prediction["scores"][keep].cpu().numpy()
            masks = prediction["masks"][keep, 0].cpu().numpy() >= mask_threshold
            for mask, score in zip(masks, scores):
                instance = compact_mask(mask, (x + tx, y + ty), float(score))
                if instance is not None and instance.area >= 4:
                    candidates.append(instance)
    candidates = keep_inside(candidates, polygon)
    kept = deduplicate(candidates, nms_threshold)
    return kept, {
        "seconds": time.perf_counter() - start,
        "tiles": len(boxes),
        "candidates_in_region": len(candidates),
        "removed_duplicates": len(candidates) - len(kept),
    }


def predict_barnacles(
    image_path,
    checkpoint_path,
    preprocess_name=None,
    score_threshold=0.5,
    mask_threshold=0.5,
    region_file="config/regions.json",
    device="cpu",
):
    """Load a photo, selected frame, and checkpoint, then return review inputs.

    density_pixels is a count per image pixel; it is not a physical density.
    """
    torch.set_num_threads(2)
    image = read_rgb(image_path)
    polygon = load_region(image_path, region_file)
    model, config = load_model(checkpoint_path, device)
    instances, timing = predict_instances(
        image,
        polygon,
        model,
        preprocess_name or config.get("preprocess", "raw"),
        score_threshold,
        mask_threshold,
        device=device,
    )
    area = int(polygon_mask(polygon, image.shape).sum())
    return {
        "image": image,
        "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        "image_original_name": Path(image_path).name,
        "instances": instances,
        "polygon": polygon,
        "count": len(instances),
        "density_pixels": len(instances) / area,
        "region_area_pixels": area,
        "timing": timing,
        "config": config,
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
    }


def export_prediction(result, path, image_ref, checkpoint_path=None):
    """Write proposal JSON for the browser, with source and checkpoint hashes.

    image_ref must resolve from the demo page or be loaded separately by the user.
    Polygon outlines simplify display; evaluation uses the original binary masks.
    """
    document = {
        "schema_version": 1,
        "image": image_ref,
        "image_sha256": result.get("image_sha256"),
        "image_original_name": result.get("image_original_name"),
        "width": result["image"].shape[1],
        "height": result["image"].shape[0],
        "polygon": result["polygon"],
        "count_rule": "instance mask centroid inside/on polygon",
        "score_threshold": result.get("score_threshold", 0.5),
        "mask_threshold": result.get("mask_threshold", 0.5),
        "region_area_pixels": result["region_area_pixels"],
        "timing": result["timing"],
        "model_train_images": result.get("config", {}).get("train_images", []),
        "checkpoint_sha256": hashlib.sha256(
            Path(checkpoint_path).read_bytes()
        ).hexdigest()
        if checkpoint_path
        else None,
        "instances": [
            instance_record(p, i + 1) for i, p in enumerate(result["instances"])
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document, separators=(",", ":")) + "\n")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--region", default="config/regions.json")
    parser.add_argument("--output", default="results/prediction.json")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = predict_barnacles(
        args.image,
        args.checkpoint,
        score_threshold=args.score_threshold,
        region_file=args.region,
    )
    export_prediction(result, args.output, args.image, args.checkpoint)
    print(
        f"{result['count']} proposals in the selected region; review before using. Saved {args.output}"
    )
