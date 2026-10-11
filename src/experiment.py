"""Run the declared two-image protocol and package real results for the demo."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import cv2
import torch
from src.baselines import bright_components
from src.dataset import read_rgb
from src.evaluate import evaluate_instances
from src.masks import annotation_instances
from src.predict import load_model, predict_instances, export_prediction
from src.regions import load_region, keep_inside, polygon_mask
from src.train import train_one_experiment, environment
from src.visualize import save_comparison


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit_annotations():
    records = []
    for i in [1, 2]:
        image = read_rgb(f"data/img{i}.png")
        annotation = read_rgb(f"data/mask{i}.png")
        polygon = load_region(f"data/img{i}.png")
        original = annotation_instances(annotation)
        closed = annotation_instances(annotation, closing_kernel=3)
        reference = keep_inside(original, polygon)
        records.append(
            {
                "image": f"img{i}.png",
                "raw_contours": len(original),
                "closed_contours": len(closed),
                "reference_in_region": len(reference),
                "excluded_outside_region": len(original) - len(reference),
                "compact_mask_bytes": sum(p.mask.nbytes for p in original),
                "full_mask_bytes_if_materialized": len(original)
                * image.shape[0]
                * image.shape[1],
                "region_area_pixels": int(polygon_mask(polygon, image.shape).sum()),
            }
        )
        # Labels, never model output: this makes the mask conversion inspectable.
        save_comparison(
            image,
            reference,
            reference,
            [(j, j, 1) for j in range(len(reference))],
            polygon,
            f"results/figures/annotation_audit_{i}.png",
            f"Annotation recovery · image {i} · {len(reference)} instances in frame",
        )
    write_json("results/annotation_audit.json", records)
    return records


def demo_image(image, name):
    # Preserve original coordinates; JPEG only reduces browser asset size.
    cv2.imwrite(
        f"demo/assets/{name}.jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 87],
    )
    return f"assets/{name}.jpg"


def export_demo(name, image, polygon, instances, timing, config, checkpoint):
    original_name = {
        "unseen": "unseen_img1.png",
        "heldout_1": "img1.png",
        "heldout_2": "img2.png",
    }[name]
    area = int(polygon_mask(polygon, image.shape).sum())
    result = {
        "image": image,
        "image_sha256": hashlib.sha256(
            (Path("data") / original_name).read_bytes()
        ).hexdigest(),
        "image_original_name": original_name,
        "polygon": polygon,
        "instances": instances,
        "region_area_pixels": area,
        "config": config,
        "timing": timing,
        "score_threshold": 0.25,
        "mask_threshold": 0.5,
    }
    export_prediction(
        result, f"demo/assets/{name}.json", demo_image(image, name), checkpoint
    )


def run(epochs=5, reuse=False, train_final=True):
    start = time.perf_counter()
    torch.set_num_threads(2)
    audit = audit_annotations()
    rows = []
    folds = []
    for train_id, test_id in [(1, 2), (2, 1)]:
        checkpoint = Path(f"outputs/fold_train{train_id}.pth")
        if not reuse or not checkpoint.exists():
            checkpoint, config = train_one_experiment(
                train_ids=[train_id], num_epochs=epochs, save_path=checkpoint
            )
        model, config = load_model(checkpoint)
        if (
            config.get("train_images") != [f"data/img{train_id}.png"]
            or config.get("epochs") != epochs
        ):
            raise ValueError("Checkpoint provenance does not match the declared fold.")
        image = read_rgb(f"data/img{test_id}.png")
        polygon = load_region(f"data/img{test_id}.png")
        reference = keep_inside(
            annotation_instances(read_rgb(f"data/mask{test_id}.png")), polygon
        )
        print(f"Evaluating train {train_id} → held-out {test_id}", flush=True)
        predictions, timing = predict_instances(
            image, polygon, model, score_threshold=0.25
        )
        fold = {
            "train_image": train_id,
            "test_image": test_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training": config,
            "inference": timing,
        }
        folds.append(fold)
        selected = [p for p in predictions if p.score >= 0.5]
        for threshold in [0.25, 0.5, 0.75]:
            metrics, matches = evaluate_instances(
                [p for p in predictions if p.score >= threshold], reference
            )
            row = {
                "method": "mask_rcnn",
                "train_image": train_id,
                "test_image": test_id,
                "score_threshold": threshold,
                **metrics,
            }
            rows.append(row)
            if threshold == 0.5:
                save_comparison(
                    image,
                    reference,
                    selected,
                    matches,
                    polygon,
                    f"results/figures/heldout_{test_id}.png",
                    f"Train image {train_id} → held-out image {test_id} · count {len(selected)} / {len(reference)} · F1 {metrics['f1']:.3f}",
                )
                write_json(
                    f"results/matches_{test_id}.json",
                    {"matches": matches, "metrics": metrics},
                )
        baseline = bright_components(image, polygon)
        metrics, _ = evaluate_instances(baseline, reference)
        rows.append(
            {
                "method": "bright_components",
                "train_image": train_id,
                "test_image": test_id,
                "score_threshold": None,
                **metrics,
            }
        )
        train_record = audit[train_id - 1]
        test_record = audit[test_id - 1]
        density_count = round(
            train_record["reference_in_region"]
            / train_record["region_area_pixels"]
            * test_record["region_area_pixels"]
        )
        error = density_count - len(reference)
        rows.append(
            {
                "method": "constant_training_density",
                "train_image": train_id,
                "test_image": test_id,
                "score_threshold": None,
                "reference_count": len(reference),
                "predicted_count": density_count,
                "signed_count_error": error,
                "absolute_count_error": abs(error),
                "relative_count_error": abs(error) / len(reference),
                **{
                    k: None
                    for k in [
                        "tp",
                        "fp",
                        "fn",
                        "precision",
                        "recall",
                        "f1",
                        "mean_matched_mask_iou",
                        "iou_threshold",
                    ]
                },
            }
        )
        export_demo(
            f"heldout_{test_id}",
            image,
            polygon,
            predictions,
            timing,
            config,
            checkpoint,
        )
        print(json.dumps([r for r in rows if r["test_image"] == test_id]), flush=True)
        del model
    if train_final:
        checkpoint = Path("outputs/final_raw.pth")
        if not reuse or not checkpoint.exists():
            train_one_experiment(
                train_ids=[1, 2], num_epochs=epochs, save_path=checkpoint
            )
        model, config = load_model(checkpoint)
        if (
            config.get("train_images") != ["data/img1.png", "data/img2.png"]
            or config.get("epochs") != epochs
        ):
            raise ValueError(
                "Final checkpoint provenance does not match the declared protocol."
            )
        image = read_rgb("data/unseen_img1.png")
        polygon = load_region("data/unseen_img1.png")
        predictions, timing = predict_instances(
            image, polygon, model, score_threshold=0.25
        )
        export_demo("unseen", image, polygon, predictions, timing, config, checkpoint)
        write_json(
            "results/unseen_demo.json",
            {
                "image": "unseen_img1.png",
                "proposals_at_0_5": sum(p.score >= 0.5 for p in predictions),
                "labeled": False,
                "accuracy": None,
                "inference": timing,
                "training": config,
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            },
        )
    report = {
        "protocol": "two directional image-held-out folds, no test tuning; final model is a separate unlabeled demonstration",
        "primary_score_threshold": 0.5,
        "sensitivity_thresholds": [0.25, 0.75],
        "mask_threshold": 0.5,
        "iou_threshold": 0.5,
        "dedup_mask_iou_threshold": 0.3,
        "annotation_closing": False,
        "epochs": epochs,
        "seed": 42,
        "environment": environment(),
        "folds": folds,
        "metrics": rows,
        "elapsed_seconds": time.perf_counter() - start,
        "reference_note": "Agreement with incomplete provided blue contour annotations, not verified biological truth.",
    }
    write_json("results/report.json", report)
    with open("results/metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Finished protocol in {time.perf_counter() - start:.1f}s. Results: results/report.json",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse only matching checkpoints with recorded fold provenance.",
    )
    parser.add_argument("--no-final", action="store_true")
    args = parser.parse_args()
    run(args.epochs, args.reuse, not args.no_final)
