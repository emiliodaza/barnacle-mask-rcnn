"""Reproducible image-level training. No held-out image is loaded here."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import BarnacleDataset, collate_fn
from src.model import get_barnacle_maskrcnn
from src.preprocess import (
    preprocess_raw,
    preprocess_gaussian,
    preprocess_clahe,
    preprocess_gaussian_then_clahe,
)

PREPROCESSORS = {
    "raw": preprocess_raw,
    "gaussian": preprocess_gaussian,
    "clahe": preprocess_clahe,
    "gaussian_clahe": preprocess_gaussian_then_clahe,
}


def environment():
    """Record the package versions needed to interpret or repeat a model run."""
    return {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "torchvision",
            "numpy",
            "opencv-python",
            "scipy",
            "matplotlib",
        ]
    }


def seed_everything(seed):
    """Seed the generators and ask PyTorch to reject nondeterministic operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_one_experiment(
    preprocess_name="raw",
    num_epochs=5,
    patch_size=512,
    patch_stride=384,
    train_ids=(1,),
    seed=42,
    save_path=None,
    device="cpu",
    threads=2,
):
    """Train on the requested original image IDs and save weights with provenance.

    Run from the repository root. The caller decides the image-level split; this
    function never loads the other image for evaluation. Return the checkpoint
    path and its configuration, including mean losses and source-file hashes.
    """
    if not train_ids or any(i not in [1, 2] for i in train_ids) or num_epochs < 1:
        raise ValueError("Choose training image IDs 1 and/or 2 and at least one epoch.")
    seed_everything(seed)
    torch.set_num_threads(threads)
    images = [f"data/img{i}.png" for i in train_ids]
    masks = [f"data/mask{i}.png" for i in train_ids]
    dataset = BarnacleDataset(
        images,
        masks,
        PREPROCESSORS[preprocess_name],
        patch_size=patch_size,
        patch_stride=patch_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    model = get_barnacle_maskrcnn(image_size=patch_size).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )
    history = []
    start = time.perf_counter()
    print(
        f"Train only {images}: {len(dataset)} patches, seed={seed}, device={device}",
        flush=True,
    )
    for epoch in range(num_epochs):
        model.train()
        totals = {}
        for step, (batch_images, targets) in enumerate(loader):
            losses = model(
                [im.to(device) for im in batch_images],
                [{k: v.to(device) for k, v in t.items()} for t in targets],
            )
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Nonfinite loss at epoch {epoch + 1}, step {step + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for k, v in losses.items():
                totals[k] = totals.get(k, 0) + v.item()
        row = {k: v / len(loader) for k, v in totals.items()}
        row.update(epoch=epoch + 1, total_loss=sum(row.values()))
        history.append(row)
        print(
            f"Epoch {epoch + 1}/{num_epochs}: loss {row['total_loss']:.4f}, elapsed {time.perf_counter() - start:.1f}s",
            flush=True,
        )
    config = {
        "train_images": images,
        "train_masks": masks,
        "epochs": num_epochs,
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "preprocess": preprocess_name,
        "seed": seed,
        "learning_rate": 1e-4,
        "freeze_backbone": True,
        "initial_weights": "COCO_V1",
        "device": str(device),
        "threads": threads,
        "deterministic_algorithms": True,
        "environment": environment(),
        "history": history,
        "training_seconds": time.perf_counter() - start,
        "patches": len(dataset),
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in images + masks + ["config/regions.json"]
        },
    }
    path = Path(
        save_path
        or f"outputs/train_{'_'.join(map(str, train_ids))}_{preprocess_name}.pth"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)
    path.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {path}", flush=True)
    return path, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-ids", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess", choices=PREPROCESSORS, default="raw")
    parser.add_argument("--output")
    args = parser.parse_args()
    train_one_experiment(
        train_ids=args.train_ids,
        num_epochs=args.epochs,
        seed=args.seed,
        preprocess_name=args.preprocess,
        save_path=args.output,
    )
