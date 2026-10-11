Reproducible barnacle-counting and scientist-review prototype.

- Two independently trained image-held-out models, five epochs, seed 42.
- A separate final model trained on both labeled photographs for unlabeled demonstration.
- Checkpoints include training configuration, histories, source hashes, and package versions.
- Actual metrics, annotation audits, failure figures, and cached browser proposals are in the repository.

Place downloaded `.pth` files in `outputs/`. Inference constructs the architecture without downloading pretrained initial weights. Checkpoints target the pinned TorchVision version in `requirements.txt`.

These models produce proposals requiring human review. They do not establish reliable automatic counting. See README.md and docs/protocol.md for the measured limitations.
