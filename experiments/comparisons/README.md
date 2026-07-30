# Comparison experiments

This module implements the optional comparison study without modifying the
teammate repository.

## Experiment groups

| Group | Configurations | Controlled factors |
|---|---|---|
| Architecture | MLP, CNN, ViT patch 16 | data split and training protocol |
| Patch scale | ViT patch 4, 8, 16 | Transformer depth, width, and optimizer |
| Patch embedding | Conv2d, Flatten + Linear | patch size and Transformer body |

Dataset A uses random object positions; dataset B uses centered objects. Each
configuration trains one model on A and one on B, then evaluates both models on
both test distributions.

## Protocol

- 90%/10% fixed training-validation split
- 15 epochs and seed 42 for formal runs
- best checkpoint selected by validation accuracy
- official test set used only for final evaluation
- A -> A, B -> B, A -> B, and B -> A reported for every configuration

## Run

```bash
python -m experiments.comparisons.run \
  --groups all \
  --epochs 15 \
  --batch-size 64 \
  --num-workers 4 \
  --download
```

For a quick pipeline check:

```bash
python -m experiments.comparisons.run \
  --groups all \
  --epochs 1 \
  --limit-train-samples 512 \
  --limit-val-samples 128 \
  --limit-test-samples 256 \
  --num-workers 0 \
  --output-dir results/comparisons_smoke
```

The formal summary is stored in
`results/comparisons/metrics.csv`. Run histories and figures are retained in
the repository for audit but are excluded from the compact submission archive.

## External record

`references/teammate_vit.csv` records public metrics from
[`kicious/translated-fashion-mnist-vit`](https://github.com/kicious/translated-fashion-mnist-vit)
at commit `943fa7b68730bc8ea7786bb41c7b8dc1d488883a`.

That repository reports the best test value observed across epochs. The present
study selects checkpoints on validation data, so the external values are
provenance only and are not treated as a statistically equivalent baseline.
