# Experiments

This subpackage contains the four-setting ViT baseline and the three controlled
comparison groups.

## Experiment groups

| Group | Configurations | Controlled factors |
|---|---|---|
| Architecture | MLP, CNN, ViT patch 8 | data split and training protocol |
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
python -m translated_fashionmnist.experiments.compare \
  --groups all \
  --epochs 15 \
  --batch-size 64 \
  --num-workers 4 \
  --download
```

For a quick pipeline check:

```bash
python -m translated_fashionmnist.experiments.compare \
  --groups all \
  --epochs 1 \
  --limit-train-samples 512 \
  --limit-val-samples 128 \
  --limit-test-samples 256 \
  --num-workers 0 \
  --output-dir results_smoke
```

The formal summary is stored in `results/metrics.csv`. The repository retains
one consolidated training-history file and three core figures; raw checkpoints
are excluded.
