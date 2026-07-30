# Spatial Generalization on Translated FashionMNIST

A controlled study of how MLP, CNN, and Vision Transformer classifiers respond
when FashionMNIST objects move within a 64 x 64 canvas.

[Report (PDF)](reports/comparison_study.pdf) ·
[Formal metrics](results/metrics.csv) ·
[Experiment record](results/README.md)

## Study design

Each 28 x 28 FashionMNIST image is placed on a black 64 x 64 canvas.

- **Distribution A — random:** each sample is assigned a fixed pseudorandom
  valid position.
- **Distribution B — centered:** every sample is placed at top-left coordinate
  `(18, 18)`.

Each configuration is trained once on A and once on B. Both fitted models are
then evaluated on A and B, giving four settings: A -> A, B -> B, A -> B, and
B -> A.

The study contains three comparisons:

1. MLP, CNN, and ViT with patch size 8;
2. ViT patch sizes 4, 8, and 16;
3. Conv2d and Flatten + Linear patch projections at patch size 16.

## Results

| Configuration | Parameters | A -> A | B -> B | A -> B | B -> A |
|---|---:|---:|---:|---:|---:|
| MLP | 542,474 | 71.89% | 89.57% | 71.97% | 13.99% |
| CNN | 205,994 | **92.35%** | **93.14%** | **92.87%** | **35.40%** |
| ViT, patch 4 | 829,834 | 75.28% | 87.22% | 74.97% | 17.02% |
| ViT, patch 8 | 811,402 | 80.40% | 88.27% | 81.13% | 18.81% |
| ViT, patch 16 | 829,834 | 80.07% | 88.87% | 80.97% | 16.91% |
| ViT, patch 16 (Linear) | 829,834 | 79.67% | 88.80% | 80.48% | 16.30% |

Across all six configurations, switching the test distribution from A to B
after training on A changes accuracy by at most 0.90 percentage points. The
reverse shift is much harder: models trained only at the center lose
57.74-75.58 points on random positions. CNN leads every setting while using the
fewest parameters in the comparison. Among the ViTs, patch size 8 leads three
settings; the two patch projections remain within 0.61 points.

![Architecture comparison](results/figures/model_comparison.png)

These are single-seed results. Small differences are not treated as
statistically significant.

## Setup

```bash
python -m pip install -r requirements.txt
```

The formal record was produced with Python 3.12.13. Exact package versions are
in `requirements-lock.txt`; hardware and runtime details are in
`results/manifest.json`.

## Quick check

Run the unit and protocol checks:

```bash
python -m unittest discover -s tests -v
```

For a short end-to-end check:

```bash
python -m translated_fashionmnist.experiments.run \
  --groups model \
  --epochs 1 \
  --limit-train-samples 512 \
  --limit-val-samples 128 \
  --limit-test-samples 256 \
  --num-workers 0 \
  --download
```

The command writes only to the ignored `outputs/` directory.

## Full reproduction

```bash
python -m translated_fashionmnist.experiments.run \
  --groups all \
  --epochs 15 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --seed 42 \
  --num-workers 4 \
  --amp \
  --output-dir outputs/formal-reproduction \
  --download
```

| Item | Formal setting |
|---|---|
| Training / validation split | 54,000 / 6,000, fixed with seed 42 |
| Test set | official 10,000-image FashionMNIST test partition |
| Optimizer | AdamW with cosine annealing |
| Checkpoint rule | highest validation accuracy |
| Configurations / fitted models | 6 / 12 |
| Final evaluations | 24 |
| Determinism | seeded; deterministic algorithms disabled |

The test partition is used only after checkpoint selection. Every configuration
shares the same split and test placements.

## Code map

```text
translated_fashionmnist/
|-- data.py                 fixed split and A/B canvases
|-- models.py               MLP, CNN, ViT, and patch projections
|-- engine.py               shared training and evaluation steps
|-- utils.py                seeds and result serialization
`-- experiments/
    |-- config.py           six comparison definitions
    |-- protocol.py         common fit/evaluate protocol
    |-- run.py              single formal experiment entry point
    `-- plots.py            figures generated from metrics
results/
|-- metrics.csv             24 final test measurements
|-- training_history.csv    12 complete training histories
|-- manifest.json           protocol and environment record
`-- figures/                three comparison figures
reports/                    report PDF and reproducible builder
tests/                      dataset, model, protocol, and record checks
tools/                      compact submission builder
```

Build the four-file submission archive:

```bash
python tools/build_submission.py
```

## Scope

The models share a training recipe but are not matched for parameter count or
tuned separately. The record contains one training seed and one fixed set of
placement seeds, so it provides no uncertainty estimate. The experiment
isolates translation on a black canvas; it does not test rotation, scale,
clutter, or natural-image shifts.
