# Translated FashionMNIST: Position Generalization

This project studies how image classifiers respond when object position changes
between training and testing. Each 28 x 28 FashionMNIST image is placed on a
64 x 64 black canvas:

- **A - random position:** the object is translated independently per sample.
- **B - centered position:** the object is fixed at the canvas center.

The four evaluations are A -> A, B -> B, A -> B, and B -> A. The comparison
study covers three questions:

1. How do MLP, CNN, and ViT differ under a position shift?
2. Which ViT patch size works best under the shared training budget?
3. Does Conv2d patch embedding differ from Flatten + Linear?

## Results

| Configuration | A -> A | B -> B | A -> B | B -> A |
|---|---:|---:|---:|---:|
| MLP | 71.89 | 89.57 | 71.97 | 13.99 |
| CNN | **92.35** | **93.14** | **92.87** | **35.40** |
| ViT, patch 16 | 80.07 | 88.87 | 80.97 | 16.91 |
| ViT, patch 8 | 80.40 | 88.27 | 81.13 | 18.81 |
| ViT, patch 4 | 75.28 | 87.22 | 74.97 | 17.02 |
| ViT, Flatten + Linear | 79.67 | 88.80 | 80.48 | 16.30 |

CNN is the strongest model in all four settings. Patch size 8 is the most
balanced ViT configuration. The two patch-embedding implementations differ by
at most 0.61 percentage points.

## Repository layout

```text
datasets/                  translated FashionMNIST dataset
models/                    ViT implementation
experiments/comparisons/   comparison models, protocol, runner, and plots
results/comparisons/       metrics, run records, and generated figures
reports/                   report source and final PDF
tests/                     dataset, model, and protocol checks
tools/                     submission builder
```

## Reproduce the comparison study

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m experiments.comparisons.run \
  --groups all \
  --epochs 15 \
  --batch-size 64 \
  --num-workers 4 \
  --download
```

The official training set is split 90%/10% for training and validation. The
best validation checkpoint is selected before evaluation on the official test
set. Formal runs use seed 42.

## Report and submission

- Final report: [`reports/comparison_study.pdf`](reports/comparison_study.pdf)
- Formal metrics: [`results/comparisons/metrics.csv`](results/comparisons/metrics.csv)
- Build the compact submission: `python tools/build_submission.py`

The submission archive contains four visible files: the report, metrics,
source-code archive, and a short reading guide. Datasets, checkpoints, caches,
and per-epoch run files are excluded.

## Attribution

The optional experiments, training runs, visualizations, and report are
credited to **bc**. The teammate repository is used only as an external record;
see [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) for provenance and protocol notes.
