# Translated FashionMNIST

Controlled experiments on spatial generalization in image classifiers. A
FashionMNIST object is placed on a 64 x 64 canvas either at a random location
(distribution A) or at the center (distribution B). Every model is evaluated
under A -> A, B -> B, A -> B, and B -> A.

## Results

| Configuration | A -> A | B -> B | A -> B | B -> A |
|---|---:|---:|---:|---:|
| MLP | 71.89 | 89.57 | 71.97 | 13.99 |
| CNN | **92.35** | **93.14** | **92.87** | **35.40** |
| ViT, patch 16 | 80.07 | 88.87 | 80.97 | 16.91 |
| ViT, patch 8 | 80.40 | 88.27 | 81.13 | 18.81 |
| ViT, patch 4 | 75.28 | 87.22 | 74.97 | 17.02 |
| ViT, Flatten + Linear | 79.67 | 88.80 | 80.48 | 16.30 |

The CNN is strongest in all four settings. Among the ViTs, patch size 8 gives
the most balanced result. Conv2d and Flatten + Linear patch embeddings differ
by no more than 0.61 percentage points.

## Repository layout

```text
translated_fashionmnist/
├── data.py                 translated-canvas dataset
├── engine.py               shared training and evaluation loop
├── models.py               MLP, CNN, ViT, and patch embeddings
├── training.py             shared ViT training pipeline
├── utils.py                reproducibility and file utilities
├── visualize.py            dataset preview
└── experiments/
    ├── baseline.py         original four-setting ViT run
    ├── compare.py          three controlled comparison groups
    ├── config.py           experiment definitions
    ├── protocol.py         common comparison protocol
    └── plots.py            report figures
results/                    metrics, training history, and figures
reports/                    four-page report and its build script
tests/                      dataset, model, and protocol checks
tools/                      compact submission builder
```

## Reproduce

Install and test:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Use `requirements-lock.txt` to recreate the recorded Python 3.12 package
versions.

Run the original ViT baseline:

```bash
python -m translated_fashionmnist.experiments.baseline
```

Run all three comparisons:

```bash
python -m translated_fashionmnist.experiments.compare \
  --groups all \
  --epochs 15 \
  --batch-size 64 \
  --num-workers 4 \
  --download
```

Visualize the two data distributions:

```bash
python -m translated_fashionmnist.visualize
```

Formal runs use seed 42, a fixed 90%/10% training-validation split, and
validation-based checkpoint selection. The official test set is used only for
final evaluation.

## Deliverables

- [Final report](reports/comparison_study.pdf)
- [Result record](results/README.md)
- [Formal metrics](results/metrics.csv)

Build the compact four-file submission:

```bash
python tools/build_submission.py
```

## Provenance

bc designed and implemented the comparison protocol, MLP/CNN/ViT study,
patch-size ablation, patch-embedding ablation, formal runs, visualizations,
report, and submission package.
