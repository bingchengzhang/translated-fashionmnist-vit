# Translated FashionMNIST

A compact PyTorch project for studying position generalization on FashionMNIST.
Each 28 x 28 object is placed on a 64 x 64 canvas using one of two distributions:

- **A:** random position
- **B:** centered position

Models are evaluated on A -> A, B -> B, A -> B, and B -> A. The repository
contains the original ViT baseline and three optional comparisons in one shared
Python package.

## Results

| Configuration | A -> A | B -> B | A -> B | B -> A |
|---|---:|---:|---:|---:|
| MLP | 71.89 | 89.57 | 71.97 | 13.99 |
| CNN | **92.35** | **93.14** | **92.87** | **35.40** |
| ViT, patch 16 | 80.07 | 88.87 | 80.97 | 16.91 |
| ViT, patch 8 | 80.40 | 88.27 | 81.13 | 18.81 |
| ViT, patch 4 | 75.28 | 87.22 | 74.97 | 17.02 |
| ViT, Flatten + Linear | 79.67 | 88.80 | 80.48 | 16.30 |

CNN is strongest in all four settings. Patch size 8 is the most balanced ViT
configuration. Conv2d and Flatten + Linear patch embeddings differ by at most
0.61 percentage points.

## Structure

```text
translated_fashionmnist/   shared data, models, training, and experiments
results/                    metrics, consolidated history, and three figures
reports/                    final PDF and reproducible report source
tests/                      eight dataset, model, and protocol tests
tools/                      compact submission builder
```

The former `datasets/`, `models/`, `experiments/`, and root-level training
scripts are unified under `translated_fashionmnist/`.

## Commands

Install and test:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the original four-setting ViT baseline:

```bash
python -m translated_fashionmnist.baseline
```

Run all optional comparisons:

```bash
python -m translated_fashionmnist.comparisons.run \
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
validation-based checkpoint selection. The official test set is reserved for
final evaluation.

## Deliverables

- [Final report](reports/comparison_study.pdf)
- [Result record](results/README.md)
- [Formal metrics](results/metrics.csv)

Build the four-file teaching-assistant submission:

```bash
python tools/build_submission.py
```

## Attribution

The optional comparisons, training runs, report, and packaging are credited to
**bc**. The teammate repository is retained only as an external record; details
are in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
