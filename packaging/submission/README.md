# Submission guide

This archive contains the optional comparison study only.

## Read in this order

1. `REPORT.pdf` - four-page experimental report.
2. `RESULTS.csv` - all 24 formal test measurements.
3. `SOURCE_CODE.zip` - modular source code and tests.

## Included comparisons

- MLP vs. CNN vs. ViT
- ViT patch size 4 vs. 8 vs. 16
- Conv2d vs. Flatten + Linear patch embedding

All formal runs use seed 42, a fixed 90%/10% training-validation split, and
validation-based checkpoint selection. The official test set is reserved for
final evaluation.

## Reproduce

Extract `SOURCE_CODE.zip`, then run:

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

Raw datasets, checkpoints, caches, per-epoch histories, and duplicate figure
files are intentionally excluded. The report contains the final vector charts.
