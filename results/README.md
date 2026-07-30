# Formal experiment record

This directory is the compact, reviewable record used by the report. It
contains six configurations, two fits per configuration, and four final test
settings per configuration.

| Configuration | Parameters | A -> A | B -> B | A -> B | B -> A |
|---|---:|---:|---:|---:|---:|
| MLP | 542,474 | 71.89 | 89.57 | 71.97 | 13.99 |
| CNN | 205,994 | 92.35 | 93.14 | 92.87 | 35.40 |
| ViT, patch 4 | 829,834 | 75.28 | 87.22 | 74.97 | 17.02 |
| ViT, patch 8 | 811,402 | 80.40 | 88.27 | 81.13 | 18.81 |
| ViT, patch 16 | 829,834 | 80.07 | 88.87 | 80.97 | 16.91 |
| ViT, patch 16 (Linear) | 829,834 | 79.67 | 88.80 | 80.48 | 16.30 |

- `metrics.csv` contains the 24 final test measurements.
- `training_history.csv` contains all 12 fifteen-epoch histories.
- `manifest.json` records the protocol, package versions, and hardware.
- `figures/` contains one figure per comparison.

Raw checkpoints and per-fit folders are intentionally excluded. The committed
tables and figures are generated from the same CSV record.

![Architecture comparison](figures/model_comparison.png)

![Patch-size comparison](figures/patch_size_comparison.png)

![Patch-embedding comparison](figures/patch_embedding_comparison.png)
