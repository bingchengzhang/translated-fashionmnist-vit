# Four-setting baseline results

## Configuration

- Date: 2026-07-30
- Canvas: 64x64
- FashionMNIST object: 28x28
- Patch size: 8
- Patch embedding: convolution
- Embedding dimension: 128
- Transformer depth: 4
- Attention heads: 4
- MLP dimension: 256
- Optimizer: AdamW
- Learning rate: 3e-4 with cosine decay
- Weight decay: 0.05
- Batch size: 128
- Epochs: 15
- Seed: 42
- Train/validation split: 90% / 10%
- Model selection: best validation accuracy on the matching training distribution

Two models were trained. The A model used randomly translated training images;
the B model used centered training images. Each best checkpoint was evaluated
on both official FashionMNIST test distributions.

## Results

| Setting | Train | Test | Test loss | Test accuracy |
|---|---|---|---:|---:|
| 1 | A (random) | A (random) | 0.5896 | 79.00% |
| 2 | B (center) | B (center) | 0.3443 | 87.83% |
| 3 | A (random) | B (center) | 0.5777 | 79.87% |
| 4 | B (center) | A (random) | 3.9940 | 25.66% |

![Four-setting accuracy](assets/four_settings_summary.png)

## Initial interpretation

1. B -> B is the easiest setting because training and testing share one fixed
   location. It achieves the highest in-distribution accuracy.
2. A -> A is harder because the model must classify the object across many
   patch locations.
3. A -> B is slightly better than A -> A. The A model has learned useful
   position robustness, while centered test images form a simpler distribution.
4. B -> A drops sharply. A model trained only on centered objects relies heavily
   on the training position and does not generalize to arbitrary translations.

The large asymmetric gap between A -> B and B -> A is the main baseline result:
random translation during training improves cross-position generalization, while
fixed-position training does not.

These results are a single seeded baseline. A stronger report should repeat the
experiment with multiple seeds and report mean and standard deviation.
