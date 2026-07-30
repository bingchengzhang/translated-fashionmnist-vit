"""Visualize A (random) and B (centered) samples side by side."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from .data import TranslatedFashionMNIST


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/data_samples.png")
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    base = datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        download=args.download,
        transform=transforms.ToTensor(),
    )
    random_dataset = TranslatedFashionMNIST(
        base,
        canvas_size=args.canvas_size,
        mode="A",
        seed=args.seed,
        return_position=True,
    )
    center_dataset = TranslatedFashionMNIST(
        base,
        canvas_size=args.canvas_size,
        mode="B",
        seed=args.seed,
        return_position=True,
    )

    figure, axes = plt.subplots(2, args.num_samples, figsize=(2.5 * args.num_samples, 5))
    if args.num_samples == 1:
        axes = axes.reshape(2, 1)
    for column in range(args.num_samples):
        for row, (name, dataset) in enumerate(
            (("A: random", random_dataset), ("B: center", center_dataset))
        ):
            image, label, position = dataset[column]
            y, x = position.tolist()
            axis = axes[row, column]
            axis.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
            axis.add_patch(
                patches.Rectangle(
                    (x, y),
                    28,
                    28,
                    linewidth=1.2,
                    edgecolor="red",
                    facecolor="none",
                )
            )
            axis.set_title(f"{name}\nlabel={int(label)}, (y,x)=({y},{x})")
            axis.axis("off")

    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"Saved visualization to {output.resolve()}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
