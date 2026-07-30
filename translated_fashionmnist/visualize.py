"""Render compact A/B sample pairs for inspection or reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from .data import TranslatedFashionMNIST


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
ACCENT = "#2E7485"
INK = "#182532"
MUTED = "#667480"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/data_samples.png")
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.num_samples < 1:
        raise ValueError("num-samples must be at least 1.")
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

    panel_count = 2 * args.num_samples
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(2.15 * panel_count, 2.55),
        squeeze=False,
    )
    figure.patch.set_facecolor("white")
    for sample_index in range(args.num_samples):
        for mode_index, (mode, dataset) in enumerate(
            (("A / random", random_dataset), ("B / center", center_dataset))
        ):
            image, label, position = dataset[sample_index]
            y, x = position.tolist()
            axis = axes[0, 2 * sample_index + mode_index]
            axis.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
            axis.add_patch(
                patches.Rectangle(
                    (x, y),
                    28,
                    28,
                    linewidth=1.1,
                    edgecolor=ACCENT,
                    facecolor="none",
                )
            )
            axis.set_title(
                f"{mode}\n(y,x)=({y},{x})",
                color=INK,
                fontsize=8.5,
                fontweight="semibold",
                pad=5,
            )
            axis.text(
                0.5,
                -0.09,
                CLASS_NAMES[int(label)],
                transform=axis.transAxes,
                ha="center",
                va="top",
                color=MUTED,
                fontsize=8,
            )
            axis.axis("off")

    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.83,
        bottom=0.13,
        wspace=0.08,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    print(f"Saved visualization to {output.resolve()}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
