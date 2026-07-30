"""Position-controlled FashionMNIST datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torchvision.transforms import functional as TF

from .utils import seed_worker

PositionMode = Literal["A", "B", "random", "center"]
POSITION_MODES = ("A", "B")
TRAIN_POSITION_SEED_OFFSET = 101
VALIDATION_POSITION_SEED_OFFSET = 202
TEST_POSITION_SEED_OFFSET = 303


def load_fashion_mnist(
    data_dir: str | Path,
    download: bool = False,
) -> tuple[Dataset, Dataset]:
    """Load the official FashionMNIST train and test partitions."""
    transform = transforms.ToTensor()
    train = tv_datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test = tv_datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )
    return train, test


def limit_dataset(dataset: Dataset, limit: int) -> Dataset:
    """Return a deterministic prefix for smoke runs; non-positive means no limit."""
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def split_train_validation(
    dataset: Dataset,
    val_fraction: float,
    seed: int,
    train_limit: int = 0,
    val_limit: int = 0,
) -> tuple[Dataset, Dataset]:
    """Create one deterministic, disjoint train/validation split."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    val_size = round(len(dataset) * val_fraction)
    if val_size == 0 or val_size == len(dataset):
        raise ValueError("dataset is too small for the requested validation split.")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    validation = Subset(dataset, order[:val_size])
    train = Subset(dataset, order[val_size:])
    return limit_dataset(train, train_limit), limit_dataset(validation, val_limit)


def create_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    """Build a seeded loader with consistent worker behavior."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def normalize_mode(mode: PositionMode | str) -> Literal["A", "B"]:
    normalized = str(mode).strip().lower()
    if normalized in {"a", "random", "translated"}:
        return "A"
    if normalized in {"b", "center", "centered", "centred"}:
        return "B"
    raise ValueError(f"Unsupported position mode: {mode!r}. Use A/random or B/center.")


class TranslatedFashionMNIST(Dataset):
    """Place each FashionMNIST image on a larger canvas.

    Mode A uses a deterministic pseudo-random location for every sample.
    Mode B places every sample in the center. Deterministic positions make
    repeated experiments comparable without storing multi-gigabyte .pt files.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        canvas_size: int = 64,
        mode: PositionMode | str = "A",
        seed: int = 42,
        return_position: bool = False,
        resample_each_epoch: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.canvas_size = int(canvas_size)
        self.mode = normalize_mode(mode)
        self.seed = int(seed)
        self.return_position = return_position
        self.resample_each_epoch = resample_each_epoch
        # A shared tensor keeps persistent DataLoader workers synchronized when
        # the training loop advances the epoch.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

        if self.canvas_size < 28:
            raise ValueError("canvas_size must be at least 28 for FashionMNIST.")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by optional per-epoch position resampling."""
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        self._epoch.fill_(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def _random_position(self, index: int, height: int, width: int) -> tuple[int, int]:
        epoch = self.epoch if self.resample_each_epoch else 0
        # Use a per-sample generator so positions do not depend on DataLoader order.
        mixed_seed = (
            self.seed
            + int(index) * 1_000_003
            + epoch * 10_000_019
        ) % (2**63 - 1)
        generator = torch.Generator().manual_seed(mixed_seed)
        max_y = self.canvas_size - height
        max_x = self.canvas_size - width
        y = int(torch.randint(max_y + 1, (1,), generator=generator).item())
        x = int(torch.randint(max_x + 1, (1,), generator=generator).item())
        return y, x

    def _position(self, index: int, height: int, width: int) -> tuple[int, int]:
        if height > self.canvas_size or width > self.canvas_size:
            raise ValueError(
                f"Image size {(height, width)} exceeds canvas size {self.canvas_size}."
            )
        if self.mode == "B":
            return (self.canvas_size - height) // 2, (self.canvas_size - width) // 2
        return self._random_position(index, height, width)

    @staticmethod
    def _to_float_tensor(image: object) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if not torch.is_floating_point(tensor):
                tensor = tensor.float().div(255)
            else:
                tensor = tensor.float()
        else:
            tensor = TF.to_tensor(image)

        if tensor.ndim != 3:
            raise ValueError(f"Expected image with [C,H,W] dimensions, got {tensor.shape}.")
        if tensor.shape[0] != 1:
            raise ValueError(f"FashionMNIST must be grayscale, got {tensor.shape[0]} channels.")
        return tensor

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        image = self._to_float_tensor(image)
        _, height, width = image.shape
        y, x = self._position(index, height, width)

        canvas = image.new_zeros((1, self.canvas_size, self.canvas_size))
        canvas[:, y : y + height, x : x + width] = image
        label_tensor = torch.as_tensor(label, dtype=torch.long)

        if self.return_position:
            position = torch.tensor([y, x], dtype=torch.long)
            return canvas, label_tensor, position
        return canvas, label_tensor
