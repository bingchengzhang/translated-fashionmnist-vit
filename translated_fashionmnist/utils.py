"""Shared training, serialization, and reproducibility utilities."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * count
        self.count += count


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_accuracy: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_accuracy": best_val_accuracy,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def plot_history(history: list[dict[str, float]], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        epochs,
        [100 * row["train_accuracy"] for row in history],
        label="train",
    )
    axes[1].plot(
        epochs,
        [100 * row["val_accuracy"] for row in history],
        label="validation",
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def serializable_config(namespace: object) -> dict[str, Any]:
    config = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            value = str(value)
        config[key] = value
    return config


def system_summary(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python_pid": os.getpid(),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        summary.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            }
        )
    return summary
