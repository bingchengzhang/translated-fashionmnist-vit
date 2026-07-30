"""Shared training and evaluation primitives."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import AverageMeter


class TrainingSettings(Protocol):
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    val_fraction: float
    num_workers: int


def validate_training_settings(settings: TrainingSettings) -> None:
    """Reject invalid values before datasets or checkpoints are created."""
    if settings.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if settings.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if settings.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if settings.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if not 0 < settings.val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    if settings.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    for name in ("limit_train_samples", "limit_val_samples", "limit_test_samples"):
        if getattr(settings, name, 0) < 0:
            raise ValueError(f"{name} cannot be negative.")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        loss_meter.update(loss.item(), batch_size)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Training loader is empty.")
    return {"loss": loss_meter.average, "accuracy": correct / total}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.shape[0]
        loss_meter.update(loss.item(), batch_size)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Evaluation loader is empty.")
    return {"loss": loss_meter.average, "accuracy": correct / total}
