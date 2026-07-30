"""Training and evaluation protocol for the optional comparisons."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms

from datasets import TranslatedFashionMNIST
from utils import (
    AverageMeter,
    plot_history,
    resolve_device,
    save_json,
    seed_worker,
    set_seed,
    write_csv,
)

from .configurations import (
    EXPERIMENTS,
    SETTING_ORDER,
    ExperimentDefinition,
    groups_for_configuration,
)
from .models import build_model, count_trainable_parameters


@dataclass
class ProtocolConfig:
    data_dir: str = "data"
    output_dir: str = "optional_experiments_vs_teammate/results"
    canvas_size: int = 64
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    deterministic: bool = False
    download: bool = False
    limit_train_samples: int = 0
    limit_val_samples: int = 0
    limit_test_samples: int = 0


def _limit_dataset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def _base_datasets(config: ProtocolConfig):
    transform = transforms.ToTensor()
    train = tv_datasets.FashionMNIST(
        root=config.data_dir,
        train=True,
        download=config.download,
        transform=transform,
    )
    test = tv_datasets.FashionMNIST(
        root=config.data_dir,
        train=False,
        download=config.download,
        transform=transform,
    )
    return train, test


def _split_train_validation(base_train: Dataset, config: ProtocolConfig):
    val_size = round(len(base_train) * config.val_fraction)
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(base_train), generator=generator).tolist()
    val_indices = order[:val_size]
    train_indices = order[val_size:]
    train_subset = _limit_dataset(Subset(base_train, train_indices), config.limit_train_samples)
    val_subset = _limit_dataset(Subset(base_train, val_indices), config.limit_val_samples)
    return train_subset, val_subset


def _loader(
    dataset: Dataset,
    config: ProtocolConfig,
    device: torch.device,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def create_train_validation_loaders(
    config: ProtocolConfig,
    train_mode: str,
    device: torch.device,
):
    base_train, _ = _base_datasets(config)
    train_subset, val_subset = _split_train_validation(base_train, config)
    train_dataset = TranslatedFashionMNIST(
        train_subset,
        canvas_size=config.canvas_size,
        mode=train_mode,
        seed=config.seed + 101,
    )
    val_dataset = TranslatedFashionMNIST(
        val_subset,
        canvas_size=config.canvas_size,
        mode=train_mode,
        seed=config.seed + 202,
    )
    return (
        _loader(train_dataset, config, device, shuffle=True, seed_offset=11),
        _loader(val_dataset, config, device, shuffle=False, seed_offset=12),
    )


def create_test_loader(
    config: ProtocolConfig,
    test_mode: str,
    device: torch.device,
) -> DataLoader:
    _, base_test = _base_datasets(config)
    base_test = _limit_dataset(base_test, config.limit_test_samples)
    test_dataset = TranslatedFashionMNIST(
        base_test,
        canvas_size=config.canvas_size,
        mode=test_mode,
        seed=config.seed + 303,
    )
    return _loader(test_dataset, config, device, shuffle=False, seed_offset=13)


def _train_epoch(
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
    return {"loss": loss_meter.average, "accuracy": correct / total}


@torch.no_grad()
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
    return {"loss": loss_meter.average, "accuracy": correct / total}


def _expected_metadata(
    definition: ExperimentDefinition,
    config: ProtocolConfig,
    train_mode: str,
) -> dict:
    return {
        "definition": asdict(definition),
        "protocol": asdict(config),
        "train_mode": train_mode,
    }


def _can_resume(run_dir: Path, expected_metadata: dict) -> bool:
    metadata_path = run_dir / "metadata.json"
    checkpoint_path = run_dir / "best.pt"
    if not metadata_path.exists() or not checkpoint_path.exists():
        return False
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return existing.get("experiment") == expected_metadata


def fit_model(
    definition: ExperimentDefinition,
    train_mode: str,
    config: ProtocolConfig,
    resume: bool,
):
    set_seed(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    run_dir = Path(config.output_dir) / "runs" / definition.config_id / f"train_{train_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_metadata = _expected_metadata(definition, config, train_mode)
    model = build_model(definition, image_size=config.canvas_size).to(device)
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = run_dir / "best.pt"

    if resume and _can_resume(run_dir, expected_metadata):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        result = json.loads((run_dir / "training_result.json").read_text(encoding="utf-8"))
        print(f"Reusing {definition.config_id}, train {train_mode}")
        return model, device, criterion, result

    train_loader, val_loader = create_train_validation_loaders(
        config,
        train_mode,
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    parameter_count = count_trainable_parameters(model)
    save_json(
        {
            "experiment": expected_metadata,
            "parameter_count": parameter_count,
            "device": str(device),
            "torch_version": torch.__version__,
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
        },
        run_dir / "metadata.json",
    )

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_epoch = 0
    started = time.time()
    print(
        f"Training {definition.config_id}, mode {train_mode} | "
        f"{parameter_count:,} parameters | {device}"
    )
    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"  epoch {epoch:02d}/{config.epochs} | "
            f"train {100 * row['train_accuracy']:.2f}% | "
            f"val {100 * row['val_accuracy']:.2f}%"
        )
        if row["val_accuracy"] > best_accuracy:
            best_accuracy = row["val_accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_accuracy": best_accuracy,
                    "definition": asdict(definition),
                },
                checkpoint_path,
            )

    write_csv(history, run_dir / "history.csv")
    plot_history(history, run_dir / "training_curves.png")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    result = {
        "config_id": definition.config_id,
        "train_mode": train_mode,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "elapsed_seconds": time.time() - started,
        "parameter_count": parameter_count,
        "checkpoint": str(checkpoint_path),
    }
    save_json(result, run_dir / "training_result.json")
    return model, device, criterion, result


def run_configuration(
    config_id: str,
    protocol: ProtocolConfig,
    resume: bool = True,
) -> list[dict]:
    definition = EXPERIMENTS[config_id]
    measured: dict[tuple[str, str], dict] = {}
    for train_mode in ("A", "B"):
        model, device, criterion, training_result = fit_model(
            definition,
            train_mode,
            protocol,
            resume,
        )
        for test_mode in ("A", "B"):
            test_loader = create_test_loader(protocol, test_mode, device)
            metrics = evaluate(model, test_loader, criterion, device)
            measured[(train_mode, test_mode)] = {
                "config_id": config_id,
                "display_name": definition.display_name,
                "groups": ";".join(groups_for_configuration(config_id)),
                "model_type": definition.model_type,
                "patch_size": definition.patch_size or "",
                "patch_embedding": definition.patch_embedding or "",
                "train_mode": train_mode,
                "test_mode": test_mode,
                "test_loss": metrics["loss"],
                "test_accuracy": metrics["accuracy"],
                "best_val_accuracy": training_result["best_val_accuracy"],
                "best_epoch": training_result["best_epoch"],
                "train_elapsed_seconds": training_result["elapsed_seconds"],
                "parameter_count": training_result["parameter_count"],
            }
            print(
                f"  evaluate {train_mode}->{test_mode}: "
                f"{100 * metrics['accuracy']:.2f}%"
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows: list[dict] = []
    for setting, train_mode, test_mode in SETTING_ORDER:
        rows.append(
            {
                "setting": setting,
                **measured[(train_mode, test_mode)],
            }
        )
    return rows
