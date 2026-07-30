"""Training and evaluation protocol for the controlled comparisons."""

from __future__ import annotations

import csv
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data import (
    TEST_POSITION_SEED_OFFSET,
    TRAIN_POSITION_SEED_OFFSET,
    VALIDATION_POSITION_SEED_OFFSET,
    TranslatedFashionMNIST,
    create_data_loader,
    limit_dataset,
    load_fashion_mnist,
    split_train_validation,
)
from ..engine import evaluate, train_epoch, validate_training_settings
from ..models import CNNClassifier, MLPClassifier, VisionTransformer
from ..models import count_trainable_parameters
from ..utils import (
    plot_history,
    resolve_device,
    save_json,
    set_seed,
    write_csv,
)

from .config import (
    EXPERIMENTS,
    SETTING_ORDER,
    ExperimentDefinition,
    groups_for_configuration,
)


@dataclass
class ProtocolConfig:
    data_dir: str = "data"
    output_dir: str = "outputs/comparison"
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

    def __post_init__(self) -> None:
        validate_training_settings(self)


def build_model(
    definition: ExperimentDefinition,
    image_size: int = 64,
    num_classes: int = 10,
) -> nn.Module:
    """Construct one architecture while keeping the protocol configuration separate."""
    if definition.model_type == "mlp":
        return MLPClassifier(image_size=image_size, num_classes=num_classes)
    if definition.model_type == "cnn":
        return CNNClassifier(num_classes=num_classes)
    if definition.model_type == "vit":
        if definition.patch_size is None or definition.patch_embedding is None:
            raise ValueError("ViT definitions require patch size and embedding type.")
        return VisionTransformer(
            image_size=image_size,
            patch_size=definition.patch_size,
            in_channels=1,
            num_classes=num_classes,
            embed_dim=128,
            depth=4,
            num_heads=4,
            mlp_dim=512,
            dropout=0.1,
            patch_embedding=definition.patch_embedding,
        )
    raise ValueError(f"Unsupported model type: {definition.model_type}")


def _base_datasets(config: ProtocolConfig):
    return load_fashion_mnist(config.data_dir, download=config.download)


def _split_train_validation(base_train: Dataset, config: ProtocolConfig):
    return split_train_validation(
        base_train,
        val_fraction=config.val_fraction,
        seed=config.seed,
        train_limit=config.limit_train_samples,
        val_limit=config.limit_val_samples,
    )


def _loader(
    dataset: Dataset,
    config: ProtocolConfig,
    device: torch.device,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    return create_data_loader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed + seed_offset,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
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
        seed=config.seed + TRAIN_POSITION_SEED_OFFSET,
    )
    val_dataset = TranslatedFashionMNIST(
        val_subset,
        canvas_size=config.canvas_size,
        mode=train_mode,
        seed=config.seed + VALIDATION_POSITION_SEED_OFFSET,
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
    base_test = limit_dataset(base_test, config.limit_test_samples)
    test_dataset = TranslatedFashionMNIST(
        base_test,
        canvas_size=config.canvas_size,
        mode=test_mode,
        seed=config.seed + TEST_POSITION_SEED_OFFSET,
    )
    return _loader(test_dataset, config, device, shuffle=False, seed_offset=13)


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
    result_path = run_dir / "training_result.json"
    history_path = run_dir / "history.csv"
    if not all(
        path.is_file()
        for path in (metadata_path, checkpoint_path, result_path, history_path)
    ):
        return False
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        expected_epochs = int(expected_metadata["protocol"]["epochs"])
        required_history_fields = {
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        }
        complete_history = (
            bool(history)
            and required_history_fields.issubset(history[0])
            and [int(row["epoch"]) for row in history]
            == list(range(1, expected_epochs + 1))
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    required_result_keys = {
        "best_epoch",
        "best_val_accuracy",
        "elapsed_seconds",
        "parameter_count",
    }
    if not required_result_keys.issubset(result):
        return False
    existing_experiment = existing.get("experiment", {})
    existing_protocol = dict(existing_experiment.get("protocol", {}))
    expected_protocol = dict(expected_metadata.get("protocol", {}))
    existing_protocol.pop("output_dir", None)
    expected_protocol.pop("output_dir", None)
    existing_experiment = {
        **existing_experiment,
        "protocol": existing_protocol,
    }
    portable_expected = {
        **expected_metadata,
        "protocol": expected_protocol,
    }
    return complete_history and existing_experiment == portable_expected


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
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state"])
            result = json.loads(
                (run_dir / "training_result.json").read_text(encoding="utf-8")
            )
        except (
            OSError,
            RuntimeError,
            KeyError,
            EOFError,
            ValueError,
            pickle.UnpicklingError,
            json.JSONDecodeError,
        ) as error:
            print(f"Ignoring incomplete cached run: {error}")
        else:
            result["checkpoint"] = str(checkpoint_path)
            save_json(result, run_dir / "training_result.json")
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata["experiment"] = expected_metadata
            save_json(metadata, run_dir / "metadata.json")
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
        train_metrics = train_epoch(
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
