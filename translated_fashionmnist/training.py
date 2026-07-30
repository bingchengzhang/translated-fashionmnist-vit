"""Train one ViT on position mode A or B."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets as tv_datasets
from torchvision import transforms

from .data import TranslatedFashionMNIST
from .models import VisionTransformer, count_trainable_parameters
from .common import (
    AverageMeter,
    plot_history,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_worker,
    serializable_config,
    set_seed,
    system_summary,
    write_csv,
)


def add_training_arguments(
    parser: argparse.ArgumentParser,
    include_train_mode: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/single_run")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    if include_train_mode:
        parser.add_argument("--train-mode", choices=["A", "B"], default="A")
        parser.add_argument("--val-mode", choices=["A", "B"], default=None)

    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--patch-embedding", choices=["conv", "linear"], default="conv")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resample-train-positions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate new deterministic A positions at every epoch.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer deterministic kernels (slower on some GPUs).",
    )

    # Useful for fast smoke tests; zero means use the complete split.
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-val-samples", type=int, default=0)
    parser.add_argument("--limit-test-samples", type=int, default=0)
    return parser


def create_parser() -> argparse.ArgumentParser:
    return add_training_arguments(
        argparse.ArgumentParser(description=__doc__),
        include_train_mode=True,
    )


def _limit_dataset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def create_base_datasets(args: argparse.Namespace | SimpleNamespace):
    transform = transforms.ToTensor()
    train_base = tv_datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=args.download,
        transform=transform,
    )
    test_base = tv_datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        download=args.download,
        transform=transform,
    )
    return train_base, test_base


def create_train_val_loaders(args: argparse.Namespace | SimpleNamespace):
    train_base, _ = create_base_datasets(args)
    val_size = round(len(train_base) * args.val_fraction)
    train_size = len(train_base) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(
        train_base,
        [train_size, val_size],
        generator=split_generator,
    )
    train_subset = _limit_dataset(train_subset, args.limit_train_samples)
    val_subset = _limit_dataset(val_subset, args.limit_val_samples)

    val_mode = args.val_mode or args.train_mode
    train_dataset = TranslatedFashionMNIST(
        train_subset,
        canvas_size=args.canvas_size,
        mode=args.train_mode,
        seed=args.seed + 100,
        resample_each_epoch=args.resample_train_positions,
    )
    val_dataset = TranslatedFashionMNIST(
        val_subset,
        canvas_size=args.canvas_size,
        mode=val_mode,
        seed=args.seed + 200,
    )

    loader_generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def create_test_loader(
    args: argparse.Namespace | SimpleNamespace,
    mode: str,
) -> DataLoader:
    _, test_base = create_base_datasets(args)
    test_base = _limit_dataset(test_base, args.limit_test_samples)
    test_dataset = TranslatedFashionMNIST(
        test_base,
        canvas_size=args.canvas_size,
        mode=mode,
        seed=args.seed + 300,
    )
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )


def create_model(args: argparse.Namespace | SimpleNamespace) -> VisionTransformer:
    return VisionTransformer(
        image_size=args.canvas_size,
        patch_size=args.patch_size,
        in_channels=1,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        patch_embedding=args.patch_embedding,
    )


def train_one_epoch(
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


def run_training(args: argparse.Namespace | SimpleNamespace):
    if args.val_mode is None:
        args.val_mode = args.train_mode
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_train_val_loaders(args)
    model = create_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    config = serializable_config(args)
    metadata = {
        "config": config,
        "system": system_summary(device),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "trainable_parameters": count_trainable_parameters(model),
    }
    save_json(metadata, output_dir / "metadata.json")
    print(
        f"Training mode {args.train_mode} on {device} | "
        f"{metadata['trainable_parameters']:,} parameters"
    )

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_path = output_dir / "best.pt"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)
        train_metrics = train_one_epoch(
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
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {row['train_loss']:.4f}, acc {100 * row['train_accuracy']:.2f}% | "
            f"val loss {row['val_loss']:.4f}, acc {100 * row['val_accuracy']:.2f}%"
        )

        if row["val_accuracy"] > best_accuracy:
            best_accuracy = row["val_accuracy"]
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_accuracy,
                config,
            )

    write_csv(history, output_dir / "history.csv")
    plot_history(history, output_dir / "curves.png")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    elapsed = time.time() - started
    result = {
        "train_mode": args.train_mode,
        "val_mode": args.val_mode,
        "best_epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["best_val_accuracy"],
        "elapsed_seconds": elapsed,
        "checkpoint": str(best_path),
    }
    save_json(result, output_dir / "training_result.json")
    print(
        f"Best validation accuracy: {100 * best_accuracy:.2f}% | "
        f"elapsed: {elapsed / 60:.1f} min"
    )
    return model, device, criterion, result


def main() -> None:
    args = create_parser().parse_args()
    model, device, criterion, result = run_training(args)
    test_loader = create_test_loader(args, args.train_mode)
    test_metrics = evaluate(model, test_loader, criterion, device)
    result.update(
        {
            "test_mode": args.train_mode,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
    )
    save_json(result, Path(args.output_dir) / "training_result.json")
    print(f"Test accuracy ({args.train_mode}): {100 * test_metrics['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
