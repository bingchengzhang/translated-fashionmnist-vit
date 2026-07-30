"""Build one compact teaching-assistant submission archive."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path("translated-fashionmnist-submission")
SOURCE_ROOT = Path("translated-fashionmnist-source")
REPORT_SOURCE = ROOT / "reports" / "comparison_study.pdf"
METRICS_SOURCE = ROOT / "results" / "metrics.csv"

SUBMISSION_README = """# Submission guide

This archive contains the controlled comparison study.

Read the files in this order:

1. REPORT.pdf - five-page experimental report.
2. RESULTS.csv - all 24 formal test measurements.
3. SOURCE_CODE.zip - unified Python package and tests.

The study compares MLP, CNN, and ViT; ViT patch sizes 4, 8, and 16; and Conv2d
against Flatten + Linear patch embedding. Formal runs use seed 42 and
validation-based checkpoint selection. Results come from one training and
placement seed, so small numerical differences are not significance claims.

To reproduce, extract SOURCE_CODE.zip and run:

    python -m pip install -r requirements-lock.txt
    python -m unittest discover -s tests -v
    python -m translated_fashionmnist.experiments.compare --groups all --download

Datasets, checkpoints, caches, duplicate figures, and per-run folders are
intentionally excluded.
"""

SOURCE_README = """# Source code

The baseline and all controlled comparisons share one data pipeline, training
engine, and experiment registry:

    translated_fashionmnist/
        data.py             loading, fixed split, loaders, and A/B canvases
        engine.py           shared training and evaluation loops
        models.py           MLP, CNN, ViT, and patch embeddings
        training.py         single-model ViT entry point
        utils.py
        visualize.py
        experiments/
            baseline.py     four-setting ViT runner
            compare.py      three controlled comparison groups
            config.py       experiment definitions
            protocol.py     common comparison protocol
            plots.py        comparison figures

Run the tests:

    python -m unittest discover -s tests -v

Run all comparisons:

    python -m translated_fashionmnist.experiments.compare --groups all --download
"""


def code_files() -> list[tuple[Path, Path]]:
    """Return files for the nested modular source archive."""
    files: list[tuple[Path, Path]] = [
        (ROOT / "requirements.txt", Path("requirements.txt")),
        (ROOT / "requirements-lock.txt", Path("requirements-lock.txt")),
    ]
    for path in sorted((ROOT / "translated_fashionmnist").rglob("*")):
        if path.is_file() and path.suffix == ".py":
            files.append((path, path.relative_to(ROOT)))
    for path in sorted((ROOT / "tests").glob("*.py")):
        files.append((path, path.relative_to(ROOT)))
    return files


def validate_sources(files: list[tuple[Path, Path]]) -> None:
    forbidden_names = {".git", "__pycache__", ".ipynb_checkpoints", "data"}
    forbidden_suffixes = {".pt", ".pth", ".pyc"}
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError("Missing package sources:\n" + "\n".join(missing))
    for _, relative in files:
        if forbidden_names.intersection(relative.parts):
            raise ValueError(f"Forbidden directory in package: {relative}")
        if relative.suffix.lower() in forbidden_suffixes:
            raise ValueError(f"Forbidden file type in package: {relative}")


def validate_metrics(path: Path) -> None:
    """Require one complete 4-setting record for every formal configuration."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"config_id", "train_mode", "test_mode", "test_accuracy"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Metrics file is empty or missing required columns.")
    keys = {
        (row["config_id"], row["train_mode"], row["test_mode"])
        for row in rows
    }
    configurations = {row["config_id"] for row in rows}
    expected_settings = {("A", "A"), ("B", "B"), ("A", "B"), ("B", "A")}
    complete = all(
        {
            (row["train_mode"], row["test_mode"])
            for row in rows
            if row["config_id"] == config_id
        }
        == expected_settings
        for config_id in configurations
    )
    if len(rows) != 24 or len(keys) != 24 or len(configurations) != 6 or not complete:
        raise ValueError("Expected 24 unique formal measurements.")


def build_source_archive(files: list[tuple[Path, Path]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(str(SOURCE_ROOT / "README.md"), SOURCE_README)
        for source, relative in files:
            archive.write(source, SOURCE_ROOT / relative)
    return buffer.getvalue()


def build_archive(output_path: Path) -> Path:
    files = code_files()
    validate_sources(files)
    for path in (REPORT_SOURCE, METRICS_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(f"Missing submission source: {path}")
    validate_metrics(METRICS_SOURCE)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(str(ARCHIVE_ROOT / "README.md"), SUBMISSION_README)
        archive.write(REPORT_SOURCE, ARCHIVE_ROOT / "REPORT.pdf")
        archive.write(METRICS_SOURCE, ARCHIVE_ROOT / "RESULTS.csv")
        archive.writestr(
            str(ARCHIVE_ROOT / "SOURCE_CODE.zip"),
            build_source_archive(files),
        )
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "translated-fashionmnist-submission.zip",
        help="Destination ZIP path.",
    )
    return parser


if __name__ == "__main__":
    arguments = create_parser().parse_args()
    print(build_archive(arguments.output))
