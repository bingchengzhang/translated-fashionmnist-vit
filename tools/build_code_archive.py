"""Build a compact teaching-assistant submission ZIP."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path("translated-fashionmnist-submission")
PACKAGE_SOURCE = ROOT / "packaging" / "submission"
REPORT_SOURCE = ROOT / "reports" / "comparison_study.pdf"
FIGURES = (
    "training_dynamics.png",
    "model_comparison.png",
    "position_generalization.png",
    "patch_size_comparison.png",
    "patch_embedding_comparison.png",
)


def source_files() -> list[tuple[Path, Path]]:
    """Return (source, archive-relative path) pairs."""
    files: list[tuple[Path, Path]] = [
        (PACKAGE_SOURCE / "README_FIRST.md", Path("README_FIRST.md")),
        (PACKAGE_SOURCE / "VERIFY.py", Path("VERIFY.py")),
        (REPORT_SOURCE, Path("REPORT.pdf")),
        (ROOT / "requirements.txt", Path("requirements.txt")),
        (ROOT / "results" / "comparisons" / "metrics.csv", Path("RESULTS/metrics.csv")),
        (ROOT / "results" / "comparisons" / "summary.md", Path("RESULTS/summary.md")),
        (ROOT / "utils.py", Path("CODE/utils.py")),
    ]

    for name in FIGURES:
        files.append(
            (
                ROOT / "results" / "comparisons" / "figures" / name,
                Path("RESULTS/figures") / name,
            )
        )

    for directory in ("datasets", "models", "tests"):
        for path in sorted((ROOT / directory).glob("*.py")):
            files.append((path, Path("CODE") / path.relative_to(ROOT)))

    experiments_root = ROOT / "experiments"
    for path in sorted(experiments_root.glob("*.py")):
        files.append((path, Path("CODE") / path.relative_to(ROOT)))
    for path in sorted((experiments_root / "comparisons").rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".py", ".csv"}:
            files.append((path, Path("CODE") / path.relative_to(ROOT)))

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


def build_archive(output_path: Path) -> Path:
    files = source_files()
    validate_sources(files)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, relative in files:
            archive.write(source, ARCHIVE_ROOT / relative)
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
