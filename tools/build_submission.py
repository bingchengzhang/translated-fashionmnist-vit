"""Build one compact teaching-assistant submission archive."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path("translated-fashionmnist-submission")
SOURCE_ROOT = Path("translated-fashionmnist-source")
PACKAGE_SOURCE = ROOT / "packaging" / "submission"
REPORT_SOURCE = ROOT / "reports" / "comparison_study.pdf"


def code_files() -> list[tuple[Path, Path]]:
    """Return source files for the nested modular code archive."""
    files: list[tuple[Path, Path]] = [
        (ROOT / "experiments" / "comparisons" / "README.md", Path("README.md")),
        (ROOT / "requirements.txt", Path("requirements.txt")),
        (ROOT / "utils.py", Path("utils.py")),
    ]

    for directory in ("datasets", "models", "tests"):
        for path in sorted((ROOT / directory).glob("*.py")):
            files.append((path, path.relative_to(ROOT)))

    experiments_root = ROOT / "experiments"
    for path in sorted(experiments_root.glob("*.py")):
        files.append((path, path.relative_to(ROOT)))
    for path in sorted((experiments_root / "comparisons").rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".py", ".csv"}:
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


def build_source_archive(files: list[tuple[Path, Path]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, relative in files:
            archive.write(source, SOURCE_ROOT / relative)
    return buffer.getvalue()


def build_archive(output_path: Path) -> Path:
    files = code_files()
    validate_sources(files)
    outer_sources = (
        PACKAGE_SOURCE / "README.md",
        REPORT_SOURCE,
        ROOT / "results" / "comparisons" / "metrics.csv",
    )
    missing = [str(path) for path in outer_sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing submission sources:\n" + "\n".join(missing))

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(PACKAGE_SOURCE / "README.md", ARCHIVE_ROOT / "README.md")
        archive.write(REPORT_SOURCE, ARCHIVE_ROOT / "REPORT.pdf")
        archive.write(
            ROOT / "results" / "comparisons" / "metrics.csv",
            ARCHIVE_ROOT / "RESULTS.csv",
        )
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
