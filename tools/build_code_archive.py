"""Build a clean, structured ZIP of the comparison-study code and records."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "translated-fashionmnist-comparison-code"
README_SOURCE = ROOT / "packaging" / "comparison_code" / "README.md"


def source_files() -> list[tuple[Path, Path]]:
    """Return (source, archive-relative path) pairs."""
    files: list[tuple[Path, Path]] = [
        (README_SOURCE, Path("README.md")),
        (ROOT / "requirements.txt", Path("requirements.txt")),
        (ROOT / "utils.py", Path("utils.py")),
        (
            ROOT / "tools" / "build_code_archive.py",
            Path("tools") / "build_code_archive.py",
        ),
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
        if path.suffix in {".py", ".md", ".csv"}:
            files.append((path, path.relative_to(ROOT)))

    results_root = ROOT / "results" / "comparisons"
    for name in ("metrics.csv", "manifest.json", "summary.md"):
        path = results_root / name
        files.append((path, path.relative_to(ROOT)))
    for path in sorted((results_root / "figures").glob("*.png")):
        files.append((path, path.relative_to(ROOT)))
    for path in sorted((results_root / "runs").rglob("*")):
        if path.is_file() and path.suffix in {".csv", ".json"}:
            files.append((path, path.relative_to(ROOT)))

    return files


def validate_sources(files: list[tuple[Path, Path]]) -> None:
    forbidden_names = {".git", "__pycache__", ".ipynb_checkpoints", "data"}
    forbidden_suffixes = {".pt", ".pth", ".pyc", ".pdf"}
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
            archive.write(source, Path(ARCHIVE_ROOT) / relative)
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / f"{ARCHIVE_ROOT}.zip",
        help="Destination ZIP path.",
    )
    return parser


if __name__ == "__main__":
    arguments = create_parser().parse_args()
    print(build_archive(arguments.output))
