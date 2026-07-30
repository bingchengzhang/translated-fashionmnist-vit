"""One-command validation for the teaching-assistant submission package."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "CODE"
RESULTS = ROOT / "RESULTS"
EXPECTED = {
    "mlp": (0.7189, 0.8957, 0.7197, 0.1399),
    "cnn": (0.9235, 0.9314, 0.9287, 0.3540),
    "vit_p16_conv": (0.8007, 0.8887, 0.8097, 0.1691),
    "vit_p8_conv": (0.8040, 0.8827, 0.8113, 0.1881),
    "vit_p4_conv": (0.7528, 0.8722, 0.7497, 0.1702),
    "vit_p16_linear": (0.7967, 0.8880, 0.8048, 0.1630),
}


def check_layout() -> None:
    required = (
        ROOT / "REPORT.pdf",
        ROOT / "requirements.txt",
        RESULTS / "metrics.csv",
        RESULTS / "summary.md",
        CODE / "utils.py",
        CODE / "datasets" / "translated_fmnist.py",
        CODE / "models" / "vit.py",
        CODE / "experiments" / "comparisons" / "run.py",
        CODE / "tests" / "test_comparisons.py",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("缺少提交文件：" + "、".join(missing))

    forbidden_directories = {".git", ".ipynb_checkpoints", "data"}
    forbidden_suffixes = {".pt", ".pth", ".pyc"}
    forbidden: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts:
            continue
        if forbidden_directories.intersection(relative.parts):
            forbidden.append(str(relative))
        if path.is_file() and path.suffix.lower() in forbidden_suffixes:
            forbidden.append(str(relative))
    if forbidden:
        raise RuntimeError("提交包包含不应提交的文件：" + "、".join(sorted(set(forbidden))))


def check_metrics() -> None:
    with (RESULTS / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 24:
        raise RuntimeError(f"metrics.csv 应有 24 行，实际为 {len(rows)} 行")

    measured: dict[str, dict[int, float]] = {}
    for row in rows:
        measured.setdefault(row["config_id"], {})[int(row["setting"])] = float(
            row["test_accuracy"]
        )
    if set(measured) != set(EXPECTED):
        raise RuntimeError("metrics.csv 的配置集合与正式实验不一致")

    for config_id, expected_values in EXPECTED.items():
        for setting, expected in enumerate(expected_values, start=1):
            actual = measured[config_id].get(setting)
            if actual is None or abs(actual - expected) > 1e-9:
                raise RuntimeError(
                    f"{config_id} setting {setting} 指标不一致："
                    f"expected={expected}, actual={actual}"
                )


def run_code_checks() -> None:
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=CODE,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "experiments.comparisons.run", "--help"],
        cwd=CODE,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main() -> None:
    check_layout()
    print("[1/3] 文件结构：通过")
    check_metrics()
    print("[2/3] 正式指标：通过")
    run_code_checks()
    print("[3/3] 代码测试与命令行入口：通过")
    print("验收完成：报告、结果和代码相互一致。")


if __name__ == "__main__":
    main()
