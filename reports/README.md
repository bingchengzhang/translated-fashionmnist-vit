# Comparison report

`comparison_study.pdf` 是三组对比实验的最终报告，不包含个人信息或小组基础实验。

在 Windows 主机上使用工作区提供的 Python 与 ReportLab 生成：

```powershell
python reports/build_comparison_report.py
```

报告中的数值和图表分别来自 `results/comparisons/metrics.csv` 与
`results/comparisons/figures/`。报告不附大段源码；报告、结果和代码通过
`tools/build_code_archive.py` 合并为助教验收用提交包。
