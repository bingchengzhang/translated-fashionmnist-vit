# Comparison report

`comparison_study.pdf` 是三组对比实验的最终报告，不包含个人信息或小组基础实验。

在 Windows 主机上使用工作区提供的 Python 与 ReportLab 生成：

```powershell
python reports/build_comparison_report.py
```

报告中的数值和图表分别来自 `results/comparisons/metrics.csv` 与
`results/comparisons/figures/`；关键代码附录从最终仓库源码读取，避免报告与代码版本不一致。
