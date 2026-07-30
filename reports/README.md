# Report

`comparison_study.pdf` is the five-page English report for the optional
comparison study. It emphasizes experimental process, learning curves,
comparison results, and analysis, and contains no personal information.

Build it from the repository root:

```bash
python reports/build_comparison_report.py
```

The script reads `results/metrics.csv` and `results/training_history.csv`, then
draws all charts as vectors. Source Serif 4 and Source Sans 3 are embedded from
`reports/fonts/`; both families are distributed under the SIL Open Font License.
