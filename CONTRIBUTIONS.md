# Contributions

## bc

- 建立训练/验证/测试分离的可选实验协议。
- 实现 MLP、CNN、ViT 三类模型对比。
- 实现 ViT patch size 4/8/16 对比。
- 实现 Conv2d 与 Flatten+Linear patch embedding 对比。
- 生成训练曲线、四 setting 柱状图、跨位置性能下降图和外部基线参考图。
- 整理复现命令、结果表和实验分析材料。

## Baseline reference

同学项目：
[kicious/translated-fashion-mnist-vit](https://github.com/kicious/translated-fashion-mnist-vit)。

本仓库没有修改或复制同学项目代码。外部对照仅使用该仓库 commit
`943fa7b68730bc8ea7786bb41c7b8dc1d488883a` 已公开的四项汇总指标，
来源记录在 `optional_experiments_vs_teammate/teammate_baseline.csv`。
