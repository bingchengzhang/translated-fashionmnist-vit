# 位置可变 FashionMNIST：三组对比实验代码

本代码包对应课程报告中的三组扩展实验：

1. MLP、CNN 与 Vision Transformer 的结构比较；
2. ViT patch size 4、8、16 的尺度比较；
3. Conv2d 与 Flatten+Linear patch embedding 的实现比较。

代码包保留正式实验的配置、测试、汇总指标、训练历史和绘图结果。原始
FashionMNIST 数据与模型 checkpoint 未包含在压缩包中。

## 目录

```text
translated-fashionmnist-comparison-code/
├── datasets/                    # 位置可变数据集
├── models/                      # ViT 与两种 patch embedding
├── experiments/comparisons/    # 对比配置、模型、协议、入口和绘图
├── tests/                       # 数据、模型与对比实验测试
├── results/comparisons/         # 正式指标、训练历史和图表
├── tools/                       # 代码包构建脚本
├── requirements.txt
├── utils.py
└── README.md
```

`results/comparisons/runs/` 只包含 CSV 和 JSON 记录，不含 `best.pt`。
`experiments/comparisons/references/teammate_vit.csv` 是同学项目的外部参考
指标；由于模型选择协议不同，该数据不作严格等价比较。

## 环境

建议使用 Python 3.12。安装依赖：

```bash
python -m pip install -r requirements.txt
```

先运行测试和命令行检查：

```bash
python -m unittest discover -s tests -v
python -m experiments.comparisons.run --help
```

正式实验需要 FashionMNIST 数据。首次运行可使用 `--download`：

```bash
python -m experiments.comparisons.run \
  --groups all \
  --epochs 15 \
  --batch-size 64 \
  --num-workers 4 \
  --download \
  --output-dir results/comparisons
```

每个配置只训练 A、B 两个模型，再分别在 A、B 测试集上评价。最佳
checkpoint 由验证集准确率选择，测试集不参与模型选择。完整参数见
`results/comparisons/manifest.json`。

## 结果文件

- `metrics.csv`：六个配置在四种设置下的最终测试指标；
- `manifest.json`：正式实验参数与运行环境；
- `summary.md`：结果表和位置泛化摘要；
- `figures/`：报告使用的图表；
- `runs/`：每次训练的历史、元数据和最佳验证记录。

正式结果只包含一个随机种子。小于 1 个百分点的差异不应解释为稳定优势；
如需统计结论，应增加多个随机种子并报告均值与标准差。
