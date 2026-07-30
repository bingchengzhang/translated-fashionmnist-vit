# 助教验收入口

本压缩包只包含三组扩展对比实验。建议按以下顺序检查：

1. 打开 `REPORT.pdf` 阅读实验设计、结果与分析；
2. 查看 `RESULTS/metrics.csv` 核对全部 24 项最终指标；
3. 运行 `python VERIFY.py` 检查代码、结果文件和 8 项单元测试；
4. 如需重新训练，按本文末尾命令运行。

原始 FashionMNIST 数据与模型 checkpoint 未放入提交包。

## 实验对象

原始 28×28 FashionMNIST 图像被放入 64×64 画布：

- A：每张图像随机平移；
- B：每张图像固定居中。

评价设置为 A→A、B→B、A→B、B→A，箭头左侧为训练分布，右侧为测试分布。

## 三组比较

| 比较 | 控制变量 | 配置 |
|---|---|---|
| 模型结构 | 训练协议与数据划分 | MLP、CNN、ViT patch 16 |
| Patch 尺度 | ViT 主体结构 | patch size 4、8、16 |
| Patch Embedding | patch size 与 ViT 主体 | Conv2d、Flatten+Linear |

## 最终结果

| 配置 | A→A | B→B | A→B | B→A |
|---|---:|---:|---:|---:|
| MLP | 71.89% | 89.57% | 71.97% | 13.99% |
| CNN | 92.35% | 93.14% | 92.87% | 35.40% |
| ViT, patch 16 | 80.07% | 88.87% | 80.97% | 16.91% |
| ViT, patch 8 | 80.40% | 88.27% | 81.13% | 18.81% |
| ViT, patch 4 | 75.28% | 87.22% | 74.97% | 17.02% |
| ViT, Linear patch 16 | 79.67% | 88.80% | 80.48% | 16.30% |

正式实验使用 seed 42。小于 1 个百分点的差异不解释为稳定优势。

## 目录

```text
translated-fashionmnist-submission/
├── README_FIRST.md       # 本文件
├── REPORT.pdf            # 最终实验报告
├── VERIFY.py             # 一键验收
├── requirements.txt      # Python 依赖
├── RESULTS/
│   ├── metrics.csv       # 24 项最终指标
│   ├── summary.md        # 文本结果摘要
│   └── figures/          # 报告中的 5 幅核心图
└── CODE/
    ├── datasets/         # 位置可变数据集
    ├── models/           # ViT 与 Patch Embedding
    ├── experiments/      # 三组比较与统一实验协议
    ├── tests/            # 8 项单元测试
    └── utils.py
```

## 快速验收

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行：

```bash
python VERIFY.py
```

验收脚本会检查：

- 报告、指标和核心源码是否齐全；
- 压缩包中是否混入 checkpoint、缓存或数据目录；
- `metrics.csv` 是否包含预期的 6 个配置和 4 个设置；
- 保存的正式准确率是否与报告一致；
- 8 项单元测试是否通过；
- 实验命令行入口是否可用。

## 重新训练

首次运行需要下载 FashionMNIST：

```bash
cd CODE
python -m experiments.comparisons.run \
  --groups all \
  --epochs 15 \
  --batch-size 64 \
  --num-workers 4 \
  --download \
  --output-dir ../RESULTS/reproduced
```

每个配置只训练 A、B 两个模型，再分别在 A、B 测试集上评价。最佳 checkpoint
由验证集选择，官方测试集不参与模型选择。

同学项目公开指标保存在
`CODE/experiments/comparisons/references/teammate_vit.csv`，只用于核对结果量级；
双方 checkpoint 选择协议不同，不作严格等价比较。
