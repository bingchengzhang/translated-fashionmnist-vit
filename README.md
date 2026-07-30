# Position-Variable FashionMNIST Classification

短学期课程项目。研究对象是服饰在画布中的位置变化对图像分类模型的影响。

原始 FashionMNIST 图像为 `28×28`。实验将其放入 `64×64` 黑色画布，构造
两种位置分布：

- **A：随机位置**。每张图像在画布范围内独立平移。
- **B：固定位置**。每张图像均位于画布中心。

核心模型为带可学习绝对位置编码的 Vision Transformer。四项评价设置为
A→A、B→B、A→B 和 B→A，其中箭头左侧表示训练分布，右侧表示测试分布。

## 仓库内容

```text
.
├── datasets/                         # 位置可变数据集
├── models/                           # ViT
├── optional_experiments_vs_teammate/ # bc 负责的可选实验与外部基线对比
├── tests/                            # 数据、模型和实验配置测试
├── train.py                          # 单模型训练
├── run_experiments.py                # 基础四 setting 实验
├── visualize_data.py                 # A/B 数据检查
├── RESULTS.md                        # 基础实验记录
└── CONTRIBUTIONS.md                  # 分工与来源说明
```

## 基础实验

默认配置：

- ViT：embedding dimension 128，4 层 encoder，4 个注意力头；
- 画布 `64×64`，patch size 8；
- AdamW，15 epochs，seed 42；
- FashionMNIST 官方训练集按 90%/10% 划分训练集和验证集；
- 最佳 checkpoint 由验证集选择，官方测试集仅用于最终评价。

| Setting | Train | Test | Accuracy |
|---:|:---:|:---:|---:|
| 1 | A | A | 79.00% |
| 2 | B | B | 87.83% |
| 3 | A | B | 79.87% |
| 4 | B | A | 25.66% |

随机位置训练能够覆盖居中测试分布；只在中心位置训练时，模型对随机平移的
泛化明显不足。完整配置和训练曲线见 [RESULTS.md](RESULTS.md)。

![Four-setting baseline](assets/four_settings_summary.png)

## bc：可选实验与同学项目对比

可选实验位于
[optional_experiments_vs_teammate](optional_experiments_vs_teammate/README.md)，
包含三组对比：

1. MLP、CNN、ViT；
2. ViT patch size 4、8、16；
3. Conv2d 与 Flatten+Linear patch embedding。

所有配置采用同一数据划分和评价协议。训练结果、图表和可直接用于报告的结果
表保存在该目录中。外部参考来自
[kicious/translated-fashion-mnist-vit](https://github.com/kicious/translated-fashion-mnist-vit)
的公开训练记录；来源 commit 和指标口径均在目录内注明。

## 环境

当前 WSL 环境：

- Python 3.12
- PyTorch 2.11
- CUDA 12.8
- NVIDIA GeForce RTX 5070 Laptop GPU

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch101
cd /home/bccc/dl-course
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 运行基础实验

检查数据和模型：

```bash
python -m unittest discover -s tests -v
python visualize_data.py
```

运行基础四 setting：

```bash
python run_experiments.py \
  --epochs 15 \
  --batch-size 128 \
  --canvas-size 64 \
  --patch-size 8 \
  --output-dir outputs/four_settings
```

运行 bc 的全部可选实验：

```bash
python -m optional_experiments_vs_teammate.run_comparisons \
  --groups all \
  --epochs 15 \
  --batch-size 64
```

详细参数、输出文件和口径说明见
[可选实验说明](optional_experiments_vs_teammate/README.md)。

## 数据与版本控制

原始数据、生成数据、checkpoint 和临时输出不提交到 Git。仓库保留代码、
配置、汇总指标和报告所需图表，以便复核实验结论，同时避免将可再生成的大文件
写入版本历史。

## 分工

bc 负责可选实验的设计、代码、训练、可视化和文档。基础实现与外部对照来源见
[CONTRIBUTIONS.md](CONTRIBUTIONS.md)。
