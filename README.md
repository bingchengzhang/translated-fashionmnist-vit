# Position-Variable FashionMNIST with ViT

短学期大作业：使用带原始绝对位置编码的 Vision Transformer，研究服饰位置变化对 FashionMNIST 分类的影响。

## 实验设计

- **A（random）**：`28x28` 服饰随机放置在 `64x64` 黑色画布中。
- **B（center）**：`28x28` 服饰固定放置在 `64x64` 画布中央。
- 默认模型：patch size 8、embedding dimension 128、4 层 Transformer Encoder、4 个注意力头。
- 训练集内部按 9:1 划分训练/验证集，验证集负责选择最佳 checkpoint。
- FashionMNIST 官方测试集只用于最终评价。

为了避免重复训练，项目训练两个模型，并分别在 A/B 测试集上评价：

| 设置 | 训练分布 | 测试分布 |
|---|---|---|
| 1 | A | A |
| 2 | B | B |
| 3 | A | B |
| 4 | B | A |

设置编号与课程材料保持一致。实现只训练两个模型，每个模型分别在 A/B 测试集评价。

## 项目结构

```text
.
├── datasets/
│   └── translated_fmnist.py  # 动态生成 A/B 位置数据
├── models/
│   └── vit.py                # ViT 与两种 patch embedding
├── tests/
│   └── test_smoke.py         # 数据和模型冒烟测试
├── train.py                  # 单个训练模式
├── run_experiments.py        # 两次训练、四组最终评价
├── visualize_data.py         # A/B 数据可视化与坐标检查
├── utils.py                  # 指标、随机种子、checkpoint 和绘图
└── 训练循环.ipynb            # 原始课堂 MLP Notebook（保留）
```

## 环境

WSL 中已有环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch101
cd /home/bccc/dl-course
```

如需补齐绘图库：

```bash
python -m pip install matplotlib
```

当前项目不预生成大体积 `.pt` 文件。A 模式的位置由 `seed + sample index` 确定，因此不依赖 DataLoader 顺序并且可以复现。

## 快速验证

先检查数据与模型：

```bash
python -m unittest discover -s tests -v
python visualize_data.py
```

再用少量样本验证完整流程：

```bash
python run_experiments.py \
  --epochs 1 \
  --limit-train-samples 512 \
  --limit-val-samples 128 \
  --limit-test-samples 256 \
  --num-workers 0 \
  --output-dir outputs/smoke
```

## 正式运行四组实验

```bash
python run_experiments.py \
  --epochs 15 \
  --batch-size 128 \
  --canvas-size 64 \
  --patch-size 8 \
  --output-dir outputs/four_settings
```

运行过程中会训练：

1. A 训练集模型，以 A 验证集选择最佳参数；
2. B 训练集模型，以 B 验证集选择最佳参数。

随后两个最佳模型都会在 A、B 官方测试集上评价。

## 输出文件

```text
outputs/four_settings/
├── train_A/
│   ├── best.pt
│   ├── curves.png
│   ├── history.csv
│   ├── metadata.json
│   └── training_result.json
├── train_B/
│   └── ...
├── summary.csv
├── summary.json
└── summary.png
```

实验报告可以直接使用：

- `metadata.json`：模型与训练超参数；
- `history.csv`、`curves.png`：训练过程；
- `summary.csv`、`summary.png`：四组对照结果；
- 最佳 checkpoint：复现实验或继续分析。

## 首轮正式结果

使用默认配置（64x64 画布、patch size 8、15 epochs、seed 42）得到：

| 设置 | 训练 | 测试 | Test accuracy |
|---|---|---|---:|
| 1 | A | A | 79.00% |
| 2 | B | B | 87.83% |
| 3 | A | B | 79.87% |
| 4 | B | A | 25.66% |

随机位置训练获得了较好的跨位置泛化能力；只在中心位置训练的模型则明显依赖训练时的位置分布。完整记录与初步分析见 [RESULTS.md](RESULTS.md)。

![Four-setting accuracy](assets/four_settings_summary.png)

## 常用参数

```bash
python run_experiments.py --help
```

- `--patch-size 8`：可改为 4 或 16 做扩展实验。
- `--patch-embedding conv|linear`：比较卷积切块与直接 flatten。
- `--resample-train-positions`：A 训练集每个 epoch 重新生成可复现位置。
- `--seed 42`：控制数据划分、位置和模型初始化。
- `--no-amp`：关闭 CUDA 混合精度。
- `--deterministic`：优先采用确定性算子，可能降低速度。
