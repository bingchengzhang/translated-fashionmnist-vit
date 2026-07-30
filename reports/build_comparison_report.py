"""Generate the concise Chinese comparison report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "comparisons"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_PATH = ROOT / "reports" / "comparison_study.pdf"

INK = colors.HexColor("#172231")
MUTED = colors.HexColor("#596473")
ACCENT = colors.HexColor("#244A73")
RULE = colors.HexColor("#C7CFD9")
PALE = colors.HexColor("#F2F5F8")
WHITE = colors.white


def register_fonts() -> None:
    candidates = [
        (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\msyhbd.ttc")),
        (Path("/mnt/c/Windows/Fonts/msyh.ttc"), Path("/mnt/c/Windows/Fonts/msyhbd.ttc")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ReportCN", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportCN-Bold", str(bold)))
            return
    raise FileNotFoundError("Microsoft YaHei fonts are required.")


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="ReportCN-Bold",
            fontSize=20,
            leading=29,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=5 * mm,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="ReportCN",
            fontSize=9.2,
            leading=14,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=7 * mm,
        ),
        "Heading1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="ReportCN-Bold",
            fontSize=15,
            leading=21,
            textColor=ACCENT,
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "Heading2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName="ReportCN-Bold",
            fontSize=11.2,
            leading=17,
            textColor=INK,
            spaceBefore=3 * mm,
            spaceAfter=1.5 * mm,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="ReportCN",
            fontSize=9.4,
            leading=15.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            firstLineIndent=18.8,
            wordWrap="CJK",
            spaceAfter=2.2 * mm,
        ),
        "BodyNoIndent": ParagraphStyle(
            "BodyNoIndent",
            parent=base["BodyText"],
            fontName="ReportCN",
            fontSize=9.4,
            leading=15.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            wordWrap="CJK",
            spaceAfter=2.2 * mm,
        ),
        "Abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="ReportCN",
            fontSize=9.1,
            leading=14.6,
            textColor=INK,
            alignment=TA_JUSTIFY,
            firstLineIndent=18.2,
            leftIndent=7 * mm,
            rightIndent=7 * mm,
            wordWrap="CJK",
            spaceAfter=2 * mm,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="ReportCN",
            fontSize=9.2,
            leading=14.8,
            textColor=INK,
            leftIndent=6 * mm,
            firstLineIndent=-4 * mm,
            wordWrap="CJK",
            spaceAfter=1.3 * mm,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="ReportCN",
            fontSize=8,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            wordWrap="CJK",
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
        ),
        "Table": ParagraphStyle(
            "Table",
            parent=base["Normal"],
            fontName="ReportCN",
            fontSize=7.6,
            leading=10.5,
            textColor=INK,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "TableLeft": ParagraphStyle(
            "TableLeft",
            parent=base["Normal"],
            fontName="ReportCN",
            fontSize=7.6,
            leading=10.5,
            textColor=INK,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "Reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="ReportCN",
            fontSize=8.2,
            leading=13,
            textColor=INK,
            leftIndent=5 * mm,
            firstLineIndent=-5 * mm,
            wordWrap="CJK",
            spaceAfter=1.4 * mm,
        ),
    }


class ReportTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=21 * mm,
            rightMargin=21 * mm,
            topMargin=19 * mm,
            bottomMargin=17 * mm,
            title="位置分布变化下的 FashionMNIST 分类：模型结构、Patch 尺度与嵌入方式的对比实验",
            author="",
            subject="Translated FashionMNIST comparison study",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="content",
        )
        self.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=self.draw_page)])
        self._heading_sequence = 0

    def draw_page(self, canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.45)
        canvas.line(21 * mm, A4[1] - 13 * mm, A4[0] - 21 * mm, A4[1] - 13 * mm)
        canvas.setFont("ReportCN", 7.6)
        canvas.setFillColor(MUTED)
        if document.page > 1:
            canvas.drawString(
                21 * mm,
                A4[1] - 10.2 * mm,
                "位置分布变化下的 FashionMNIST 分类对比实验",
            )
        canvas.drawRightString(A4[0] - 21 * mm, 9.5 * mm, str(document.page))
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        if flowable.style.name not in {"Heading1", "Heading2"}:
            return
        level = 0 if flowable.style.name == "Heading1" else 1
        self._heading_sequence += 1
        key = f"heading-{self._heading_sequence}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(flowable.getPlainText(), key, level=level, closed=False)


def p(text: str, styles, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str, styles) -> Paragraph:
    return Paragraph(f"• {text}", styles["Bullet"])


def make_table(
    rows: list[list[str]],
    widths: list[float],
    styles,
    *,
    left_columns: set[int] | None = None,
) -> Table:
    left_columns = left_columns or set()
    rendered: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        output_row = []
        for column_index, value in enumerate(row):
            value = escape(str(value))
            if row_index == 0:
                value = f"<b>{value}</b>"
            style = styles["TableLeft"] if column_index in left_columns else styles["Table"]
            output_row.append(Paragraph(value, style))
        rendered.append(output_row)
    result = Table(rendered, colWidths=widths, repeatRows=1, hAlign="CENTER")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    return result


def figure(path: Path, caption: str, styles, *, max_height: float) -> KeepTogether:
    image = Image(str(path))
    max_width = 165 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return KeepTogether([image, Paragraph(caption, styles["Caption"])])


def read_metrics() -> list[dict]:
    with (RESULTS_DIR / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["setting"] = int(row["setting"])
        row["test_accuracy"] = float(row["test_accuracy"])
        row["train_elapsed_seconds"] = float(row["train_elapsed_seconds"])
        row["parameter_count"] = int(row["parameter_count"])
    return rows


def accuracy(rows: list[dict], config_id: str, setting: int) -> float:
    record = next(
        row
        for row in rows
        if row["config_id"] == config_id and row["setting"] == setting
    )
    return 100 * record["test_accuracy"]


def total_minutes(rows: list[dict], config_id: str) -> float:
    by_train_mode = {
        row["train_mode"]: row["train_elapsed_seconds"]
        for row in rows
        if row["config_id"] == config_id
    }
    return sum(by_train_mode.values()) / 60


def add_opening(story, styles) -> None:
    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph(
                "位置分布变化下的 FashionMNIST 分类：<br/>"
                "模型结构、Patch 尺度与嵌入方式的对比实验",
                styles["Title"],
            ),
            Paragraph("短学期深度学习课程扩展实验", styles["Subtitle"]),
            Paragraph("摘要", styles["Heading1"]),
            p(
                "本实验研究位置分布变化对 FashionMNIST 分类模型的影响。原始 28×28 图像被放入 "
                "64×64 画布。由此构成随机位置分布 A 与固定居中分布 B。在相同的数据划分、训练轮数和"
                "模型选择规则下，分别比较 MLP、CNN 与 Vision Transformer（ViT），比较 ViT 的 "
                "patch size 4、8、16，并比较 Conv2d 与 Flatten+Linear 两种 patch embedding。"
                "结果显示，CNN 在四项设置中均领先，B→A 为 35.40%。patch "
                "size 8 在准确率与训练成本之间较为均衡；两种 patch embedding 的四项准确率差异"
                "均不超过 0.61 个百分点。所有模型从 B→B 切换到 B→A 时均明显下降，表明固定位置"
                "训练不足以覆盖随机平移分布。由于正式实验只使用一个随机种子，本文将结果解释为"
                "受控条件下的观察，不作统计显著性判断。",
                styles,
                "Abstract",
            ),
            p(
                "<b>关键词：</b>FashionMNIST；位置泛化；卷积神经网络；Vision Transformer；"
                "Patch Embedding",
                styles,
                "Abstract",
            ),
            Paragraph("1 研究问题", styles["Heading1"]),
            p(
                "图像类别不随平移改变，但分类器未必具备稳定的平移泛化能力。课程基础任务采用 ViT "
                "处理位置可变 FashionMNIST；本实验只讨论三项扩展比较，不重复基础模型的实现过程。"
                "研究问题是：不同结构的空间归纳偏置是否影响位置泛化；更细的 patch 是否改善小图像"
                "分类；两种常用 patch embedding 实现是否产生可测的性能差异。",
                styles,
            ),
            bullet("结构比较：MLP、CNN、patch-16 ViT。", styles),
            bullet("Patch 尺度比较：在相同 ViT 主体下采用 patch size 4、8、16。", styles),
            bullet("嵌入比较：在 patch size 16 下采用 Conv2d 或 Flatten+Linear。", styles),
            p(
                "评价同时报告同分布结果 A→A、B→B 和跨分布结果 A→B、B→A。箭头左侧为训练"
                "分布，右侧为测试分布。B→A 是最具区分度的设置：模型只见过居中样本，却需要识别"
                "随机位置样本。",
                styles,
            ),
        ]
    )


def add_method(story, styles, manifest: dict) -> None:
    protocol = manifest["protocol"]
    story.extend(
        [
            PageBreak(),
            Paragraph("2 数据与实验方法", styles["Heading1"]),
            Paragraph("2.1 数据构造与评价设置", styles["Heading2"]),
            p(
                "FashionMNIST 官方训练集含 60,000 张图像，按固定随机顺序划分为 54,000 个训练"
                "样本和 6,000 个验证样本；官方测试集含 10,000 个样本。每张 28×28 灰度图像均"
                "嵌入 64×64 黑色画布。分布 A 对每个样本独立采样合法平移，分布 B 将图像固定在"
                "画布中心。类别标签与原始数据保持一致。",
                styles,
            ),
            make_table(
                [
                    ["设置", "训练分布", "测试分布", "评价含义"],
                    ["A→A", "随机位置", "随机位置", "随机位置同分布性能"],
                    ["B→B", "固定居中", "固定居中", "固定位置同分布性能"],
                    ["A→B", "随机位置", "固定居中", "随机训练对中心位置的覆盖"],
                    ["B→A", "固定居中", "随机位置", "固定训练对平移分布的泛化"],
                ],
                [22 * mm, 32 * mm, 32 * mm, 79 * mm],
                styles,
                left_columns={3},
            ),
            Paragraph("表 1　四种训练—测试设置", styles["Caption"]),
            Paragraph("2.2 训练协议", styles["Heading2"]),
            p(
                f"所有配置采用 AdamW，学习率 {protocol['learning_rate']}，权重衰减 "
                f"{protocol['weight_decay']}，batch size {protocol['batch_size']}，训练 "
                f"{protocol['epochs']} 个 epoch，随机种子 {protocol['seed']}。每个配置分别训练"
                "一个 A 模型和一个 B 模型；最佳 checkpoint 由对应分布的验证准确率选择，测试集"
                "仅用于最终评价。这样既避免四个设置各自重复训练，也避免以测试集选择 epoch。",
                styles,
            ),
            p(
                "训练启用自动混合精度，但未开启完全确定性 CUDA 算法。正式记录来自 Python "
                f"{manifest['environment']['python']}、PyTorch {manifest['environment']['torch']}、"
                f"{manifest['environment']['gpu']}。运行时间用于比较本机同一批实验的相对成本，"
                "不作为跨硬件基准。",
                styles,
            ),
            Paragraph("2.3 控制变量与模型", styles["Heading2"]),
            p(
                "模型结构比较使用 MLP、两层卷积 CNN 和 patch-16 ViT。Patch 尺度实验只改变 "
                "patch size，保持 embedding dimension、Transformer 深度和注意力头数不变。"
                "Patch embedding 实验固定 patch size 16，并保持参数量一致。不同结构的参数量"
                "未强制匹配，因此模型结构比较反映的是完整配置差异，而非纯粹的参数效率比较。",
                styles,
            ),
            p(
                "实验程序保存训练历史、最佳验证准确率、最终测试指标、配置清单和绘图数据。"
                "图表全部由保存的 CSV/JSON 记录生成，报告数值不依赖人工抄录。",
                styles,
            ),
        ]
    )


def add_overall_results(story, styles, rows: list[dict]) -> None:
    labels = [
        ("mlp", "MLP"),
        ("cnn", "CNN"),
        ("vit_p16_conv", "ViT, patch 16"),
        ("vit_p8_conv", "ViT, patch 8"),
        ("vit_p4_conv", "ViT, patch 4"),
        ("vit_p16_linear", "ViT, Linear patch 16"),
    ]
    result_rows = [["配置", "参数量", "A→A", "B→B", "A→B", "B→A"]]
    for config_id, label in labels:
        parameters = next(
            row["parameter_count"] for row in rows if row["config_id"] == config_id
        )
        result_rows.append(
            [
                label,
                f"{parameters:,}",
                *[f"{accuracy(rows, config_id, setting):.2f}%" for setting in (1, 2, 3, 4)],
            ]
        )
    story.extend(
        [
            PageBreak(),
            Paragraph("3 总体结果", styles["Heading1"]),
            p(
                "表 2 汇总六个配置的最终测试准确率。B→B 普遍高于 A→A，说明固定居中数据的"
                "变化范围更小；A→B 与 A→A 接近，说明随机位置训练通常能够覆盖中心位置。相反，"
                "B→A 对所有配置都困难，且准确率远低于对应的 B→B。",
                styles,
            ),
            make_table(
                result_rows,
                [44 * mm, 27 * mm, 23.5 * mm, 23.5 * mm, 23.5 * mm, 23.5 * mm],
                styles,
                left_columns={0},
            ),
            Paragraph("表 2　六个配置在四种位置设置下的测试准确率", styles["Caption"]),
            figure(
                FIGURES_DIR / "training_dynamics.png",
                "图 1　六种配置在 A、B 验证集上的训练过程。每条曲线对应一个独立训练模型。",
                styles,
                max_height=91 * mm,
            ),
            p(
                "验证曲线在 15 个 epoch 内整体趋于平稳，没有持续发散。B 分布的曲线通常更快达到"
                "较高准确率，这与其只包含中心位置、样本变化较少一致。A 分布同时包含类别差异和"
                "位置变化，训练准确率较低并不直接表示模型失效，而是反映任务范围扩大。后续比较"
                "因此以同一设置内的模型差异为主，不将 A 与 B 的绝对准确率简单等同。",
                styles,
            ),
        ]
    )


def add_model_comparison(story, styles, rows: list[dict]) -> None:
    mlp_drop = accuracy(rows, "mlp", 2) - accuracy(rows, "mlp", 4)
    cnn_drop = accuracy(rows, "cnn", 2) - accuracy(rows, "cnn", 4)
    vit_drop = accuracy(rows, "vit_p16_conv", 2) - accuracy(rows, "vit_p16_conv", 4)
    story.extend(
        [
            PageBreak(),
            Paragraph("4 对比一：MLP、CNN 与 ViT", styles["Heading1"]),
            figure(
                FIGURES_DIR / "model_comparison.png",
                "图 2　MLP、CNN 与 patch-16 ViT 在四种位置设置下的测试准确率。",
                styles,
                max_height=78 * mm,
            ),
            p(
                "CNN 在四项设置中均为最高：A→A 92.35%，B→B 93.14%，A→B 92.87%，"
                "B→A 35.40%。MLP 的 A→A 与 A→B 均约为 72%；patch-16 ViT 对应结果约为 "
                "80% 和 81%。在随机位置训练的两个设置中，CNN 比 patch-16 ViT 高 12 个百分点"
                "左右，差异在本次实验中具有明确的量级。",
                styles,
            ),
            figure(
                FIGURES_DIR / "position_generalization.png",
                "图 3　固定位置训练后的性能变化；下降量定义为 B→B 减 B→A。",
                styles,
                max_height=68 * mm,
            ),
            p(
                f"从 B→B 切换到 B→A 后，MLP、CNN 和 ViT 分别下降 {mlp_drop:.2f}、"
                f"{cnn_drop:.2f} 和 {vit_drop:.2f} 个百分点。CNN 的下降仍然很大，但 B→A "
                "准确率比 MLP 高 21.41 个百分点，比 patch-16 ViT 高 18.49 个百分点。"
                "局部卷积核在空间位置之间共享参数，为平移后的局部模式提供了直接复用机制；"
                "MLP 对不同像素位置使用不同连接；当前 ViT 使用可学习绝对位置编码，token 表征"
                "与训练位置关联更强。这些结构性质与观察到的相对顺序一致，但单次实验不能把"
                "全部差异唯一归因于某一个组件。",
                styles,
            ),
        ]
    )


def add_patch_size(story, styles, rows: list[dict]) -> None:
    story.extend(
        [
            PageBreak(),
            Paragraph("5 对比二：ViT Patch 尺度", styles["Heading1"]),
            figure(
                FIGURES_DIR / "patch_size_comparison.png",
                "图 4　保持 ViT 主体结构不变时，patch size 4、8、16 的测试准确率。",
                styles,
                max_height=87 * mm,
            ),
            p(
                "patch size 8 在 A→A、A→B 与 B→A 上分别得到 80.40%、81.13% 和 18.81%，"
                "是三种尺度中的最高值。patch size 16 在 B→B 上最高，为 88.87%。patch size 4 "
                "在 A→A 与 A→B 上下降到 75.28% 和 74.97%，未获得更细粒度所预期的准确率"
                "收益。三种设置的参数量接近，因此差异主要伴随 token 数和优化过程变化。",
                styles,
            ),
            Paragraph("序列长度与训练成本", styles["Heading2"]),
            p(
                "在 64×64 画布上，patch size 4、8、16 分别产生 256、64、16 个图像 token。"
                "自注意力需要建立 token 两两关系，其矩阵元素数分别为 65,536、4,096 和 256。"
                "这一理论规模不等同于端到端时间，因为数据加载、线性层和 GPU 并行效率也占用"
                "时间，但可解释细粒度 patch 的额外计算压力。",
                styles,
            ),
            p(
                f"本机训练 A、B 两个模型的累计时间分别为：patch 4 为 "
                f"{total_minutes(rows, 'vit_p4_conv'):.2f} 分钟，patch 8 为 "
                f"{total_minutes(rows, 'vit_p8_conv'):.2f} 分钟，patch 16 为 "
                f"{total_minutes(rows, 'vit_p16_conv'):.2f} 分钟。patch size 4 最慢，"
                "但准确率没有提高。当前任务中，patch size 8 避免了 patch 16 的较粗空间划分，"
                "又不承担 patch 4 的长序列成本，因而是更合理的折中。该判断只适用于当前画布、"
                "模型深度和训练预算。",
                styles,
            ),
            p(
                "更小 patch 并不必然带来更好结果。细粒度输入增加了可用空间信息，也增加了序列"
                "长度和优化难度；当训练轮数固定时，后者可能抵消前者。若要判断 patch 4 是否"
                "受训练不足影响，需要进一步增加训练预算或调整学习率，而不能只依据单一设置外推。",
                styles,
            ),
        ]
    )


def add_patch_embedding(story, styles, rows: list[dict]) -> None:
    differences = [
        abs(
            accuracy(rows, "vit_p16_conv", setting)
            - accuracy(rows, "vit_p16_linear", setting)
        )
        for setting in (1, 2, 3, 4)
    ]
    story.extend(
        [
            PageBreak(),
            Paragraph("6 对比三：Patch Embedding 实现", styles["Heading1"]),
            figure(
                FIGURES_DIR / "patch_embedding_comparison.png",
                "图 5　patch size 16 时，Conv2d 与 Flatten+Linear 的测试准确率。",
                styles,
                max_height=90 * mm,
            ),
            p(
                "Conv2d 版本在 A→A、B→B、A→B、B→A 上分别为 80.07%、88.87%、"
                "80.97%、16.91%；Flatten+Linear 版本分别为 79.67%、88.80%、80.48%、"
                "16.30%。四项绝对差异为 "
                + "、".join(f"{value:.2f}" for value in differences)
                + f" 个百分点，最大差异为 {max(differences):.2f} 个百分点。两者参数量均为 "
                "829,834，累计训练时间分别为 6.46 和 7.25 分钟。",
                styles,
            ),
            Paragraph("等价关系与结果解释", styles["Heading2"]),
            p(
                "当卷积核大小与步长都等于 patch size、patch 之间不重叠时，Conv2d 对每个 patch "
                "应用同一组卷积核。将卷积核展平后，可得到从 patch_dim 到 embed_dim 的线性"
                "映射；这与对每个展平 patch 共享同一个 Linear 层在表达形式上等价。项目测试"
                "通过重排同一组权重，验证了两种前向结果在浮点误差范围内一致。",
                styles,
            ),
            p(
                "训练结果中的小差异可能来自参数初始化后的随机优化路径、底层算子和浮点计算顺序。"
                "由于只运行一个随机种子，0.61 个百分点不足以支持稳定优劣判断。就当前非重叠 "
                "patch 设置而言，选择哪种实现应更多考虑代码接口、可读性和后续扩展需求。若采用"
                "重叠 patch、卷积前处理或不同归一化，两种方案才会不再对应同一个简单线性映射。",
                styles,
            ),
            p(
                "因此，本组实验的主要结果不是某种实现提高了准确率，而是实测结果与理论等价关系"
                "相符：在控制参数量和 patch 划分后，embedding 写法本身没有形成可辨认的性能差距。",
                styles,
            ),
        ]
    )


def add_discussion(story, styles) -> None:
    story.extend(
        [
            PageBreak(),
            Paragraph("7 讨论", styles["Heading1"]),
            Paragraph("7.1 三组结果的共同指向", styles["Heading2"]),
            p(
                "三组比较共同表明，位置泛化的主要困难不是固定位置样本本身的分类，而是训练分布"
                "是否覆盖测试时可能出现的位置。B→B 中所有模型都取得较高准确率，但这一结果不能"
                "预测 B→A。CNN 的局部参数共享改善了跨位置复用，仍无法替代随机位置训练。对 ViT "
                "而言，增加 token 数也没有自动解决位置泛化；patch size 4 的结果反而下降。"
            , styles),
            p(
                "A→B 通常接近或略高于 A→A，说明随机平移训练包含了中心位置所需的视觉模式。"
                "这与数据增强的基本作用一致：训练分布扩大后，模型面对较窄的测试分布不困难；"
                "反向从窄分布迁移到宽分布则缺少必要样本。若课程后续需要提高 B→A，更直接的方向"
                "是平移增强、相对位置编码或显式平移不变设计，而不是只增加参数量。",
                styles,
            ),
            Paragraph("7.2 与同学项目的外部参考", styles["Heading2"]),
            p(
                "同学仓库 kicious/translated-fashion-mnist-vit 在 commit "
                "943fa7b68730bc8ea7786bb41c7b8dc1d488883a 公开的四项 reported best 为 "
                "79.35%、88.22%、80.48%、19.37%。本实验 patch-16 ViT 为 80.07%、88.87%、"
                "80.97%、16.91%。前三项差异小于 1 个百分点，B→A 差 2.46 个百分点，数值"
                "量级基本一致。",
                styles,
            ),
            p(
                "两组记录不构成严格对照。同学项目在每个 epoch 使用测试集评价并保留最佳记录；"
                "本实验使用验证集选择 checkpoint，测试集只用于最终评价。模型选择规则、训练实现"
                "和随机过程均不同，因此这里仅检查结果是否处于相近范围，不据此判断项目优劣。",
                styles,
            ),
            Paragraph("7.3 局限性", styles["Heading2"]),
            bullet("正式结果仅包含 seed 42，未报告均值、标准差和显著性检验。", styles),
            bullet("MLP、CNN 与 ViT 参数量不同，结构比较同时包含容量差异。", styles),
            bullet("CUDA 未开启完全确定性；小于 1 个百分点的差异不作强解释。", styles),
            bullet("训练时间来自单一笔记本 GPU，只能用于本次实验内部比较。", styles),
            bullet("数据只包含平移变化，结论不直接外推到旋转、缩放或自然图像。", styles),
        ]
    )


def add_conclusion(story, styles) -> None:
    story.extend(
        [
            PageBreak(),
            Paragraph("8 结论与复现说明", styles["Heading1"]),
            p(
                "在统一实验协议下，CNN 是三类结构中表现最好的配置：四项准确率均为最高，且 "
                "B→A 达到 35.40%。其局部感受野和跨位置参数共享与较强的位置泛化相符，但从 "
                "B→B 到 B→A 仍下降 57.74 个百分点，说明结构先验不能弥补训练位置覆盖不足。",
                styles,
            ),
            p(
                "ViT 的 patch size 8 在本实验中取得更好的精度—成本折中。patch size 4 产生 "
                "256 个 token，训练时间增加到 8.93 分钟，却未获得准确率收益。Conv2d 与 "
                "Flatten+Linear patch embedding 的最大准确率差异为 0.61 个百分点，且两者在"
                "非重叠 patch 条件下具有等价的线性表达，本次结果不支持其中一种稳定优于另一种。",
                styles,
            ),
            p(
                "对该任务，训练位置分布的覆盖范围比单纯增加 token 数更关键。若继续实验，优先级"
                "应是多随机种子复现实验，以及相对位置编码或平移增强；前者用于量化不确定性，后者"
                "直接针对 B→A 的分布偏移。",
                styles,
            ),
            Paragraph("复现说明", styles["Heading2"]),
            p(
                "完整代码、配置、测试、指标文件、训练历史和绘图结果另附于结构化 ZIP。ZIP 不含"
                "原始数据和模型 checkpoint；解压后安装 requirements.txt，可先运行 "
                "<font name='Courier'>python -m unittest discover -s tests -v</font>，再运行 "
                "<font name='Courier'>python -m experiments.comparisons.run --help</font> 检查"
                "命令行入口。正式实验配置记录在 results/comparisons/manifest.json。",
                styles,
            ),
            Paragraph("参考文献", styles["Heading1"]),
            p(
                "[1] Xiao H., Rasul K., Vollgraf R. Fashion-MNIST: a Novel Image Dataset "
                "for Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.",
                styles,
                "Reference",
            ),
            p(
                "[2] Dosovitskiy A., Beyer L., Kolesnikov A., et al. An Image is Worth "
                "16×16 Words: Transformers for Image Recognition at Scale. ICLR, 2021.",
                styles,
                "Reference",
            ),
            p(
                "[3] LeCun Y., Bottou L., Bengio Y., Haffner P. Gradient-Based Learning "
                "Applied to Document Recognition. Proceedings of the IEEE, 1998.",
                styles,
                "Reference",
            ),
            p(
                "[4] 课程材料：《位置可变的 FashionMNIST 数据生成》《人工神经网络》"
                "《实验实现》《关于作业与实验报告》，2026。",
                styles,
                "Reference",
            ),
            p(
                "[5] kicious/translated-fashion-mnist-vit, commit "
                "943fa7b68730bc8ea7786bb41c7b8dc1d488883a.",
                styles,
                "Reference",
            ),
        ]
    )


def build_report(output_path: Path = OUTPUT_PATH) -> Path:
    register_fonts()
    styles = build_styles()
    rows = read_metrics()
    manifest = json.loads((RESULTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    story: list = []
    add_opening(story, styles)
    add_method(story, styles, manifest)
    add_overall_results(story, styles, rows)
    add_model_comparison(story, styles, rows)
    add_patch_size(story, styles, rows)
    add_patch_embedding(story, styles, rows)
    add_discussion(story, styles)
    add_conclusion(story, styles)

    ReportTemplate(str(output_path)).build(story)
    return output_path


if __name__ == "__main__":
    print(build_report())
