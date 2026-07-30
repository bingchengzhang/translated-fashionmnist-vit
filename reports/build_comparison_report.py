"""Build the comparison-study report as a Chinese PDF."""

from __future__ import annotations

import csv
import json
import textwrap
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
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "comparisons"
FIGURES_DIR = RESULTS_DIR / "figures"
CODE_DIR = ROOT / "experiments" / "comparisons"
OUTPUT_PATH = ROOT / "reports" / "comparison_study.pdf"

INK = colors.HexColor("#172033")
BLUE = colors.HexColor("#244A73")
BLUE_LIGHT = colors.HexColor("#EAF1F8")
ORANGE = colors.HexColor("#C96A2B")
GRAY = colors.HexColor("#5D6878")
GRID = colors.HexColor("#C9D1DC")
PAPER = colors.HexColor("#FBFCFE")


def register_fonts() -> None:
    candidates = [
        (
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),
            Path(r"C:\Windows\Fonts\consola.ttf"),
        ),
        (
            Path("/mnt/c/Windows/Fonts/msyh.ttc"),
            Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
            Path("/mnt/c/Windows/Fonts/consola.ttf"),
        ),
    ]
    for regular, bold, mono in candidates:
        if regular.exists() and bold.exists() and mono.exists():
            pdfmetrics.registerFont(TTFont("ReportCN", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportCN-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("ReportMono", str(mono)))
            return
    raise FileNotFoundError("Microsoft YaHei and Consolas fonts are required.")


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="ReportCN-Bold",
            fontSize=25,
            leading=36,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="ReportCN",
            fontSize=12,
            leading=20,
            textColor=GRAY,
            alignment=TA_CENTER,
        ),
        "Heading1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="ReportCN-Bold",
            fontSize=17,
            leading=24,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "Heading2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName="ReportCN-Bold",
            fontSize=13,
            leading=19,
            textColor=INK,
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="ReportCN",
            fontSize=10.3,
            leading=17,
            textColor=INK,
            alignment=TA_JUSTIFY,
            firstLineIndent=20.6,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "BodyNoIndent": ParagraphStyle(
            "BodyNoIndent",
            parent=base["BodyText"],
            fontName="ReportCN",
            fontSize=10.3,
            leading=17,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="ReportCN",
            fontSize=10,
            leading=16,
            textColor=INK,
            leftIndent=15,
            firstLineIndent=-9,
            bulletIndent=3,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="ReportCN",
            fontSize=8.6,
            leading=13,
            textColor=GRAY,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "Table": ParagraphStyle(
            "Table",
            parent=base["Normal"],
            fontName="ReportCN",
            fontSize=8.1,
            leading=11,
            textColor=INK,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "TableLeft": ParagraphStyle(
            "TableLeft",
            parent=base["Normal"],
            fontName="ReportCN",
            fontSize=8.1,
            leading=11,
            textColor=INK,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "Callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="ReportCN",
            fontSize=10,
            leading=16,
            textColor=INK,
            leftIndent=12,
            rightIndent=12,
            borderColor=colors.HexColor("#93A9C2"),
            borderWidth=0.8,
            borderPadding=9,
            backColor=BLUE_LIGHT,
            spaceBefore=7,
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "Code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="ReportMono",
            fontSize=6.3,
            leading=8.1,
            textColor=colors.HexColor("#1F2937"),
            backColor=colors.HexColor("#F4F6F8"),
            borderColor=GRID,
            borderWidth=0.4,
            borderPadding=6,
            leftIndent=0,
            rightIndent=0,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "TOCHeading": ParagraphStyle(
            "TOCHeading",
            parent=base["Heading1"],
            fontName="ReportCN-Bold",
            fontSize=18,
            leading=24,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "TOC0": ParagraphStyle(
            "TOC0",
            fontName="ReportCN",
            fontSize=10.5,
            leading=18,
            leftIndent=10,
            firstLineIndent=-10,
            textColor=INK,
        ),
        "TOC1": ParagraphStyle(
            "TOC1",
            fontName="ReportCN",
            fontSize=9.5,
            leading=16,
            leftIndent=26,
            firstLineIndent=-8,
            textColor=GRAY,
        ),
    }


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, styles: dict[str, ParagraphStyle]):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=18 * mm,
            title="位置变化下的模型归纳偏置与 ViT Patch 设计对比实验",
            author="",
            subject="Translated FashionMNIST comparison study",
        )
        self.styles = styles
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="content",
        )
        self.addPageTemplates(
            [PageTemplate(id="main", frames=frame, onPage=self._draw_page)]
        )
        self._heading_sequence = 0

    def _draw_page(self, canvas, doc) -> None:
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.5)
            canvas.line(20 * mm, A4[1] - 13 * mm, A4[0] - 20 * mm, A4[1] - 13 * mm)
            canvas.setFont("ReportCN", 8)
            canvas.setFillColor(GRAY)
            canvas.drawString(
                20 * mm,
                A4[1] - 10 * mm,
                "位置变化下的模型归纳偏置与 ViT Patch 设计对比实验",
            )
            canvas.drawRightString(
                A4[0] - 20 * mm,
                10 * mm,
                f"第 {doc.page} 页",
            )
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        if flowable.style.name not in {"Heading1", "Heading2"}:
            return
        level = 0 if flowable.style.name == "Heading1" else 1
        if not hasattr(flowable, "_bookmark_name"):
            self._heading_sequence += 1
            flowable._bookmark_name = f"heading-{self._heading_sequence}"
        key = flowable._bookmark_name
        text = flowable.getPlainText()
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def paragraph(text: str, styles, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str, styles) -> Paragraph:
    return Paragraph(f"• {text}", styles["Bullet"])


def table(
    rows: list[list[str]],
    widths: list[float],
    styles,
    left_columns: set[int] | None = None,
    repeat_rows: int = 1,
) -> Table:
    left_columns = left_columns or set()
    rendered = []
    for row_index, row in enumerate(rows):
        rendered_row = []
        for column_index, value in enumerate(row):
            style = styles["TableLeft"] if column_index in left_columns else styles["Table"]
            if row_index == 0:
                value = f"<b>{escape(str(value))}</b>"
            else:
                value = escape(str(value))
            rendered_row.append(Paragraph(value, style))
        rendered.append(rendered_row)
    result = Table(rendered, colWidths=widths, repeatRows=repeat_rows, hAlign="CENTER")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "ReportCN-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.45, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def figure(path: Path, caption: str, styles, max_height: float = 102 * mm):
    image = Image(str(path))
    max_width = 168 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return KeepTogether(
        [
            image,
            Paragraph(caption, styles["Caption"]),
        ]
    )


def code_block(code: str, styles) -> XPreformatted:
    wrapped_lines = []
    for line in code.rstrip().splitlines():
        if not line:
            wrapped_lines.append("")
            continue
        indent = line[: len(line) - len(line.lstrip())]
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=94,
                subsequent_indent=indent + "    ",
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            or [line]
        )
    return XPreformatted(
        escape("\n".join(wrapped_lines)),
        styles["Code"],
    )


def extract_block(path: Path, start: str, end: str | None) -> str:
    source = path.read_text(encoding="utf-8")
    start_index = source.index(start)
    if end is None:
        return source[start_index:]
    end_index = source.index(end, start_index)
    return source[start_index:end_index].rstrip()


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
    row = next(
        row
        for row in rows
        if row["config_id"] == config_id and row["setting"] == setting
    )
    return 100 * row["test_accuracy"]


def total_minutes(rows: list[dict], config_id: str) -> float:
    unique = {
        row["train_mode"]: row["train_elapsed_seconds"]
        for row in rows
        if row["config_id"] == config_id
    }
    return sum(unique.values()) / 60


def add_title_page(story, styles) -> None:
    story.extend(
        [
            Spacer(1, 43 * mm),
            Paragraph(
                "位置变化下的模型归纳偏置<br/>与 ViT Patch 设计对比实验",
                styles["Title"],
            ),
            Spacer(1, 7 * mm),
            Paragraph(
                "位置可变 FashionMNIST 分类 · 三组扩展实验报告",
                styles["Subtitle"],
            ),
            Spacer(1, 22 * mm),
            Table(
                [["研究对象", "64×64 画布上的随机位置与固定位置 FashionMNIST"],
                 ["对比内容", "模型结构、Patch Size、Patch Embedding"],
                 ["实验协议", "验证集选择最佳模型，官方测试集仅作最终评价"],
                 ["实验环境", "PyTorch 2.11 · CUDA 12.8 · RTX 5070 Laptop GPU"]],
                colWidths=[35 * mm, 112 * mm],
                style=[
                    ("FONTNAME", (0, 0), (-1, -1), "ReportCN"),
                    ("FONTNAME", (0, 0), (0, -1), "ReportCN-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("LEADING", (0, 0), (-1, -1), 16),
                    ("BACKGROUND", (0, 0), (0, -1), BLUE_LIGHT),
                    ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                    ("GRID", (0, 0), (-1, -1), 0.5, GRID),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ],
            ),
            Spacer(1, 30 * mm),
            Paragraph("短学期深度学习课程实验 · 2026 年夏", styles["Subtitle"]),
            PageBreak(),
        ]
    )


def add_toc(story, styles) -> None:
    toc = TableOfContents()
    toc.levelStyles = [styles["TOC0"], styles["TOC1"]]
    story.extend(
        [
            Paragraph("目录", styles["TOCHeading"]),
            toc,
            PageBreak(),
        ]
    )


def add_abstract(story, styles) -> None:
    story.extend(
        [
            Paragraph("摘要", styles["Heading1"]),
            paragraph(
                "本实验围绕位置可变 FashionMNIST 分类中的三个扩展问题展开。"
                "首先，在统一训练与评价协议下比较 MLP、CNN 与 Vision Transformer（ViT），"
                "考察不同结构归纳偏置对位置分布变化的影响；其次，保持 ViT 主体结构不变，"
                "比较 patch size 为 4、8、16 时的准确率与训练成本；最后，对比 Conv2d 与 "
                "Flatten+Linear 两种 patch embedding 实现。全部配置使用相同的数据划分、"
                "优化器、训练轮数和随机种子，最佳 checkpoint 由验证集选择，官方测试集仅用于"
                "最终评价。结果表明，CNN 在四个位置 setting 中均取得最高准确率，尤其在固定"
                "位置训练、随机位置测试的 B→A 设置下达到 35.40%；patch size 8 在精度与成本"
                "之间表现出更合理的折中；两种 patch embedding 的最大准确率差异仅为 0.61 个"
                "百分点，与其数学等价性相符。实验同时显示，所有模型在 B→A 下都存在明显下降，"
                "说明只使用固定位置样本训练不足以获得稳定的位置泛化能力。",
                styles,
            ),
            paragraph(
                "<b>关键词：</b>FashionMNIST；位置泛化；卷积神经网络；"
                "Vision Transformer；Patch Embedding",
                styles,
                "BodyNoIndent",
            ),
            PageBreak(),
        ]
    )


def add_design(story, styles, manifest: dict) -> None:
    story.extend(
        [
            Paragraph("1 实验目标与设计", styles["Heading1"]),
            Paragraph("1.1 研究问题", styles["Heading2"]),
            paragraph(
                "基础任务把 28×28 的 FashionMNIST 图像放入 64×64 黑色画布。"
                "A 表示图像在画布范围内随机平移，B 表示图像固定在画布中心。"
                "本报告不重复基础 ViT 实现，而是集中回答三个对比问题。",
                styles,
            ),
            bullet("模型结构：MLP、CNN 与 ViT 谁更能适应位置分布变化？", styles),
            bullet("Patch Size：更细的 patch 是否必然带来更好的分类结果？", styles),
            bullet(
                "Patch Embedding：卷积切块与 Flatten+Linear 投影是否产生实质差异？",
                styles,
            ),
            Paragraph("1.2 四种评价设置", styles["Heading2"]),
            paragraph(
                "箭头左侧表示训练分布，右侧表示测试分布。A→A 与 B→B 衡量同分布性能，"
                "A→B 和 B→A 衡量训练与测试位置分布不一致时的泛化能力。",
                styles,
            ),
            table(
                [
                    ["Setting", "训练分布", "测试分布", "用途"],
                    ["1", "A：随机位置", "A：随机位置", "随机位置同分布评价"],
                    ["2", "B：固定居中", "B：固定居中", "固定位置同分布评价"],
                    ["3", "A：随机位置", "B：固定居中", "随机训练向中心位置泛化"],
                    ["4", "B：固定居中", "A：随机位置", "固定训练向随机位置泛化"],
                ],
                [18 * mm, 35 * mm, 35 * mm, 78 * mm],
                styles,
                left_columns={3},
            ),
            Spacer(1, 5 * mm),
            Paragraph("1.3 统一实验协议", styles["Heading2"]),
            paragraph(
                "FashionMNIST 官方训练集按固定随机顺序划分为 54,000 个训练样本和 "
                "6,000 个验证样本；官方测试集包含 10,000 个样本。每个配置只训练 A 模型和 "
                "B 模型各一次，再分别在 A、B 测试集上评价，从而得到四个 setting。"
                "checkpoint 依据同分布验证集准确率选择，避免使用测试集选择最佳 epoch。",
                styles,
            ),
        ]
    )
    protocol = manifest["protocol"]
    environment = manifest["environment"]
    story.append(
        table(
            [
                ["项目", "设置", "项目", "设置"],
                ["画布大小", f"{protocol['canvas_size']}×{protocol['canvas_size']}",
                 "训练轮数", str(protocol["epochs"])],
                ["Batch size", str(protocol["batch_size"]),
                 "初始学习率", f"{protocol['learning_rate']:.0e}"],
                ["优化器", "AdamW", "权重衰减", f"{protocol['weight_decay']:.0e}"],
                ["验证集比例", f"{100 * protocol['val_fraction']:.0f}%",
                 "随机种子", str(protocol["seed"])],
                ["混合精度", "开启", "模型选择", "最佳验证准确率"],
                ["Python / PyTorch", f"{environment['python']} / {environment['torch']}",
                 "GPU", environment["gpu"]],
            ],
            [31 * mm, 53 * mm, 31 * mm, 53 * mm],
            styles,
            left_columns={1, 3},
        )
    )
    story.extend(
        [
            Spacer(1, 5 * mm),
            PageBreak(),
            Paragraph("1.4 模型配置", styles["Heading2"]),
            paragraph(
                "三组实验共包含六个配置。ViT 均使用 128 维 token、4 层 Transformer "
                "Encoder、4 个注意力头、512 维前馈层、GELU 和可学习绝对位置编码。"
                "模型结构对比并非严格参数量匹配，结论主要针对当前课程实验配置。",
                styles,
            ),
            table(
                [
                    ["配置", "关键结构", "Patch", "Token 数", "参数量"],
                    ["MLP", "4096→128→128→10", "-", "-", "542,474"],
                    ["CNN", "32/64/128 通道卷积 + 池化", "-", "-", "205,994"],
                    ["ViT (patch 16)", "4 层 Encoder，4 heads", "16", "16", "829,834"],
                    ["ViT (patch 8)", "4 层 Encoder，4 heads", "8", "64", "811,402"],
                    ["ViT (patch 4)", "4 层 Encoder，4 heads", "4", "256", "829,834"],
                    ["ViT (Flatten+Linear)", "patch 16，线性投影", "16", "16", "829,834"],
                ],
                [37 * mm, 68 * mm, 18 * mm, 20 * mm, 25 * mm],
                styles,
                left_columns={0, 1},
            ),
        ]
    )


def add_overall_results(story, styles, rows: list[dict]) -> None:
    labels = [
        ("mlp", "MLP"),
        ("cnn", "CNN"),
        ("vit_p16_conv", "ViT (patch 16)"),
        ("vit_p8_conv", "ViT (patch 8)"),
        ("vit_p4_conv", "ViT (patch 4)"),
        ("vit_p16_linear", "ViT (Flatten+Linear)"),
    ]
    result_rows = [["配置", "参数量", "A→A", "B→B", "A→B", "B→A"]]
    for config_id, label in labels:
        param = next(row["parameter_count"] for row in rows if row["config_id"] == config_id)
        result_rows.append(
            [
                label,
                f"{param:,}",
                *[f"{accuracy(rows, config_id, setting):.2f}%" for setting in (1, 2, 3, 4)],
            ]
        )
    story.extend(
        [
            PageBreak(),
            Paragraph("2 总体结果", styles["Heading1"]),
            paragraph(
                "表 3 汇总六个配置在四种 setting 下的最终测试准确率。"
                "该表用于统一查阅，后续各节只在对应控制变量范围内进行比较。",
                styles,
            ),
            table(
                result_rows,
                [43 * mm, 27 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm],
                styles,
                left_columns={0},
            ),
            Paragraph(
                "表 3　六个配置在四种位置 setting 下的测试准确率",
                styles["Caption"],
            ),
            figure(
                FIGURES_DIR / "training_dynamics.png",
                "图 1　六种配置在 A、B 训练分布上的验证准确率变化。"
                "曲线整体在 15 个 epoch 内趋于稳定，未出现持续发散。",
                styles,
                max_height=91 * mm,
            ),
            paragraph(
                "训练曲线显示，B 分布只包含中心位置，任务相对集中，因此多数模型在 B "
                "验证集上的准确率更高、前期上升更快。A 分布包含随机平移，模型需要同时处理"
                "类别与位置变化，验证准确率普遍较低。后续分析重点不是比较 A/B 数据集的绝对"
                "难度，而是观察同一个训练分布在位置发生变化后能保留多少性能。",
                styles,
            ),
        ]
    )


def add_model_comparison(story, styles, rows: list[dict]) -> None:
    story.extend(
        [
            PageBreak(),
            Paragraph("3 MLP、CNN 与 ViT 对比", styles["Heading1"]),
            Paragraph("3.1 实验结果", styles["Heading2"]),
            figure(
                FIGURES_DIR / "model_comparison.png",
                "图 2　MLP、CNN 与 patch-16 ViT 在四种位置 setting 下的测试准确率。",
                styles,
                max_height=93 * mm,
            ),
            paragraph(
                "CNN 在四个 setting 中均取得最高准确率：A→A 为 92.35%，B→B 为 "
                "93.14%，A→B 为 92.87%，B→A 为 35.40%。在同分布或从随机位置训练迁移到"
                "中心位置测试时，CNN 都保持在 92% 以上。相比之下，MLP 的 A→A 与 A→B "
                "约为 72%，patch-16 ViT 约为 80% 至 81%。",
                styles,
            ),
            Paragraph("3.2 位置泛化分析", styles["Heading2"]),
            figure(
                FIGURES_DIR / "position_generalization.png",
                "图 3　固定位置训练时的分布变化。右图给出 B→B 与 B→A 的准确率差值。",
                styles,
                max_height=79 * mm,
            ),
            paragraph(
                f"从 B→B 切换到 B→A 后，MLP、CNN 和 ViT 分别下降 "
                f"{accuracy(rows, 'mlp', 2) - accuracy(rows, 'mlp', 4):.2f}、"
                f"{accuracy(rows, 'cnn', 2) - accuracy(rows, 'cnn', 4):.2f} 和 "
                f"{accuracy(rows, 'vit_p16_conv', 2) - accuracy(rows, 'vit_p16_conv', 4):.2f} "
                "个百分点。CNN 的下降幅度最小，B→A 也比 MLP 和 ViT 分别高 21.41 和 "
                "18.49 个百分点。",
                styles,
            ),
            paragraph(
                "这一现象与模型结构的空间归纳偏置相符。CNN 通过局部感受野和跨位置共享的"
                "卷积核重复使用同一特征检测器，因而对平移具有更有利的结构先验；MLP 对每个"
                "像素位置使用独立连接，难以把中心位置学到的模式直接迁移到其他位置；当前 "
                "ViT 使用可学习绝对位置编码，token 表征与训练位置绑定较强。需要强调的是，"
                "CNN 仍下降 57.74 个百分点，因此局部参数共享只能缓解而不能消除由固定位置"
                "训练造成的分布偏移。",
                styles,
            ),
            paragraph(
                "<b>结论：</b>CNN 是当前配置中最有效的模型基线；随机位置训练对三类模型"
                "都很重要，不能用固定居中训练的高准确率替代位置泛化评价。",
                styles,
                "Callout",
            ),
        ]
    )


def add_patch_size(story, styles, rows: list[dict]) -> None:
    story.extend(
        [
            PageBreak(),
            Paragraph("4 ViT Patch Size 对比", styles["Heading1"]),
            Paragraph("4.1 精度结果", styles["Heading2"]),
            figure(
                FIGURES_DIR / "patch_size_comparison.png",
                "图 4　保持 ViT 主体不变时，patch size 4、8、16 的测试准确率。",
                styles,
                max_height=92 * mm,
            ),
            paragraph(
                "patch size 8 在 A→A、A→B 与 B→A 上分别达到 80.40%、81.13% 和 "
                "18.81%，均为三个 patch 配置中的最高值；patch size 16 在 B→B 上略高，"
                "达到 88.87%。patch size 4 的 A→A 与 A→B 下降到 75.28% 和 74.97%，"
                "没有体现出更细分块带来的精度收益。",
                styles,
            ),
            Paragraph("4.2 计算成本与折中", styles["Heading2"]),
            figure(
                FIGURES_DIR / "training_time_comparison.png",
                "图 5　统一协议下，各配置训练 A、B 两个模型的累计时间。该时间仅反映当前硬件。",
                styles,
                max_height=88 * mm,
            ),
            table(
                [
                    ["Patch size", "Token 数", "注意力矩阵规模", "A/B 总训练时间"],
                    ["4", "256", "256² = 65,536", f"{total_minutes(rows, 'vit_p4_conv'):.2f} min"],
                    ["8", "64", "64² = 4,096", f"{total_minutes(rows, 'vit_p8_conv'):.2f} min"],
                    ["16", "16", "16² = 256", f"{total_minutes(rows, 'vit_p16_conv'):.2f} min"],
                ],
                [32 * mm, 31 * mm, 57 * mm, 47 * mm],
                styles,
            ),
            Paragraph(
                "表 4　不同 patch size 的 token 数、理论注意力规模与实测训练时间",
                styles["Caption"],
            ),
            paragraph(
                "在 64×64 画布上，patch size 4、8、16 分别产生 256、64、16 个图像 token。"
                "自注意力的两两关系规模随 token 数平方增长，因此 patch size 4 的注意力矩阵"
                "元素数在理论上是 patch size 16 的 256 倍。端到端训练时间不会按同样比例增长，"
                "因为数据加载、线性层、GPU 并行效率等也占用时间；实际记录中，patch size 4 "
                "训练两个模型需要 8.93 分钟，仍是三个设置中最慢的。",
                styles,
            ),
            paragraph(
                "更小 patch 提供更细的局部采样，但也增加序列长度和优化难度。在本实验规模下，"
                "patch size 8 同时避免了 patch size 16 的粗粒度和 patch size 4 的长序列负担，"
                "表现出较合理的精度与成本折中。该结论只适用于当前画布、模型深度和训练预算，"
                "不能直接外推到更大数据集或更深 ViT。",
                styles,
            ),
            paragraph(
                "<b>结论：</b>更小的 patch 并不必然提高分类准确率；当前配置优先采用 "
                "patch size 8。",
                styles,
                "Callout",
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
            Paragraph("5 Patch Embedding 实现对比", styles["Heading1"]),
            Paragraph("5.1 实验结果", styles["Heading2"]),
            figure(
                FIGURES_DIR / "patch_embedding_comparison.png",
                "图 6　patch size 16 时，Conv2d 与 Flatten+Linear patch embedding 的结果。",
                styles,
                max_height=93 * mm,
            ),
            paragraph(
                "两种实现的参数量均为 829,834。Conv2d 版本在 A→A、B→B、A→B、B→A "
                "上分别得到 80.07%、88.87%、80.97%、16.91%；Flatten+Linear 版本分别为 "
                "79.67%、88.80%、80.48%、16.30%。四项差异依次为 "
                + "、".join(f"{value:.2f}" for value in differences)
                + f" 个百分点，最大差异为 {max(differences):.2f} 个百分点。",
                styles,
            ),
            Paragraph("5.2 数学关系", styles["Heading2"]),
            paragraph(
                "当卷积核大小和步长都等于 patch size，且 patch 之间不重叠时，Conv2d "
                "会对每个 patch 使用同一组卷积核权重。把每个卷积核展平后，它对应一个从 "
                "patch_dim 到 embed_dim 的线性变换。因此，Conv2d patch embedding 与对每个"
                "展平 patch 共享同一个 Linear 层在表达形式上等价。项目测试通过权重重排验证了"
                "两种实现的输出在浮点误差范围内一致。",
                styles,
            ),
            paragraph(
                "单次训练中仍会出现不足一个百分点的差异，可能来自底层算子、随机训练过程和"
                "浮点计算顺序，而不是两种 embedding 具有不同的理论表达能力。当前只有一个"
                "随机种子，不能对 0.61 个百分点的差异作显著性判断。",
                styles,
            ),
            paragraph(
                "<b>结论：</b>当前结果不支持“卷积版 patch embedding 更强”的判断。"
                "在非重叠 patch 设置下，两种实现应主要按照代码可读性和工程便利性选择。",
                styles,
                "Callout",
            ),
        ]
    )


def add_external_reference(story, styles, rows: list[dict]) -> None:
    story.extend(
        [
            PageBreak(),
            Paragraph("6 与同学项目的外部参考", styles["Heading1"]),
            figure(
                FIGURES_DIR / "teammate_baseline_comparison.png",
                "图 7　本项目 patch-16 ViT 与同学仓库公开记录的参考比较。",
                styles,
                max_height=90 * mm,
            ),
            paragraph(
                "外部数据来自 kicious/translated-fashion-mnist-vit 仓库 commit "
                "943fa7b68730bc8ea7786bb41c7b8dc1d488883a。该版本公开记录的四项最佳"
                "准确率为 79.35%、88.22%、80.48% 和 19.37%；本项目对应结果为 "
                "80.07%、88.87%、80.97% 和 16.91%。前三项相差不足一个百分点，B→A "
                "相差 2.46 个百分点。",
                styles,
            ),
            paragraph(
                "这组数字不能解释为严格的模型优劣比较。同学仓库在每个 epoch 都使用测试集"
                "评价并保留 reported best，本项目则使用独立验证集选择 checkpoint，测试集只"
                "评价一次。由于模型选择规则和训练实现不同，图 7 的作用是检查结果量级是否一致，"
                "而不是进行统计等价比较。",
                styles,
            ),
        ]
    )


def add_limitations_and_conclusion(story, styles) -> None:
    story.extend(
        [
            Paragraph("7 局限性", styles["Heading1"]),
            bullet(
                "正式结果来自 seed 42 的一次训练，没有重复多个随机种子，因此不报告均值、"
                "标准差或显著性。",
                styles,
            ),
            bullet(
                "CUDA 训练未开启完全确定性；小于一个百分点的差异应视为观察值，不能作过强解释。",
                styles,
            ),
            bullet(
                "MLP、CNN 与 ViT 的参数量并未严格匹配，模型对比同时包含结构和容量差异。",
                styles,
            ),
            bullet(
                "训练时间只来自 RTX 5070 Laptop GPU 的一次记录，不能作为跨硬件基准。",
                styles,
            ),
            bullet(
                "实验只覆盖 FashionMNIST、64×64 画布和一种平移规则，结论不直接外推到"
                "自然图像、缩放、旋转或更复杂背景。",
                styles,
            ),
            bullet(
                "同学项目结果与本项目的 checkpoint 选择规则不同，只能作为外部参考。",
                styles,
            ),
            Paragraph("8 结论", styles["Heading1"]),
            paragraph(
                "本报告在统一协议下完成了三组对比。第一，CNN 在所有 setting 上均取得最高"
                "准确率，并在 B→A 中显示出相对更强的位置泛化能力，但固定位置训练造成的下降"
                "依然明显。第二，ViT 的 patch size 8 在当前训练预算下取得了较好的精度与计算"
                "成本平衡，patch size 4 增加了序列长度和训练时间，却没有获得精度收益。第三，"
                "Conv2d 与 Flatten+Linear patch embedding 的结果非常接近，结合数学等价性和"
                "单元测试，没有证据表明其中一种实现具有稳定优势。",
                styles,
            ),
            paragraph(
                "三组实验共同说明，位置泛化不仅取决于模型容量，也与空间参数共享、训练位置"
                "覆盖范围和 token 化方式有关。对当前任务而言，直接增加 token 数不如选择合适"
                "的结构归纳偏置和训练分布更有效。",
                styles,
            ),
        ]
    )


def add_appendix(story, styles) -> None:
    configs = (CODE_DIR / "configs.py").read_text(encoding="utf-8")
    models = (CODE_DIR / "models.py").read_text(encoding="utf-8")
    protocol_path = CODE_DIR / "protocol.py"
    split_code = extract_block(
        protocol_path,
        "@dataclass\nclass ProtocolConfig",
        "\ndef _loader(",
    )
    fit_code = extract_block(
        protocol_path,
        "def _can_resume(",
        "\ndef run_configuration(",
    )
    evaluate_code = extract_block(
        protocol_path,
        "def run_configuration(",
        None,
    )
    test_code = extract_block(
        ROOT / "tests" / "test_comparisons.py",
        "    def test_conv_and_linear_patch_projection_are_equivalent",
        "\n\nif __name__",
    )

    story.extend(
        [
            PageBreak(),
            Paragraph("附录 A 关键代码", styles["Heading1"]),
            paragraph(
                "附录列出三组对比直接依赖的配置、模型、实验协议和等价性测试。图表生成、"
                "通用工具及基础数据集代码不重复附录，完整代码见仓库对应目录。",
                styles,
            ),
            Paragraph("A.1 实验配置：experiments/comparisons/configs.py", styles["Heading2"]),
            code_block(configs, styles),
            Paragraph("A.2 对比模型：experiments/comparisons/models.py", styles["Heading2"]),
            code_block(models, styles),
            Paragraph("A.3 数据划分与协议配置", styles["Heading2"]),
            code_block(split_code, styles),
            Paragraph("A.4 Checkpoint 选择与训练流程", styles["Heading2"]),
            code_block(fit_code, styles),
            Paragraph("A.5 四种 setting 的最终评价", styles["Heading2"]),
            code_block(evaluate_code, styles),
            Paragraph("A.6 Patch Embedding 等价性测试", styles["Heading2"]),
            code_block(test_code, styles),
            Paragraph("附录 B 复现命令", styles["Heading1"]),
            code_block(
                "cd /home/bccc/dl-course\n"
                "source ~/miniconda3/etc/profile.d/conda.sh\n"
                "conda activate torch101\n\n"
                "python -m unittest discover -s tests -v\n\n"
                "python -m experiments.comparisons.run \\\n"
                "  --groups all \\\n"
                "  --epochs 15 \\\n"
                "  --batch-size 64 \\\n"
                "  --num-workers 4 \\\n"
                "  --output-dir results/comparisons",
                styles,
            ),
            Paragraph("参考资料", styles["Heading1"]),
            paragraph(
                "[1] 课程材料：《关于作业与实验报告》《实验实现》，2026 年夏。<br/>"
                "[2] Xiao H., Rasul K., Vollgraf R. Fashion-MNIST: a Novel Image Dataset "
                "for Benchmarking Machine Learning Algorithms, 2017.<br/>"
                "[3] Dosovitskiy A. et al. An Image is Worth 16×16 Words: Transformers "
                "for Image Recognition at Scale, ICLR 2021.<br/>"
                "[4] kicious/translated-fashion-mnist-vit, commit "
                "943fa7b68730bc8ea7786bb41c7b8dc1d488883a.",
                styles,
                "BodyNoIndent",
            ),
        ]
    )


def build_report(output_path: Path = OUTPUT_PATH) -> Path:
    register_fonts()
    styles = build_styles()
    rows = read_metrics()
    manifest = json.loads((RESULTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    story = []
    add_title_page(story, styles)
    add_toc(story, styles)
    add_abstract(story, styles)
    add_design(story, styles, manifest)
    add_overall_results(story, styles, rows)
    add_model_comparison(story, styles, rows)
    add_patch_size(story, styles, rows)
    add_patch_embedding(story, styles, rows)
    add_external_reference(story, styles, rows)
    add_limitations_and_conclusion(story, styles)
    add_appendix(story, styles)

    document = ReportDocTemplate(str(output_path), styles)
    document.multiBuild(story)
    return output_path


if __name__ == "__main__":
    print(build_report())
