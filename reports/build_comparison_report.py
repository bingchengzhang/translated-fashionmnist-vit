"""Build the four-page English comparison report."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics.csv"
OUTPUT_PATH = ROOT / "reports" / "comparison_study.pdf"
FONT_DIR = ROOT / "reports" / "fonts"

INK = colors.HexColor("#182532")
NAVY = colors.HexColor("#23445E")
ACCENT = colors.HexColor("#2E7485")
BLUE = colors.HexColor("#557E9D")
ORANGE = colors.HexColor("#C88152")
GRAY = colors.HexColor("#A8B4BC")
MUTED = colors.HexColor("#667480")
RULE = colors.HexColor("#D4DCE1")
PALE = colors.HexColor("#F5F7F8")
PALE_BLUE = colors.HexColor("#EEF4F5")

SETTINGS = ("A -> A", "B -> B", "A -> B", "B -> A")
SETTING_KEYS = (("A", "A"), ("B", "B"), ("A", "B"), ("B", "A"))


def register_fonts() -> None:
    font_files = {
        "SourceSerif": "SourceSerif4-Regular.ttf",
        "SourceSerif-Semibold": "SourceSerif4-Semibold.ttf",
        "SourceSerif-Italic": "SourceSerif4-It.ttf",
        "SourceSans": "SourceSans3-Regular.ttf",
        "SourceSans-Semibold": "SourceSans3-Semibold.ttf",
    }
    for name, filename in font_files.items():
        path = FONT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing report font: {path}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "SourceSerif",
        normal="SourceSerif",
        bold="SourceSerif-Semibold",
        italic="SourceSerif-Italic",
        boldItalic="SourceSerif-Semibold",
    )
    pdfmetrics.registerFontFamily(
        "SourceSans",
        normal="SourceSans",
        bold="SourceSans-Semibold",
        italic="SourceSans",
        boldItalic="SourceSans-Semibold",
    )


def load_metrics() -> dict[str, dict]:
    configs: dict[str, dict] = {}
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            config = configs.setdefault(
                row["config_id"],
                {
                    "name": row["display_name"],
                    "parameters": int(row["parameter_count"]),
                    "train_seconds": {},
                    "accuracy": {},
                },
            )
            key = (row["train_mode"], row["test_mode"])
            config["accuracy"][key] = 100 * float(row["test_accuracy"])
            config["train_seconds"][row["train_mode"]] = float(
                row["train_elapsed_seconds"]
            )
    return configs


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Kicker": ParagraphStyle(
            "Kicker",
            parent=base["Normal"],
            fontName="SourceSans-Semibold",
            fontSize=8.3,
            leading=10,
            tracking=0.8,
            textColor=ACCENT,
            spaceAfter=4.5 * mm,
        ),
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="SourceSerif-Semibold",
            fontSize=28,
            leading=31,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2.4 * mm,
        ),
        "PageTitle": ParagraphStyle(
            "PageTitle",
            parent=base["Title"],
            fontName="SourceSerif-Semibold",
            fontSize=26.5,
            leading=29.5,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2.2 * mm,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="SourceSans",
            fontSize=11,
            leading=14.5,
            textColor=MUTED,
            spaceAfter=4 * mm,
        ),
        "Meta": ParagraphStyle(
            "Meta",
            parent=base["Normal"],
            fontName="SourceSans",
            fontSize=8.3,
            leading=10.8,
            textColor=MUTED,
        ),
        "AbstractLabel": ParagraphStyle(
            "AbstractLabel",
            parent=base["Normal"],
            fontName="SourceSerif-Italic",
            fontSize=10.1,
            leading=13,
            textColor=NAVY,
            spaceAfter=1.1 * mm,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="SourceSerif",
            fontSize=10.25,
            leading=14.8,
            textColor=INK,
            spaceAfter=2.2 * mm,
        ),
        "BodySmall": ParagraphStyle(
            "BodySmall",
            parent=base["BodyText"],
            fontName="SourceSerif",
            fontSize=9.25,
            leading=12.9,
            textColor=INK,
        ),
        "SansSmall": ParagraphStyle(
            "SansSmall",
            parent=base["Normal"],
            fontName="SourceSans",
            fontSize=8.25,
            leading=10.8,
            textColor=MUTED,
        ),
        "SansSmallCenter": ParagraphStyle(
            "SansSmallCenter",
            parent=base["Normal"],
            fontName="SourceSans",
            fontSize=8.25,
            leading=10.8,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
        "MiniTitle": ParagraphStyle(
            "MiniTitle",
            parent=base["Normal"],
            fontName="SourceSans-Semibold",
            fontSize=10,
            leading=12.4,
            textColor=NAVY,
            spaceAfter=1.4 * mm,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName="SourceSans-Semibold",
            fontSize=8.2,
            leading=10.3,
            textColor=NAVY,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName="SourceSans",
            fontSize=8.15,
            leading=10.3,
            textColor=INK,
        ),
        "TableCellCenter": ParagraphStyle(
            "TableCellCenter",
            parent=base["Normal"],
            fontName="SourceSans",
            fontSize=8.15,
            leading=10.3,
            alignment=TA_CENTER,
            textColor=INK,
        ),
        "TableCellStrong": ParagraphStyle(
            "TableCellStrong",
            parent=base["Normal"],
            fontName="SourceSans-Semibold",
            fontSize=8.15,
            leading=10.3,
            alignment=TA_CENTER,
            textColor=NAVY,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="SourceSans",
            fontSize=8,
            leading=10.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=2.4 * mm,
        ),
        "Callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="SourceSerif",
            fontSize=9.8,
            leading=14,
            textColor=INK,
        ),
        "FindingLabel": ParagraphStyle(
            "FindingLabel",
            parent=base["Normal"],
            fontName="SourceSans-Semibold",
            fontSize=7.9,
            leading=9.6,
            textColor=ACCENT,
            alignment=TA_CENTER,
        ),
        "FindingValue": ParagraphStyle(
            "FindingValue",
            parent=base["Normal"],
            fontName="SourceSerif-Semibold",
            fontSize=15,
            leading=17,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "FindingText": ParagraphStyle(
            "FindingText",
            parent=base["Normal"],
            fontName="SourceSans",
            fontSize=7.9,
            leading=10.1,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="SourceSerif",
            fontSize=9.25,
            leading=12.9,
            textColor=INK,
            leftIndent=3.5 * mm,
            firstLineIndent=-2.5 * mm,
            spaceAfter=1.1 * mm,
        ),
        "Code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=10.2,
            textColor=INK,
        ),
    }


class ReportDocument(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=23 * mm,
            rightMargin=23 * mm,
            topMargin=18 * mm,
            bottomMargin=17 * mm,
            title="Position Generalization in Translated FashionMNIST",
            author="",
            subject="Architecture, patch scale, and patch embedding comparisons",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="main",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates([PageTemplate(id="academic", frames=frame, onPage=self.decorate)])

    def decorate(self, canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.45)
        canvas.line(23 * mm, 13 * mm, A4[0] - 23 * mm, 13 * mm)
        canvas.setFont("SourceSans", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(23 * mm, 8.5 * mm, "TRANSLATED FASHIONMNIST COMPARISON STUDY")
        canvas.drawRightString(A4[0] - 23 * mm, 8.5 * mm, f"{document.page}")
        canvas.restoreState()


def section_header(number: str, title: str, style_map) -> Table:
    return Table(
        [[Paragraph(number, style_map["TableHeader"]), Paragraph(title, style_map["MiniTitle"])]],
        colWidths=[8 * mm, 156 * mm],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LINEBELOW", (0, 0), (-1, -1), 0.55, RULE),
            ]
        ),
        spaceBefore=4.2 * mm,
        spaceAfter=2.4 * mm,
    )


def ruled_table(rows, widths, *, header=True, compact=False) -> Table:
    line_height = 4.8 if compact else 5.5
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.65, NAVY),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.65, NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), line_height),
        ("BOTTOMPADDING", (0, 0), (-1, -1), line_height),
    ]
    if not header:
        commands.pop(0)
    return Table(rows, colWidths=widths, style=TableStyle(commands))


def side_note(text: str, style_map) -> Table:
    return Table(
        [["", Paragraph(text, style_map["Callout"])]],
        colWidths=[2.2 * mm, 161.8 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), ACCENT),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("TOPPADDING", (0, 0), (0, 0), 0),
                ("BOTTOMPADDING", (0, 0), (0, 0), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 9),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("TOPPADDING", (1, 0), (1, 0), 0),
                ("BOTTOMPADDING", (1, 0), (1, 0), 0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        ),
    )


def chart_block(drawing: Drawing, caption: str, style_map) -> KeepTogether:
    return KeepTogether([drawing, Paragraph(caption, style_map["Caption"])])


def add_axes(
    drawing: Drawing, *, maximum: float, steps: int, title: str, height: float = 120
) -> tuple[float, float, float, float]:
    left, bottom, width = 39, 29, 386
    drawing.add(
        String(
            left,
            height + 49,
            title,
            fontName="SourceSans-Semibold",
            fontSize=9.6,
            fillColor=NAVY,
        )
    )
    for index in range(steps + 1):
        value = maximum * index / steps
        y = bottom + height * index / steps
        drawing.add(Line(left, y, left + width, y, strokeColor=RULE, strokeWidth=0.4))
        drawing.add(
            String(
                left - 7,
                y - 2.2,
                f"{value:.0f}",
                fontName="SourceSans",
                fontSize=7,
                fillColor=MUTED,
                textAnchor="end",
            )
        )
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=MUTED, strokeWidth=0.7))
    return left, bottom, width, height


def make_hard_transfer_chart(data: dict[str, dict]) -> Drawing:
    drawing = Drawing(465, 185)
    left, bottom, width, height = add_axes(
        drawing,
        maximum=45,
        steps=3,
        title="Accuracy on fixed-center train -> random-position test",
    )
    config_ids = ("mlp", "cnn", "vit_p8_conv")
    labels = ("MLP", "CNN", "ViT / 8")
    fills = (GRAY, ACCENT, BLUE)
    centers = (left + width * 0.20, left + width * 0.50, left + width * 0.80)
    bar_width = 48
    for center, config_id, label, fill in zip(centers, config_ids, labels, fills):
        value = data[config_id]["accuracy"][("B", "A")]
        bar_height = height * value / 45
        drawing.add(
            Rect(
                center - bar_width / 2,
                bottom,
                bar_width,
                bar_height,
                fillColor=fill,
                strokeColor=None,
            )
        )
        drawing.add(
            String(
                center,
                bottom + bar_height + 5,
                f"{value:.1f}%",
                fontName="SourceSans-Semibold",
                fontSize=8,
                fillColor=INK,
                textAnchor="middle",
            )
        )
        drawing.add(
            String(
                center,
                bottom - 12,
                label,
                fontName="SourceSans",
                fontSize=7.8,
                fillColor=INK,
                textAnchor="middle",
            )
        )
    return drawing


def make_shift_chart(data: dict[str, dict]) -> Drawing:
    drawing = Drawing(465, 185)
    left, bottom, width, height = add_axes(
        drawing,
        maximum=100,
        steps=4,
        title="Fixed-position accuracy before and after the test distribution shifts",
    )
    config_ids = ("mlp", "cnn", "vit_p8_conv")
    labels = ("MLP", "CNN", "ViT / 8")
    centers = (left + width * 0.20, left + width * 0.50, left + width * 0.80)
    bar_width = 25
    for center, config_id, label in zip(centers, config_ids, labels):
        values = (
            data[config_id]["accuracy"][("B", "B")],
            data[config_id]["accuracy"][("B", "A")],
        )
        for offset, value, fill in (
            (-bar_width / 2, values[0], BLUE),
            (bar_width / 2, values[1], ORANGE),
        ):
            bar_height = height * value / 100
            drawing.add(
                Rect(
                    center + offset - bar_width / 2,
                    bottom,
                    bar_width,
                    bar_height,
                    fillColor=fill,
                    strokeColor=None,
                )
            )
        drop = values[0] - values[1]
        drawing.add(
            String(
                center,
                bottom + height * values[0] / 100 + 5,
                f"-{drop:.1f} pp",
                fontName="SourceSans-Semibold",
                fontSize=7.4,
                fillColor=INK,
                textAnchor="middle",
            )
        )
        drawing.add(
            String(
                center,
                bottom - 12,
                label,
                fontName="SourceSans",
                fontSize=7.8,
                fillColor=INK,
                textAnchor="middle",
            )
        )
    drawing.add(Rect(329, 171, 8, 5, fillColor=BLUE, strokeColor=None))
    drawing.add(String(341, 170, "B -> B", fontName="SourceSans", fontSize=7.2, fillColor=MUTED))
    drawing.add(Rect(382, 171, 8, 5, fillColor=ORANGE, strokeColor=None))
    drawing.add(String(394, 170, "B -> A", fontName="SourceSans", fontSize=7.2, fillColor=MUTED))
    return drawing


def make_patch_chart(data: dict[str, dict]) -> Drawing:
    drawing = Drawing(465, 185)
    left, bottom, width, height = add_axes(
        drawing,
        maximum=100,
        steps=4,
        title="Patch-size ablation under the shared ViT training budget",
    )
    config_ids = ("vit_p4_conv", "vit_p8_conv", "vit_p16_conv")
    labels = ("4", "8", "16")
    centers = (left + width * 0.20, left + width * 0.50, left + width * 0.80)
    bar_width = 25
    for center, config_id, label in zip(centers, config_ids, labels):
        values = (
            data[config_id]["accuracy"][("A", "A")],
            data[config_id]["accuracy"][("B", "A")],
        )
        for offset, value, fill in (
            (-bar_width / 2, values[0], BLUE),
            (bar_width / 2, values[1], ORANGE),
        ):
            bar_height = height * value / 100
            drawing.add(
                Rect(
                    center + offset - bar_width / 2,
                    bottom,
                    bar_width,
                    bar_height,
                    fillColor=fill,
                    strokeColor=None,
                )
            )
            drawing.add(
                String(
                    center + offset,
                    bottom + bar_height + 4,
                    f"{value:.1f}",
                    fontName="SourceSans",
                    fontSize=7.4,
                    fillColor=INK,
                    textAnchor="middle",
                )
            )
        drawing.add(
            String(
                center,
                bottom - 12,
                label,
                fontName="SourceSans",
                fontSize=7.8,
                fillColor=INK,
                textAnchor="middle",
            )
        )
    drawing.add(Rect(329, 171, 8, 5, fillColor=BLUE, strokeColor=None))
    drawing.add(String(341, 170, "A -> A", fontName="SourceSans", fontSize=7.2, fillColor=MUTED))
    drawing.add(Rect(382, 171, 8, 5, fillColor=ORANGE, strokeColor=None))
    drawing.add(String(394, 170, "B -> A", fontName="SourceSans", fontSize=7.2, fillColor=MUTED))
    return drawing


def page_heading(kicker: str, title: str, subtitle: str, style_map) -> list:
    return [
        Paragraph(kicker, style_map["Kicker"]),
        Paragraph(title, style_map["PageTitle"]),
        Paragraph(subtitle, style_map["Subtitle"]),
    ]


def accuracy_table(data: dict[str, dict], config_ids, labels, style_map) -> Table:
    rows = [
        [
            Paragraph("Configuration", style_map["TableHeader"]),
            Paragraph("Parameters", style_map["TableHeader"]),
            *[Paragraph(label, style_map["TableHeader"]) for label in SETTINGS],
        ]
    ]
    for config_id, label in zip(config_ids, labels):
        config = data[config_id]
        rows.append(
            [
                Paragraph(label, style_map["TableCell"]),
                Paragraph(f"{config['parameters']:,}", style_map["TableCellCenter"]),
                *[
                    Paragraph(
                        f"{config['accuracy'][key]:.2f}",
                        style_map["TableCellCenter"],
                    )
                    for key in SETTING_KEYS
                ],
            ]
        )
    return ruled_table(
        rows,
        [43 * mm, 25 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm],
    )


def build_story(data: dict[str, dict], style_map) -> list:
    protocol_table = ruled_table(
        [
            [
                Paragraph("Setting", style_map["TableHeader"]),
                Paragraph("Training", style_map["TableHeader"]),
                Paragraph("Testing", style_map["TableHeader"]),
                Paragraph("Purpose", style_map["TableHeader"]),
            ],
            [
                Paragraph("A -> A", style_map["TableCellStrong"]),
                Paragraph("random", style_map["TableCellCenter"]),
                Paragraph("random", style_map["TableCellCenter"]),
                Paragraph("in-distribution reference", style_map["TableCell"]),
            ],
            [
                Paragraph("B -> B", style_map["TableCellStrong"]),
                Paragraph("center", style_map["TableCellCenter"]),
                Paragraph("center", style_map["TableCellCenter"]),
                Paragraph("fixed-position reference", style_map["TableCell"]),
            ],
            [
                Paragraph("A -> B", style_map["TableCellStrong"]),
                Paragraph("random", style_map["TableCellCenter"]),
                Paragraph("center", style_map["TableCellCenter"]),
                Paragraph("random-to-fixed transfer", style_map["TableCell"]),
            ],
            [
                Paragraph("B -> A", style_map["TableCellStrong"]),
                Paragraph("center", style_map["TableCellCenter"]),
                Paragraph("random", style_map["TableCellCenter"]),
                Paragraph("hardest position shift", style_map["TableCell"]),
            ],
        ],
        [25 * mm, 31 * mm, 31 * mm, 77 * mm],
    )

    controls_table = ruled_table(
        [
            [
                Paragraph("Item", style_map["TableHeader"]),
                Paragraph("Controlled choice", style_map["TableHeader"]),
            ],
            [
                Paragraph("Data", style_map["TableCell"]),
                Paragraph("FashionMNIST on a 64 x 64 black canvas", style_map["TableCell"]),
            ],
            [
                Paragraph("Split", style_map["TableCell"]),
                Paragraph("90% train, 10% validation; official test set held out", style_map["TableCell"]),
            ],
            [
                Paragraph("Training", style_map["TableCell"]),
                Paragraph("15 epochs, AdamW, seed 42", style_map["TableCell"]),
            ],
            [
                Paragraph("Selection", style_map["TableCell"]),
                Paragraph("best validation checkpoint", style_map["TableCell"]),
            ],
            [
                Paragraph("Reporting", style_map["TableCell"]),
                Paragraph("test accuracy and elapsed training time", style_map["TableCell"]),
            ],
        ],
        [34 * mm, 130 * mm],
        compact=True,
    )

    architecture_table = accuracy_table(
        data,
        ("mlp", "cnn", "vit_p8_conv"),
        ("MLP", "CNN", "ViT, patch 8"),
        style_map,
    )

    findings = Table(
        [
            [
                Paragraph("BEST B -> A", style_map["FindingLabel"]),
                Paragraph("CNN ADVANTAGE", style_map["FindingLabel"]),
                Paragraph("SMALLEST MODEL", style_map["FindingLabel"]),
            ],
            [
                Paragraph("35.40%", style_map["FindingValue"]),
                Paragraph("+16.59 pp", style_map["FindingValue"]),
                Paragraph("205,994", style_map["FindingValue"]),
            ],
            [
                Paragraph("CNN", style_map["FindingText"]),
                Paragraph("over patch-8 ViT", style_map["FindingText"]),
                Paragraph("CNN parameters", style_map["FindingText"]),
            ],
        ],
        colWidths=[54.67 * mm] * 3,
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("LINEABOVE", (0, 0), (-1, 0), 0.55, colors.HexColor("#BAD0D5")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.55, colors.HexColor("#BAD0D5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("TOPPADDING", (0, 1), (-1, 1), 2),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 1),
                ("BOTTOMPADDING", (0, 2), (-1, 2), 6),
            ]
        ),
    )

    patch_rows = [
        [
            Paragraph("Patch", style_map["TableHeader"]),
            Paragraph("Tokens", style_map["TableHeader"]),
            Paragraph("A -> A", style_map["TableHeader"]),
            Paragraph("B -> B", style_map["TableHeader"]),
            Paragraph("A -> B", style_map["TableHeader"]),
            Paragraph("B -> A", style_map["TableHeader"]),
            Paragraph("Time", style_map["TableHeader"]),
        ]
    ]
    for patch, config_id, tokens in (
        (4, "vit_p4_conv", 256),
        (8, "vit_p8_conv", 64),
        (16, "vit_p16_conv", 16),
    ):
        config = data[config_id]
        elapsed = sum(config["train_seconds"].values()) / 60
        patch_rows.append(
            [
                Paragraph(str(patch), style_map["TableCellStrong"]),
                Paragraph(str(tokens), style_map["TableCellCenter"]),
                *[
                    Paragraph(f"{config['accuracy'][key]:.2f}", style_map["TableCellCenter"])
                    for key in SETTING_KEYS
                ],
                Paragraph(f"{elapsed:.2f} min", style_map["TableCellCenter"]),
            ]
        )
    patch_table = ruled_table(
        patch_rows,
        [18 * mm, 21 * mm, 23 * mm, 23 * mm, 23 * mm, 23 * mm, 33 * mm],
    )

    embedding_rows = [
        [
            Paragraph("Test", style_map["TableHeader"]),
            Paragraph("Conv2d", style_map["TableHeader"]),
            Paragraph("Flatten + Linear", style_map["TableHeader"]),
            Paragraph("Absolute gap", style_map["TableHeader"]),
        ]
    ]
    for label, key in zip(SETTINGS, SETTING_KEYS):
        conv = data["vit_p16_conv"]["accuracy"][key]
        linear = data["vit_p16_linear"]["accuracy"][key]
        embedding_rows.append(
            [
                Paragraph(label, style_map["TableCell"]),
                Paragraph(f"{conv:.2f}", style_map["TableCellCenter"]),
                Paragraph(f"{linear:.2f}", style_map["TableCellCenter"]),
                Paragraph(f"{abs(conv - linear):.2f}", style_map["TableCellCenter"]),
            ]
        )
    embedding_table = ruled_table(embedding_rows, [41 * mm] * 4)

    conclusions = Table(
        [
            [
                Paragraph("01", style_map["FindingValue"]),
                Paragraph(
                    "<b>CNN performs best under the shared recipe.</b> It leads all four "
                    "settings and retains the most fixed-to-random accuracy.",
                    style_map["BodySmall"],
                ),
            ],
            [
                Paragraph("02", style_map["FindingValue"]),
                Paragraph(
                    "<b>Patch size 8 leads three of four settings.</b> It gives the best "
                    "B -> A result without the 256-token cost of patch size 4.",
                    style_map["BodySmall"],
                ),
            ],
            [
                Paragraph("03", style_map["FindingValue"]),
                Paragraph(
                    "<b>Equivalent embeddings stay close in this run.</b> Conv2d and "
                    "Flatten + Linear differ by at most 0.61 percentage points.",
                    style_map["BodySmall"],
                ),
            ],
        ],
        colWidths=[20 * mm, 144 * mm],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 8),
                ("RIGHTPADDING", (1, 0), (1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        ),
    )

    limitations = Table(
        [
            [
                [
                    Paragraph("Scope", style_map["MiniTitle"]),
                    Paragraph(
                        "- One training and placement seed; no variance estimate.<br/>"
                        "- Model sizes and optimal hyperparameters are not matched.<br/>"
                        "- Runtime is hardware- and order-specific.",
                        style_map["BodySmall"],
                    ),
                ],
                [
                    Paragraph("Interpretation rule", style_map["MiniTitle"]),
                    Paragraph(
                        "Small differences are described only as numerically close in this "
                        "run; no significance claim is made. Conclusions apply to the "
                        "shared protocol rather than all possible training recipes.",
                        style_map["BodySmall"],
                    ),
                ],
            ]
        ],
        colWidths=[80 * mm, 80 * mm],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEABOVE", (0, 0), (-1, 0), 0.55, RULE),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 7),
                ("LEFTPADDING", (1, 0), (1, 0), 7),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        ),
    )

    reproducibility = ruled_table(
        [
            [
                Paragraph("Check", style_map["TableHeader"]),
                Paragraph("Recorded value", style_map["TableHeader"]),
            ],
            [
                Paragraph("Code", style_map["TableCell"]),
                Paragraph("translated_fashionmnist/experiments", style_map["TableCell"]),
            ],
            [
                Paragraph("Metrics", style_map["TableCell"]),
                Paragraph("results/metrics.csv", style_map["TableCell"]),
            ],
            [
                Paragraph("Seed / epochs", style_map["TableCell"]),
                Paragraph("42 / 15", style_map["TableCell"]),
            ],
            [
                Paragraph("Dependency lock", style_map["TableCell"]),
                Paragraph("requirements-lock.txt", style_map["TableCell"]),
            ],
            [
                Paragraph("Environment", style_map["TableCell"]),
                Paragraph(
                    "Python 3.12.13, PyTorch 2.11.0, CUDA 12.8",
                    style_map["TableCell"],
                ),
            ],
            [
                Paragraph("Hardware", style_map["TableCell"]),
                Paragraph("NVIDIA GeForce RTX 5070 Laptop GPU", style_map["TableCell"]),
            ],
        ],
        [40 * mm, 124 * mm],
        compact=True,
    )

    implementation = ruled_table(
        [
            [
                Paragraph("Component", style_map["TableHeader"]),
                Paragraph("Location", style_map["TableHeader"]),
                Paragraph("Role", style_map["TableHeader"]),
            ],
            [
                Paragraph("Dataset", style_map["TableCell"]),
                Paragraph("data.py", style_map["TableCell"]),
                Paragraph("construct A and B canvases", style_map["TableCell"]),
            ],
            [
                Paragraph("Models", style_map["TableCell"]),
                Paragraph("models.py", style_map["TableCell"]),
                Paragraph("MLP, CNN, ViT, patch embeddings", style_map["TableCell"]),
            ],
            [
                Paragraph("Protocol", style_map["TableCell"]),
                Paragraph("experiments/protocol.py", style_map["TableCell"]),
                Paragraph("train, validate, and evaluate", style_map["TableCell"]),
            ],
            [
                Paragraph("Engine", style_map["TableCell"]),
                Paragraph("engine.py", style_map["TableCell"]),
                Paragraph("shared training and evaluation primitives", style_map["TableCell"]),
            ],
            [
                Paragraph("Runner", style_map["TableCell"]),
                Paragraph("experiments/compare.py", style_map["TableCell"]),
                Paragraph("execute all three comparison groups", style_map["TableCell"]),
            ],
            [
                Paragraph("Outputs", style_map["TableCell"]),
                Paragraph("results/", style_map["TableCell"]),
                Paragraph("metrics, history, and figures", style_map["TableCell"]),
            ],
        ],
        [31 * mm, 57 * mm, 76 * mm],
        compact=True,
    )

    abstract = side_note(
        "<i>Abstract.</i> We test whether image classifiers retain accuracy when object "
        "position changes between training and testing. Under one shared training recipe, "
        "CNN records the highest accuracy and reaches 35.40% on the hardest fixed-to-random "
        "transfer. Patch size 8 leads three of four ViT settings. Equivalent Conv2d and "
        "Flatten + Linear projections remain within 0.61 percentage points in this run.",
        style_map,
    )

    story = [
        Paragraph("COURSE PROJECT · COMPARISON STUDY", style_map["Kicker"]),
        Paragraph("Position Generalization in<br/>Translated FashionMNIST", style_map["Title"]),
        Paragraph(
            "Architecture, patch scale, and patch embedding",
            style_map["Subtitle"],
        ),
        Table(
            [
                [
                    Paragraph("Final English report", style_map["Meta"]),
                    Paragraph("Controlled experiments · seed 42", style_map["Meta"]),
                ]
            ],
            colWidths=[82 * mm, 82 * mm],
            style=TableStyle(
                [
                    ("LINEABOVE", (0, 0), (-1, 0), 0.7, NAVY),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.35, RULE),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            ),
        ),
        Spacer(1, 4 * mm),
        abstract,
        section_header("1", "Question", style_map),
        Paragraph(
            "A classifier may learn class appearance and object location together. "
            "This study isolates the effect of location by comparing random-position "
            "images (A) with centered images (B).",
            style_map["Body"],
        ),
        protocol_table,
        section_header("2", "Controlled protocol", style_map),
        Paragraph(
            "Each configuration trains one model on A and one on B. Validation chooses "
            "the checkpoint; the official test set is evaluated only after selection.",
            style_map["Body"],
        ),
        controls_table,
        Spacer(1, 3 * mm),
        side_note(
            "<b>Primary measure.</b> B -> A is the hardest setting because a model "
            "trained only on centered objects must classify objects across the canvas.",
            style_map,
        ),
        PageBreak(),
        *page_heading(
            "MODEL COMPARISON",
            "Architecture and Position Shift",
            "MLP, CNN, and patch-8 ViT under one evaluation protocol",
            style_map,
        ),
        section_header("3", "Accuracy", style_map),
        architecture_table,
        Spacer(1, 2.5 * mm),
        Paragraph(
            "Table 1. Test accuracy (%). The best value in every column belongs to CNN.",
            style_map["Caption"],
        ),
        chart_block(
            make_hard_transfer_chart(data),
            "Figure 1. CNN retains substantially more accuracy on B -> A.",
            style_map,
        ),
        findings,
        section_header("4", "Effect of the position shift", style_map),
        chart_block(
            make_shift_chart(data),
            "Figure 2. All models degrade after centered training is tested at random positions.",
            style_map,
        ),
        PageBreak(),
        *page_heading(
            "VIT ABLATIONS",
            "Patch Scale and Embedding",
            "Two controlled changes to the shared Transformer configuration",
            style_map,
        ),
        section_header("5", "Patch scale", style_map),
        chart_block(
            make_patch_chart(data),
            "Figure 3. Patch size 8 leads three of four evaluated settings.",
            style_map,
        ),
        patch_table,
        Spacer(1, 2.5 * mm),
        Paragraph(
            "Table 2. Accuracy (%) and total time for the A- and B-trained models.",
            style_map["Caption"],
        ),
        side_note(
            "<b>Result.</b> Patch size 8 leads on A -> A, A -> B, and B -> A. "
            "Patch size 4 raises the token count to 256 and is slower without an "
            "accuracy gain.",
            style_map,
        ),
        section_header("6", "Patch embedding", style_map),
        Paragraph(
            "For non-overlapping patches, a strided Conv2d and a shared linear layer "
            "perform the same type of patch-wise projection. Their results remain close.",
            style_map["Body"],
        ),
        embedding_table,
        Spacer(1, 2.5 * mm),
        Paragraph(
            "Table 3. Test accuracy (%) for patch size 16. Maximum absolute gap: 0.61 points.",
            style_map["Caption"],
        ),
        PageBreak(),
        *page_heading(
            "DISCUSSION",
            "Conclusions and Reproducibility",
            "Concise interpretation and a documented reproduction path",
            style_map,
        ),
        section_header("7", "Conclusions", style_map),
        conclusions,
        section_header("8", "Limitations", style_map),
        limitations,
        section_header("9", "Reproducibility", style_map),
        reproducibility,
        Spacer(1, 2 * mm),
        Paragraph(
            "python -m translated_fashionmnist.experiments.compare --groups all --download",
            style_map["Code"],
        ),
        section_header("10", "Implementation map", style_map),
        implementation,
    ]
    return story


def build() -> Path:
    register_fonts()
    data = load_metrics()
    style_map = build_styles()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ReportDocument(str(OUTPUT_PATH)).build(build_story(data, style_map))
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build())
