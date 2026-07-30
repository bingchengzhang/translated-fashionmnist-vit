"""Build the eight-page English comparison report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
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
METRICS_PATH = ROOT / "results" / "metrics.csv"
HISTORY_PATH = ROOT / "results" / "training_history.csv"
MANIFEST_PATH = ROOT / "results" / "manifest.json"
OUTPUT_PATH = ROOT / "reports" / "comparison_study.pdf"
SAMPLE_PATH = ROOT / "reports" / "data_samples.png"
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
REPORT_CONFIG_IDS = {
    "mlp",
    "cnn",
    "vit_p16_conv",
    "vit_p8_conv",
    "vit_p4_conv",
    "vit_p16_linear",
}


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
    seen: set[tuple[str, str, str]] = set()
    row_count = 0
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            key = (row["config_id"], row["train_mode"], row["test_mode"])
            if key in seen:
                raise ValueError(f"Duplicate metric row: {key}")
            seen.add(key)
            config = configs.setdefault(
                row["config_id"],
                {
                    "name": row["display_name"],
                    "parameters": int(row["parameter_count"]),
                    "train_seconds": {},
                    "accuracy": {},
                },
            )
            setting = (row["train_mode"], row["test_mode"])
            accuracy = 100 * float(row["test_accuracy"])
            if not 0 <= accuracy <= 100:
                raise ValueError(f"Accuracy outside [0, 100]: {key}")
            config["accuracy"][setting] = accuracy
            config["train_seconds"][row["train_mode"]] = float(
                row["train_elapsed_seconds"]
            )
    if row_count != 24 or set(configs) != REPORT_CONFIG_IDS:
        raise ValueError("Expected six configurations and 24 metric rows.")
    for config_id, config in configs.items():
        if set(config["accuracy"]) != set(SETTING_KEYS):
            raise ValueError(f"Incomplete settings for {config_id}.")
        if set(config["train_seconds"]) != {"A", "B"}:
            raise ValueError(f"Incomplete training times for {config_id}.")
    return configs


def load_training_history() -> dict[tuple[str, str], list[dict[str, float]]]:
    """Load and validate the 12 recorded 15-epoch training histories."""
    histories: dict[tuple[str, str], list[dict[str, float]]] = {}
    seen: set[tuple[str, str, int]] = set()
    with HISTORY_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["config_id"], row["train_mode"])
            epoch = int(row["epoch"])
            record_key = (*key, epoch)
            if record_key in seen:
                raise ValueError(f"Duplicate history row: {record_key}")
            seen.add(record_key)
            histories.setdefault(key, []).append(
                {
                    "epoch": epoch,
                    "train_accuracy": 100 * float(row["train_accuracy"]),
                    "val_accuracy": 100 * float(row["val_accuracy"]),
                }
            )
    expected_keys = {
        (config_id, mode)
        for config_id in REPORT_CONFIG_IDS
        for mode in ("A", "B")
    }
    if set(histories) != expected_keys or len(seen) != 180:
        raise ValueError("Expected 12 complete 15-epoch training histories.")
    for key, records in histories.items():
        records.sort(key=lambda record: record["epoch"])
        if [record["epoch"] for record in records] != list(range(1, 16)):
            raise ValueError(f"Non-contiguous history for {key}.")
    return histories


def load_manifest() -> dict:
    """Load the recorded protocol and environment used by the report."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if set(manifest.get("configurations", [])) != REPORT_CONFIG_IDS:
        raise ValueError("Manifest does not describe the six report configurations.")
    if manifest.get("fit_count") != 12 or manifest.get("evaluation_count") != 24:
        raise ValueError("Manifest must record 12 fits and 24 evaluations.")
    required_sections = {"protocol", "method", "environment"}
    if not required_sections.issubset(manifest):
        raise ValueError("Manifest is missing protocol, method, or environment details.")
    required_protocol = {
        "canvas_size",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "val_fraction",
        "seed",
        "amp",
    }
    required_method = {"loss", "optimizer", "scheduler", "checkpoint"}
    required_environment = {"torch", "cuda", "gpu", "platform"}
    if not required_protocol.issubset(manifest["protocol"]):
        raise ValueError("Manifest protocol is incomplete.")
    if not required_method.issubset(manifest["method"]):
        raise ValueError("Manifest method is incomplete.")
    if not required_environment.issubset(manifest["environment"]):
        raise ValueError("Manifest environment is incomplete.")
    return manifest


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
        "TableCellBest": ParagraphStyle(
            "TableCellBest",
            parent=base["Normal"],
            fontName="SourceSans-Semibold",
            fontSize=8.15,
            leading=10.3,
            alignment=TA_CENTER,
            textColor=ACCENT,
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
            subject="Architecture, patch scale, and patch projection comparisons",
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
        self.addPageTemplates(
            [PageTemplate(id="academic", frames=frame, onPageEnd=self.decorate)]
        )

    def decorate(self, canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("SourceSans", 7.5)
        canvas.setFillColor(MUTED)
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


def finding_strip(
    entries: list[tuple[str, str, str]],
    style_map,
) -> Table:
    """Create a compact row of evidence cards."""
    return Table(
        [
            [Paragraph(label, style_map["FindingLabel"]) for label, _, _ in entries],
            [Paragraph(value, style_map["FindingValue"]) for _, value, _ in entries],
            [Paragraph(note, style_map["FindingText"]) for _, _, note in entries],
        ],
        colWidths=[164 * mm / len(entries)] * len(entries),
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


def chart_block(drawing: Drawing, caption: str, style_map) -> KeepTogether:
    return KeepTogether([drawing, Paragraph(caption, style_map["Caption"])])


def make_protocol_flow() -> Drawing:
    """Show how each configuration produces its four reported measurements."""
    drawing = Drawing(164 * mm, 35 * mm)
    steps = (
        ("FIXED SPLIT", "54,000 train", "6,000 validation"),
        ("POSITION RULE", "fit on A or B", "fixed per sample"),
        ("OPTIMIZE", "15 epochs", "AdamW + cosine"),
        ("SELECT", "best validation", "checkpoint"),
        ("FINAL TEST", "evaluate on A", "and on B"),
    )
    box_width = 75
    box_height = 55
    gap = (164 * mm - len(steps) * box_width) / (len(steps) - 1)
    y = 23
    for index, (title, line_one, line_two) in enumerate(steps):
        x = index * (box_width + gap)
        drawing.add(
            Rect(
                x,
                y,
                box_width,
                box_height,
                rx=4,
                ry=4,
                fillColor=PALE,
                strokeColor=RULE,
                strokeWidth=0.55,
            )
        )
        drawing.add(
            String(
                x + box_width / 2,
                y + 39,
                title,
                fontName="SourceSans-Semibold",
                fontSize=7.2,
                fillColor=ACCENT,
                textAnchor="middle",
            )
        )
        for offset, line in ((24, line_one), (12, line_two)):
            drawing.add(
                String(
                    x + box_width / 2,
                    y + offset,
                    line,
                    fontName="SourceSans",
                    fontSize=7.2,
                    fillColor=INK,
                    textAnchor="middle",
                )
            )
        if index < len(steps) - 1:
            start = x + box_width + 3
            end = x + box_width + gap - 3
            center_y = y + box_height / 2
            drawing.add(
                Line(start, center_y, end, center_y, strokeColor=GRAY, strokeWidth=0.9)
            )
            drawing.add(
                PolyLine(
                    [
                        (end - 4, center_y + 3),
                        (end, center_y),
                        (end - 4, center_y - 3),
                    ],
                    strokeColor=GRAY,
                    strokeWidth=0.9,
                    fillColor=None,
                )
            )
    drawing.add(
        String(
            82 * mm,
            5,
            "One configuration: two fits and four held-out test measurements",
            fontName="SourceSans",
            fontSize=7.2,
            fillColor=MUTED,
            textAnchor="middle",
        )
    )
    return drawing


def make_vit_pipeline() -> Drawing:
    """Summarize the exact ViT used in the controlled ablations."""
    drawing = Drawing(164 * mm, 52 * mm)
    steps = (
        ("IMAGE", "1 x 64 x 64", "grayscale"),
        ("PATCH TOKENS", "p in {4, 8, 16}", "d = 128"),
        ("SEQUENCE", "prepend [CLS]", "+ learned abs. pos."),
        ("ENCODER x 4", "4-head attention", "FFN 512 / GELU"),
        ("CLASSIFIER", "[CLS] vector", "linear -> 10"),
    )
    box_width = 82
    box_height = 72
    gap = (164 * mm - len(steps) * box_width) / (len(steps) - 1)
    y = 35
    for index, (title, line_one, line_two) in enumerate(steps):
        x = index * (box_width + gap)
        fill = PALE_BLUE if index in (2, 3) else PALE
        drawing.add(
            Rect(
                x,
                y,
                box_width,
                box_height,
                rx=4,
                ry=4,
                fillColor=fill,
                strokeColor=RULE,
                strokeWidth=0.55,
            )
        )
        drawing.add(
            String(
                x + box_width / 2,
                y + 52,
                title,
                fontName="SourceSans-Semibold",
                fontSize=7.3,
                fillColor=ACCENT,
                textAnchor="middle",
            )
        )
        drawing.add(
            String(
                x + box_width / 2,
                y + 32,
                line_one,
                fontName="SourceSerif-Semibold",
                fontSize=8.7,
                fillColor=INK,
                textAnchor="middle",
            )
        )
        drawing.add(
            String(
                x + box_width / 2,
                y + 17,
                line_two,
                fontName="SourceSans",
                fontSize=7,
                fillColor=MUTED,
                textAnchor="middle",
            )
        )
        if index < len(steps) - 1:
            start = x + box_width + 2
            end = x + box_width + gap - 2
            center_y = y + box_height / 2
            drawing.add(
                Line(start, center_y, end, center_y, strokeColor=GRAY, strokeWidth=0.9)
            )
            drawing.add(
                PolyLine(
                    [
                        (end - 4, center_y + 3),
                        (end, center_y),
                        (end - 4, center_y - 3),
                    ],
                    strokeColor=GRAY,
                    strokeWidth=0.9,
                    fillColor=None,
                )
            )
    drawing.add(
        String(
            82 * mm,
            15,
            "Pre-norm Transformer; dropout 0.1; non-overlapping patches",
            fontName="SourceSans",
            fontSize=7.4,
            fillColor=MUTED,
            textAnchor="middle",
        )
    )
    return drawing


def make_embedding_equivalence() -> Drawing:
    """Contrast the two patch-projection implementations used at p=16."""
    drawing = Drawing(164 * mm, 39 * mm)
    panel_width = 79 * mm
    panels = (
        (
            0,
            "CONV2D PROJECTION",
            "kernel = stride = 16",
            "1 x 16 x 16 -> 128 channels",
        ),
        (
            85 * mm,
            "FLATTEN + LINEAR",
            "unfold into 16 x 16 patches",
            "256-vector -> shared Linear(256,128)",
        ),
    )
    for x, title, line_one, line_two in panels:
        drawing.add(
            Rect(
                x,
                31,
                panel_width,
                64,
                rx=4,
                ry=4,
                fillColor=PALE,
                strokeColor=RULE,
                strokeWidth=0.55,
            )
        )
        drawing.add(
            String(
                x + panel_width / 2,
                74,
                title,
                fontName="SourceSans-Semibold",
                fontSize=7.4,
                fillColor=ACCENT,
                textAnchor="middle",
            )
        )
        drawing.add(
            String(
                x + panel_width / 2,
                55,
                line_one,
                fontName="SourceSerif-Semibold",
                fontSize=8.7,
                fillColor=INK,
                textAnchor="middle",
            )
        )
        drawing.add(
            String(
                x + panel_width / 2,
                40,
                line_two,
                fontName="SourceSans",
                fontSize=7.2,
                fillColor=MUTED,
                textAnchor="middle",
            )
        )
    drawing.add(
        String(
            82 * mm,
            11,
            "Both return 16 tokens x 128 dimensions and represent the same affine map",
            fontName="SourceSans",
            fontSize=7.5,
            fillColor=NAVY,
            textAnchor="middle",
        )
    )
    return drawing


def make_training_curve_chart(
    histories: dict[tuple[str, str], list[dict[str, float]]],
) -> Drawing:
    """Plot validation accuracy for the three architecture baselines."""
    drawing = Drawing(164 * mm, 88 * mm)
    series = (
        ("mlp", "MLP", GRAY),
        ("cnn", "CNN", ACCENT),
        ("vit_p8_conv", "ViT / 8", BLUE),
    )
    for index, (_, label, color) in enumerate(series):
        legend_x = 139 + index * 70
        drawing.add(
            Line(
                legend_x,
                239,
                legend_x + 17,
                239,
                strokeColor=color,
                strokeWidth=2.1,
            )
        )
        drawing.add(
            String(
                legend_x + 22,
                236.5,
                label,
                fontName="SourceSans",
                fontSize=7.3,
                fillColor=MUTED,
            )
        )

    for mode, left in (("A", 37), ("B", 269)):
        bottom, plot_width, plot_height = 36, 159, 164
        drawing.add(
            String(
                left + plot_width / 2,
                218,
                f"Validation accuracy - train {mode}",
                fontName="SourceSans-Semibold",
                fontSize=9.3,
                fillColor=NAVY,
                textAnchor="middle",
            )
        )
        for accuracy in (0, 25, 50, 75, 100):
            y = bottom + plot_height * accuracy / 100
            drawing.add(
                Line(
                    left,
                    y,
                    left + plot_width,
                    y,
                    strokeColor=RULE,
                    strokeWidth=0.4,
                )
            )
            drawing.add(
                String(
                    left - 7,
                    y - 2.2,
                    str(accuracy),
                    fontName="SourceSans",
                    fontSize=6.8,
                    fillColor=MUTED,
                    textAnchor="end",
                )
            )
        for epoch in (1, 5, 10, 15):
            x = left + plot_width * (epoch - 1) / 14
            drawing.add(
                String(
                    x,
                    21,
                    str(epoch),
                    fontName="SourceSans",
                    fontSize=6.8,
                    fillColor=MUTED,
                    textAnchor="middle",
                )
            )
        drawing.add(
            String(
                left + plot_width / 2,
                7,
                "epoch",
                fontName="SourceSans",
                fontSize=6.8,
                fillColor=MUTED,
                textAnchor="middle",
            )
        )
        for config_id, _, color in series:
            points = [
                (
                    left + plot_width * (record["epoch"] - 1) / 14,
                    bottom + plot_height * record["val_accuracy"] / 100,
                )
                for record in histories[(config_id, mode)]
            ]
            drawing.add(
                PolyLine(
                    points,
                    strokeColor=color,
                    strokeWidth=1.7,
                    fillColor=None,
                )
            )
            final_x, final_y = points[-1]
            drawing.add(
                Circle(
                    final_x,
                    final_y,
                    2.1,
                    fillColor=color,
                    strokeColor=colors.white,
                    strokeWidth=0.4,
                )
            )
    return drawing


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
    drawing = Drawing(465, 235)
    left, bottom, width, height = add_axes(
        drawing,
        maximum=45,
        steps=3,
        title="Accuracy on fixed-center train -> random-position test",
        height=170,
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
    drawing = Drawing(465, 235)
    left, bottom, width, height = add_axes(
        drawing,
        maximum=100,
        steps=4,
        title="Random-position test accuracy by training distribution",
        height=170,
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
    drawing.add(Rect(329, 221, 8, 5, fillColor=BLUE, strokeColor=None))
    drawing.add(String(341, 220, "A -> A", fontName="SourceSans", fontSize=7.2, fillColor=MUTED))
    drawing.add(Rect(382, 221, 8, 5, fillColor=ORANGE, strokeColor=None))
    drawing.add(String(394, 220, "B -> A", fontName="SourceSans", fontSize=7.2, fillColor=MUTED))
    return drawing


def page_heading(kicker: str, title: str, subtitle: str, style_map) -> list:
    return [
        Paragraph(kicker, style_map["Kicker"]),
        Paragraph(title, style_map["PageTitle"]),
        Paragraph(subtitle, style_map["Subtitle"]),
    ]


def accuracy_table(data: dict[str, dict], config_ids, labels, style_map) -> Table:
    best_by_setting = {
        key: max(data[config_id]["accuracy"][key] for config_id in config_ids)
        for key in SETTING_KEYS
    }
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
                        (
                            style_map["TableCellBest"]
                            if config["accuracy"][key] == best_by_setting[key]
                            else style_map["TableCellCenter"]
                        ),
                    )
                    for key in SETTING_KEYS
                ],
            ]
        )
    return ruled_table(
        rows,
        [43 * mm, 25 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm],
    )


def build_story(
    data: dict[str, dict],
    histories: dict[tuple[str, str], list[dict[str, float]]],
    manifest: dict,
    style_map,
) -> list:
    protocol = manifest["protocol"]
    method = manifest["method"]
    environment = manifest["environment"]
    configuration_count = len(manifest["configurations"])
    fit_count = int(manifest["fit_count"])
    evaluation_count = int(manifest["evaluation_count"])
    validation_samples = round(60_000 * float(protocol["val_fraction"]))
    training_samples = 60_000 - validation_samples

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
                Paragraph("transfer within A's position support", style_map["TableCell"]),
            ],
            [
                Paragraph("B -> A", style_map["TableCellStrong"]),
                Paragraph("center", style_map["TableCellCenter"]),
                Paragraph("random", style_map["TableCellCenter"]),
                Paragraph("one-to-many position transfer", style_map["TableCell"]),
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
                Paragraph(
                    f"28 x 28 FashionMNIST image on a {protocol['canvas_size']} x "
                    f"{protocol['canvas_size']} black canvas",
                    style_map["TableCell"],
                ),
            ],
            [
                Paragraph("Split", style_map["TableCell"]),
                Paragraph(
                    f"{training_samples:,} train, {validation_samples:,} validation; "
                    "official 10,000-image test set held out",
                    style_map["TableCell"],
                ),
            ],
            [
                Paragraph("Training", style_map["TableCell"]),
                Paragraph(
                    f"{protocol['epochs']} epochs; batch {protocol['batch_size']}; "
                    f"{method['loss']}; {method['optimizer']}, "
                    f"lr {float(protocol['learning_rate']):g}, "
                    f"weight decay {float(protocol['weight_decay']):g}",
                    style_map["TableCell"],
                ),
            ],
            [
                Paragraph("Selection", style_map["TableCell"]),
                Paragraph(
                    f"{method['scheduler']}; {method['checkpoint']} checkpoint",
                    style_map["TableCell"],
                ),
            ],
            [
                Paragraph("Reporting", style_map["TableCell"]),
                Paragraph(
                    f"top-1 test accuracy; elapsed training time; seed {protocol['seed']}",
                    style_map["TableCell"],
                ),
            ],
        ],
        [34 * mm, 130 * mm],
    )

    comparison_table = ruled_table(
        [
            [
                Paragraph("Group", style_map["TableHeader"]),
                Paragraph("Configurations", style_map["TableHeader"]),
                Paragraph("Changed factor / fixed factors", style_map["TableHeader"]),
            ],
            [
                Paragraph("Architecture", style_map["TableCellStrong"]),
                Paragraph("MLP, CNN, ViT p=8", style_map["TableCell"]),
                Paragraph(
                    "Architecture changes; split, optimizer, budget, and evaluation stay fixed.",
                    style_map["TableCell"],
                ),
            ],
            [
                Paragraph("Patch scale", style_map["TableCellStrong"]),
                Paragraph("ViT p=4, 8, 16", style_map["TableCell"]),
                Paragraph(
                    "Token grid changes; Transformer width, depth, heads, and projection type stay fixed.",
                    style_map["TableCell"],
                ),
            ],
            [
                Paragraph("Projection", style_map["TableCellStrong"]),
                Paragraph("Conv2d, Flatten + Linear", style_map["TableCell"]),
                Paragraph(
                    "Patch projection changes at p=16; the Transformer body is identical.",
                    style_map["TableCell"],
                ),
            ],
        ],
        [30 * mm, 48 * mm, 86 * mm],
        compact=True,
    )

    model_table = ruled_table(
        [
            [
                Paragraph("Model", style_map["TableHeader"]),
                Paragraph("Core structure", style_map["TableHeader"]),
                Paragraph("Spatial treatment", style_map["TableHeader"]),
                Paragraph("Parameters", style_map["TableHeader"]),
            ],
            [
                Paragraph("MLP", style_map["TableCellStrong"]),
                Paragraph(
                    "4096 -> 128 -> 128 -> 10; LayerNorm; GELU; dropout 0.1",
                    style_map["TableCell"],
                ),
                Paragraph("absolute input pixels", style_map["TableCell"]),
                Paragraph(
                    f"{data['mlp']['parameters']:,}",
                    style_map["TableCellCenter"],
                ),
            ],
            [
                Paragraph("CNN", style_map["TableCellStrong"]),
                Paragraph(
                    "channels 1-32-32-64-64-128; BatchNorm; two max-pools; "
                    "adaptive 2 x 2 pool; 512 -> 128 -> 10",
                    style_map["TableCell"],
                ),
                Paragraph("shared local filters", style_map["TableCell"]),
                Paragraph(
                    f"{data['cnn']['parameters']:,}",
                    style_map["TableCellCenter"],
                ),
            ],
            [
                Paragraph("ViT", style_map["TableCellStrong"]),
                Paragraph(
                    "d=128; depth 4; heads 4; FFN 512; pre-norm; GELU; dropout 0.1",
                    style_map["TableCell"],
                ),
                Paragraph("learned absolute position embedding", style_map["TableCell"]),
                Paragraph(
                    f"p4 {data['vit_p4_conv']['parameters']:,}<br/>"
                    f"p8 {data['vit_p8_conv']['parameters']:,}<br/>"
                    f"p16 {data['vit_p16_conv']['parameters']:,}",
                    style_map["TableCellCenter"],
                ),
            ],
        ],
        [20 * mm, 72 * mm, 43 * mm, 29 * mm],
    )

    model_ids = ("mlp", "cnn", "vit_p8_conv")
    model_labels = ("MLP", "CNN", "ViT, patch 8")
    model_display = dict(zip(model_ids, model_labels))
    architecture_table = accuracy_table(
        data,
        model_ids,
        model_labels,
        style_map,
    )
    best_validation_rows = [
        [
            Paragraph("Configuration", style_map["TableHeader"]),
            Paragraph("Train A", style_map["TableHeader"]),
            Paragraph("Train B", style_map["TableHeader"]),
        ]
    ]
    best_validation_epochs: list[int] = []
    best_validation_by_run: dict[tuple[str, str], dict[str, float]] = {}
    for config_id, label in zip(model_ids, model_labels):
        cells = [Paragraph(label, style_map["TableCell"])]
        for mode in ("A", "B"):
            best = max(
                histories[(config_id, mode)],
                key=lambda record: record["val_accuracy"],
            )
            best_validation_by_run[(config_id, mode)] = best
            best_validation_epochs.append(int(best["epoch"]))
            cells.append(
                Paragraph(
                    f"{best['val_accuracy']:.2f}% (epoch {int(best['epoch'])})",
                    style_map["TableCellCenter"],
                )
            )
        best_validation_rows.append(cells)
    best_validation_table = ruled_table(
        best_validation_rows,
        [54 * mm, 55 * mm, 55 * mm],
    )
    highest_validation_run = max(
        best_validation_by_run,
        key=lambda key: best_validation_by_run[key]["val_accuracy"],
    )
    highest_validation = best_validation_by_run[highest_validation_run]["val_accuracy"]
    mlp_validation_gap = (
        best_validation_by_run[("mlp", "B")]["val_accuracy"]
        - best_validation_by_run[("mlp", "A")]["val_accuracy"]
    )
    training_findings = finding_strip(
        [
            (
                "SELECTED EPOCHS",
                f"{min(best_validation_epochs)}-{max(best_validation_epochs)}",
                "all six architecture fits",
            ),
            (
                "BEST VALIDATION",
                f"{highest_validation:.2f}%",
                f"{model_display[highest_validation_run[0]]} / train "
                f"{highest_validation_run[1]}",
            ),
            (
                "LARGEST A/B GAP",
                f"{mlp_validation_gap:.2f} pp",
                "MLP validation",
            ),
        ],
        style_map,
    )
    random_to_center_max_change = max(
        abs(
            data[config_id]["accuracy"][("A", "B")]
            - data[config_id]["accuracy"][("A", "A")]
        )
        for config_id in model_ids
    )
    all_config_ids = (
        "mlp",
        "cnn",
        "vit_p4_conv",
        "vit_p8_conv",
        "vit_p16_conv",
        "vit_p16_linear",
    )
    all_random_to_center_max_change = max(
        abs(
            data[config_id]["accuracy"][("A", "B")]
            - data[config_id]["accuracy"][("A", "A")]
        )
        for config_id in all_config_ids
    )
    hard_transfer_drops = {
        config_id: (
            data[config_id]["accuracy"][("B", "B")]
            - data[config_id]["accuracy"][("B", "A")]
        )
        for config_id in all_config_ids
    }
    hard_drop_min = min(hard_transfer_drops.values())
    hard_drop_max = max(hard_transfer_drops.values())
    cnn_hard_accuracy = data["cnn"]["accuracy"][("B", "A")]
    cnn_advantage = (
        cnn_hard_accuracy - data["vit_p8_conv"]["accuracy"][("B", "A")]
    )

    findings = finding_strip(
        [
            ("B -> A LEADER", f"{cnn_hard_accuracy:.2f}%", "CNN"),
            (
                "SMALLEST B -> A DROP",
                f"{hard_drop_min:.2f} pp",
                "CNN, relative to B -> B",
            ),
            (
                "A -> B CHANGE",
                f"{random_to_center_max_change:.2f} pp max",
                "all three architectures",
            ),
        ],
        style_map,
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
    patch_specs = (
        (4, "vit_p4_conv", 256),
        (8, "vit_p8_conv", 64),
        (16, "vit_p16_conv", 16),
    )
    patch_best = {
        key: max(data[config_id]["accuracy"][key] for _, config_id, _ in patch_specs)
        for key in SETTING_KEYS
    }
    patch_times = {
        config_id: sum(data[config_id]["train_seconds"].values()) / 60
        for _, config_id, _ in patch_specs
    }
    fastest_time = min(patch_times.values())
    for patch, config_id, tokens in patch_specs:
        config = data[config_id]
        elapsed = patch_times[config_id]
        patch_rows.append(
            [
                Paragraph(str(patch), style_map["TableCellStrong"]),
                Paragraph(str(tokens), style_map["TableCellCenter"]),
                *[
                    Paragraph(
                        f"{config['accuracy'][key]:.2f}",
                        (
                            style_map["TableCellBest"]
                            if config["accuracy"][key] == patch_best[key]
                            else style_map["TableCellCenter"]
                        ),
                    )
                    for key in SETTING_KEYS
                ],
                Paragraph(
                    f"{elapsed:.2f} min",
                    (
                        style_map["TableCellBest"]
                        if elapsed == fastest_time
                        else style_map["TableCellCenter"]
                    ),
                ),
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
    embedding_gaps: list[float] = []
    for label, key in zip(SETTINGS, SETTING_KEYS):
        conv = data["vit_p16_conv"]["accuracy"][key]
        linear = data["vit_p16_linear"]["accuracy"][key]
        gap = abs(conv - linear)
        embedding_gaps.append(gap)
        embedding_rows.append(
            [
                Paragraph(label, style_map["TableCell"]),
                Paragraph(f"{conv:.2f}", style_map["TableCellCenter"]),
                Paragraph(f"{linear:.2f}", style_map["TableCellCenter"]),
                Paragraph(f"{gap:.2f}", style_map["TableCellCenter"]),
            ]
        )
    embedding_table = ruled_table(embedding_rows, [41 * mm] * 4)
    embedding_gap_max = max(embedding_gaps)

    shift_rows = [
        [
            Paragraph("Configuration", style_map["TableHeader"]),
            Paragraph("A -> B minus A -> A", style_map["TableHeader"]),
            Paragraph("B -> A minus B -> B", style_map["TableHeader"]),
        ]
    ]
    shift_labels = (
        "MLP",
        "CNN",
        "ViT, patch 4",
        "ViT, patch 8",
        "ViT, patch 16",
        "ViT, p16 Linear",
    )
    for config_id, label in zip(all_config_ids, shift_labels):
        a_delta = (
            data[config_id]["accuracy"][("A", "B")]
            - data[config_id]["accuracy"][("A", "A")]
        )
        b_delta = (
            data[config_id]["accuracy"][("B", "A")]
            - data[config_id]["accuracy"][("B", "B")]
        )
        shift_rows.append(
            [
                Paragraph(label, style_map["TableCell"]),
                Paragraph(f"{a_delta:+.2f} pp", style_map["TableCellCenter"]),
                Paragraph(f"{b_delta:+.2f} pp", style_map["TableCellCenter"]),
            ]
        )
    shift_table = ruled_table(
        shift_rows,
        [68 * mm, 48 * mm, 48 * mm],
        compact=True,
    )

    conclusions = Table(
        [
            [
                Paragraph("01", style_map["FindingValue"]),
                Paragraph(
                    "<b>Transfer is strongly asymmetric.</b> For A-trained models, the "
                    f"absolute change at center is at most "
                    f"{all_random_to_center_max_change:.2f} percentage points. B-trained "
                    f"models drop by {hard_drop_min:.2f} to {hard_drop_max:.2f} points at "
                    "random positions. CNN leads all four settings.",
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
                    "<b>Equivalent patch projections stay close in this run.</b> Conv2d and "
                    f"Flatten + Linear differ by at most {embedding_gap_max:.2f} "
                    "percentage points.",
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
                        "- Only translation on a black canvas is tested.<br/>"
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

    abstract = side_note(
        "<i>Abstract.</i> Six configurations were fitted on random or centered object "
        f"positions. Across all six, the absolute A -> B change is at most "
        f"{all_random_to_center_max_change:.2f} percentage points, whereas B -> A drops "
        f"by {hard_drop_min:.2f} to {hard_drop_max:.2f} points. CNN leads all settings "
        f"({cnn_hard_accuracy:.2f}% on B -> A). Patch size 8 leads three ViT settings; "
        f"the two patch projections differ by at most {embedding_gap_max:.2f} points.",
        style_map,
    )

    story = [
        Paragraph("COURSE PROJECT / CONTROLLED COMPARISON", style_map["Kicker"]),
        Paragraph("Position Generalization in<br/>Translated FashionMNIST", style_map["Title"]),
        Paragraph(
            "Architecture, patch scale, and patch projection",
            style_map["Subtitle"],
        ),
        Table(
            [
                [
                    Paragraph("Experimental report", style_map["Meta"]),
                    Paragraph(
                        f"{configuration_count} configurations / {fit_count} fits / "
                        f"{evaluation_count} evaluations",
                        style_map["Meta"],
                    ),
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
        section_header("1", "Study question", style_map),
        Paragraph(
            "A classifier can attach class evidence to location. To isolate this effect, "
            "each 28 x 28 FashionMNIST image in A is assigned one deterministic position "
            "sampled from the 1,369 valid integer locations; B always uses the centered "
            "position. Placements are fixed per sample and split for a given seed. The "
            "experiment asks which model choices retain accuracy when the test position "
            "distribution changes.",
            style_map["Body"],
        ),
        Image(str(SAMPLE_PATH), width=164 * mm, height=48.6 * mm),
        Paragraph(
            "Figure 1. Two official FashionMNIST items under A and B. The teal outline "
            "marks the 28 x 28 source support; only its canvas position changes.",
            style_map["Caption"],
        ),
        finding_strip(
            [
                ("POSITION SUPPORT", "1,369 vs 1", "A locations / B location"),
                (
                    "TRAINED MODELS",
                    str(fit_count),
                    f"{configuration_count} configurations x A/B",
                ),
                (
                    "FINAL TESTS",
                    str(evaluation_count),
                    "four settings per configuration",
                ),
            ],
            style_map,
        ),
        section_header("1.1", "Questions tested", style_map),
        finding_strip(
            [
                ("COMPARISON 1", "Architecture", "Which model handles B -> A?"),
                ("COMPARISON 2", "Patch scale", "How does token count affect ViT?"),
                ("COMPARISON 3", "Projection", "Do equivalent maps train alike?"),
            ],
            style_map,
        ),
        PageBreak(),
        *page_heading(
            "EXPERIMENTAL DESIGN",
            "Controlled Protocol",
            "One split, one training budget, and four evaluation settings",
            style_map,
        ),
        section_header("2", "Evaluation matrix", style_map),
        protocol_table,
        Spacer(1, 2 * mm),
        Paragraph(
            "Table 1. The official test partition is used only after checkpoint selection.",
            style_map["Caption"],
        ),
        section_header("3", "Controlled comparisons", style_map),
        comparison_table,
        Spacer(1, 2 * mm),
        Paragraph(
            "Table 2. Each group changes one stated design choice; the architecture group "
            "is not parameter-matched.",
            style_map["Caption"],
        ),
        section_header("4", "Run sequence", style_map),
        make_protocol_flow(),
        Paragraph(
            "Figure 2. Every fit uses a validation checkpoint, then both held-out test "
            "distributions are evaluated with shared deterministic placements.",
            style_map["Caption"],
        ),
        side_note(
            "<b>Direction matters.</b> B -> A extrapolates from one observed location "
            "to 1,369 possible locations. A -> B evaluates a location already contained "
            "in A's position support.",
            style_map,
        ),
        PageBreak(),
        *page_heading(
            "MODEL AND TRAINING DETAILS",
            "Implementations",
            "Model size, positional representation, and optimization settings",
            style_map,
        ),
        section_header("5", "Model definitions", style_map),
        model_table,
        Spacer(1, 2 * mm),
        Paragraph(
            "Table 3. Trainable parameter counts are taken from the fitted models.",
            style_map["Caption"],
        ),
        section_header("6", "Vision Transformer", style_map),
        make_vit_pipeline(),
        Paragraph(
            "Figure 3. ViT implementation shared by the patch-scale and projection studies.",
            style_map["Caption"],
        ),
        side_note(
            "<b>Position representation.</b> The ViT adds one learned absolute embedding "
            "to each patch index. Translation equivariance is therefore not built into "
            "the token sequence.",
            style_map,
        ),
        section_header("7", "Training recipe", style_map),
        controls_table,
        Spacer(1, 2 * mm),
        Paragraph(
            f"Table 4. Formal environment: PyTorch {environment['torch']}; "
            f"CUDA {environment['cuda']}; {environment['gpu']}; "
            f"{environment['platform']}. Mixed precision "
            f"{'enabled' if protocol['amp'] else 'disabled'}.",
            style_map["Caption"],
        ),
        PageBreak(),
        *page_heading(
            "EXPERIMENTAL PROCESS",
            "Learning Curves and Checkpoints",
            "Validation behavior for MLP, CNN, and patch-8 ViT over 15 epochs",
            style_map,
        ),
        section_header("8", "Training dynamics", style_map),
        chart_block(
            make_training_curve_chart(histories),
            "Figure 4. Validation accuracy for A-trained and B-trained architecture models.",
            style_map,
        ),
        best_validation_table,
        Spacer(1, 2.5 * mm),
        Paragraph(
            "Table 5. Best validation accuracy and selected checkpoint epoch.",
            style_map["Caption"],
        ),
        side_note(
            f"<b>Checkpoint selection.</b> The six checkpoints occur at epoch "
            f"{min(best_validation_epochs)} or {max(best_validation_epochs)}. CNN gives "
            "the highest validation accuracy for both training distributions; each model "
            "validates higher on centered images.",
            style_map,
        ),
        Spacer(1, 4 * mm),
        training_findings,
        PageBreak(),
        *page_heading(
            "ARCHITECTURE COMPARISON",
            "MLP, CNN, and ViT",
            "Shared data split and optimizer; model capacity is not matched",
            style_map,
        ),
        section_header("9", "Test accuracy", style_map),
        architecture_table,
        Spacer(1, 2 * mm),
        Paragraph(
            "Table 6. Top-1 test accuracy (%); teal values mark column leaders.",
            style_map["Caption"],
        ),
        chart_block(
            make_hard_transfer_chart(data),
            f"Figure 5. CNN reaches {cnn_hard_accuracy:.2f}% on B -> A, "
            f"{cnn_advantage:.2f} points above patch-8 ViT.",
            style_map,
        ),
        findings,
        section_header("9.1", "Interpretation", style_map),
        Paragraph(
            "CNN leads all four settings despite having fewer parameters than the MLP "
            "and ViT. Shared local filters are a plausible reason for the stronger shift "
            "result, but this run does not separate architecture from parameter count or "
            "model-specific hyperparameter tuning.",
            style_map["Body"],
        ),
        PageBreak(),
        *page_heading(
            "POSITION-SHIFT ANALYSIS",
            "The Transfer Asymmetry",
            "Accuracy changes when the training and test position supports differ",
            style_map,
        ),
        section_header("10", "Directional shift", style_map),
        chart_block(
            make_shift_chart(data),
            f"Figure 6. Centered-only training drops by {hard_drop_min:.2f} to "
            f"{hard_drop_max:.2f} points under random-position testing.",
            style_map,
        ),
        shift_table,
        Spacer(1, 2 * mm),
        Paragraph(
            "Table 7. Signed test-distribution change for all six configurations.",
            style_map["Caption"],
        ),
        side_note(
            f"<b>Consistent pattern.</b> After training on A, moving the test objects to "
            f"the center produces an absolute change of at most "
            f"{all_random_to_center_max_change:.2f} percentage points. After training "
            "on B, random placement removes at least "
            f"{hard_drop_min:.2f} points.",
            style_map,
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "The comparison supports a coverage explanation: A exposes the model to the "
            "center and many off-center positions, whereas B provides no direct evidence "
            "about most of A. The result does not imply that A is universally harder; it "
            "shows that high B -> B accuracy is not evidence of translation robustness.",
            style_map["Body"],
        ),
        PageBreak(),
        *page_heading(
            "VIT ABLATION I",
            "Patch Scale",
            "ViT body fixed at d=128, depth 4, four heads, and FFN width 512",
            style_map,
        ),
        section_header("11", "Patch-size results", style_map),
        chart_block(
            make_patch_chart(data),
            "Figure 7. Selected random-position outcomes; Table 8 reports all four settings.",
            style_map,
        ),
        patch_table,
        Spacer(1, 2 * mm),
        Paragraph(
            "Table 8. Accuracy (%) and summed training time; teal marks the best value "
            "in each column.",
            style_map["Caption"],
        ),
        side_note(
            "<b>Observed trade-off.</b> Patch size 8 leads on A -> A, A -> B, and "
            "B -> A. Patch size 4 expands the sequence to 256 tokens and takes longer "
            "without improving accuracy under the shared 15-epoch budget.",
            style_map,
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "Patch size 16 is fastest and gives the best B -> B result. Under this "
            "protocol, patch size 8 records the highest A -> A, A -> B, and B -> A "
            "accuracy with 64 tokens. Runtime is descriptive because it depends on "
            "hardware and execution order.",
            style_map["Body"],
        ),
        Spacer(1, 2 * mm),
        finding_strip(
            [
                ("PATCH 4", "65,536", "patch-patch pairs (N^2)"),
                ("PATCH 8", "4,096", "patch-patch pairs (N^2)"),
                ("PATCH 16", "256", "patch-patch pairs (N^2)"),
            ],
            style_map,
        ),
        PageBreak(),
        *page_heading(
            "VIT ABLATION II",
            "Patch Projection",
            "Conv2d and Flatten + Linear at patch size 16",
            style_map,
        ),
        section_header("12", "Equivalent projections", style_map),
        make_embedding_equivalence(),
        Paragraph(
            "Figure 8. With non-overlapping patches, the two implementations differ in "
            "layout, not in the affine function they can represent.",
            style_map["Caption"],
        ),
        embedding_table,
        Spacer(1, 2 * mm),
        Paragraph(
            f"Table 9. Test accuracy (%) at p=16; maximum absolute gap "
            f"{embedding_gap_max:.2f} points.",
            style_map["Caption"],
        ),
        side_note(
            "<b>Result.</b> The trained projections remain numerically close. Because the "
            "record contains one seed, the residual gap is not treated as evidence that "
            "either implementation is better.",
            style_map,
        ),
        section_header("13", "Conclusions and limits", style_map),
        conclusions,
        Spacer(1, 2.5 * mm),
        limitations,
    ]
    return story


def build() -> Path:
    register_fonts()
    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError(f"Missing report sample figure: {SAMPLE_PATH}")
    data = load_metrics()
    histories = load_training_history()
    manifest = load_manifest()
    style_map = build_styles()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ReportDocument(str(OUTPUT_PATH)).build(
        build_story(data, histories, manifest, style_map)
    )
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build())
