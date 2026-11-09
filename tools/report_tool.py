"""Tool 6: write a Word report (textual statistics of a region)."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from docx import Document
from docx.document import Document as DocumentType
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from pydantic import Field

from services.analytics import species_summary, tree_detail
from services.valuation import NO_FORMULA, NO_HEIGHT
from tools._common import collect_trees, ensure_priced, region_label, resolve_image
from tools.base import BaseTool, RegionArgs, ToolContext, ToolError, ToolResult
from tools.valuation_tool import MODE_TEXT

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

REPORT_FONT: str = "Microsoft YaHei"


def apply_report_font(doc: DocumentType, font: str = REPORT_FONT) -> None:
    """Set one font (Latin and East-Asian) on every paragraph/character style.

    Theme-font attributes are removed because they override explicit names in
    Word's default template (headings and the title use them).

    Args:
        doc: Document to modify.
        font: Font family name.
    """
    for style in doc.styles:
        if style.type not in (WD_STYLE_TYPE.PARAGRAPH, WD_STYLE_TYPE.CHARACTER, WD_STYLE_TYPE.TABLE):
            continue
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.rFonts
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.insert(0, rfonts)
        for theme_attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            rfonts.attrib.pop(qn(theme_attr), None)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rfonts.set(qn(attr), font)


def _add_tree_table(doc: DocumentType, detail: "pd.DataFrame", with_height: bool, currency: str,
                    total_value: float, priced: int) -> None:
    """Append the per-tree table (crown width, height if any, price) with a total row.

    Args:
        doc: Target document.
        detail: Output of :func:`services.analytics.tree_detail`.
        with_height: Whether to include the tree-height column (DEM mode).
        currency: Currency label.
        total_value: Sum of all tree prices.
        priced: Number of trees that were valued.
    """
    headers = ["No.", "Species", "Equivalent crown width (m)"] + (["Tree height (m)"] if with_height else []) + [f"Price ({currency})"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"

    def put(cell, text: str, bold: bool = False, right: bool = False) -> None:
        paragraph = cell.paragraphs[0]
        run = paragraph.add_run(text)
        run.font.size = Pt(10)
        run.bold = bold
        if right:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    for cell, header in zip(table.rows[0].cells, headers):
        put(cell, header, bold=True)
    repeat = OxmlElement("w:tblHeader")  # repeat the header row on every page
    repeat.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)

    for rec in detail.itertuples():
        cells = table.add_row().cells
        values = [str(rec.index), rec.species, f"{rec.crown_d_m:.2f}"]
        if with_height:
            values.append("—" if rec.height_m != rec.height_m else f"{rec.height_m:.2f}")
        values.append(f"{rec.price:,.2f}" if rec.price == rec.price else (rec.price_text or NO_FORMULA))
        for i, (cell, text) in enumerate(zip(cells, values)):
            put(cell, text, right=i >= 2)

    cells = table.add_row().cells
    label = cells[0].merge(cells[len(headers) - 2])
    put(label, f"Total ({len(detail)} trees, {priced} included in the price)", bold=True)
    put(cells[-1], f"{total_value:,.2f}", bold=True, right=True)


class ReportArgs(RegionArgs):
    """Arguments of :class:`ReportTool`."""

    title: Optional[str] = Field(None, description="Report title; omit to generate one")


class ReportTool(BaseTool):
    """Generate a Word document describing the trees within a range."""

    name = "generate_report"
    description = (
        "Given a statistics range, write a Word document with the tree statistics of that range "
        "(count, composition, crown width, asset value). Requires prior extraction."
    )
    args_schema = ReportArgs

    def run(self, ctx: ToolContext, args: ReportArgs) -> ToolResult:
        """Generate the ``.docx`` file.

        Args:
            ctx: Tool context.
            args: Validated arguments.

        Returns:
            Result containing the generated ``.docx`` path.

        Raises:
            ToolError: If no tree is in range.
        """
        pricing = ctx.services.pricing
        mode = ensure_priced(ctx, resolve_image(ctx, args.image_id))
        image, trees = collect_trees(ctx, args)
        if trees.empty:
            raise ToolError("No trees in the selected range; cannot generate the report.")
        summary = species_summary(trees, pricing)
        region = region_label(args, "Plot")
        title = args.title or f"{region} Nursery Statistics Report"
        total = int(summary["count"].sum())
        total_value = float(summary["value"].sum(skipna=True))
        total_area = float(summary["total_crown_area_m2"].sum())
        priced_total = int(summary["priced_count"].sum())
        detail = tree_detail(trees, list(summary["species"]))
        cur = pricing.currency
        scope = f"{region}" + (f" (range bbox={args.bbox})" if args.bbox else "") + (
            f" (clipped by vector {args.region_vector_path})" if args.region_vector_path else ""
        )

        doc = Document()
        apply_report_font(doc)
        heading = doc.add_heading(title, level=0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(f"Report date: {datetime.now():%B %d, %Y}    Data source: {image.name}")

        doc.add_heading("1. Overview", level=1)
        doc.add_paragraph(
            f"The statistics range is {scope}. Intelligent extraction from the remote-sensing image identified {total} trees "
            f"in the range, covering {len(summary)} species, with a total crown projection area of about {total_area:,.1f} square meters."
        )

        doc.add_heading("2. Species Composition", level=1)
        for r in summary.itertuples():
            doc.add_paragraph(
                f"{r.species}: {int(r.count)} trees, {r.share_pct:.1f}% of the total, "
                f"mean equivalent crown width about {r.mean_crown_d_m:.2f} m, total crown area about {r.total_crown_area_m2:,.1f} m².",
                style="List Bullet",
            )

        doc.add_heading("3. Asset Value", level=1)
        if summary["has_formula"].any():
            doc.add_paragraph(
                f"Valuation method: {MODE_TEXT[mode]}. {priced_total} trees in {region} are included in the price; "
                f"total price {total_value:,.2f} {cur}."
            )
        else:
            doc.add_paragraph(f"All species in {region}: {NO_FORMULA}; no asset valuation was performed.")
        for r in summary.itertuples():
            if not r.has_formula:
                text = f"{r.species}: {NO_FORMULA}; not included in the total value."
            elif r.priced_count == 0:
                text = f"{r.species}: {NO_HEIGHT}; not included in the total value."
            else:
                text = (
                    f"{r.species}: average unit price about {r.avg_price:,.2f} {cur}/tree, value {r.value:,.2f} {cur} "
                    f"({pricing.formula_text(r.species, mode == 'dem')})."
                )
            doc.add_paragraph(text, style="List Bullet")

        doc.add_heading("4. Per-Tree Details", level=1)
        doc.add_paragraph(
            "The table below lists the equivalent crown width" + (", tree height" if mode == "dem" else "") + f" and price of every tree, and gives the total price of {total_value:,.2f} {cur}."
        )
        _add_tree_table(doc, detail, mode == "dem", cur, total_value, priced_total)

        doc.add_heading("5. Notes", level=1)
        doc.add_paragraph(
            "In the price formulas, W is the equivalent crown width (cm) and H is the tree height (cm), taken as the highest "
            "DEM−DTM value inside the tree crown; the DTM is derived from the DEM with the progressive morphological filter (PMF)."
        )
        doc.add_paragraph(
            "The data in this report are extracted and computed automatically by models and are for reference only; "
            "verify on site before making important decisions."
        )

        out_dir = ctx.services.settings.output_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{re.sub(r'[\\/:*?\"<>|]', '_', title)}_{stamp}.docx"
        doc.save(out_path)
        logger.info("Report written: %s", out_path)
        value_part = f", total price {total_value:,.2f} {cur}" if summary["has_formula"].any() else ""
        return ToolResult(
            message=f"Generated report “{title}” ({total} trees{value_part}, with per-tree details); download it below.",
            files=[str(out_path)],
            data={"total": total, "total_value": total_value},
        )
