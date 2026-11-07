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

REPORT_FONT: str = "微软雅黑"


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
    headers = ["序号", "树种", "等效冠幅(m)"] + (["树高(m)"] if with_height else []) + [f"价格({currency})"]
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
    put(label, f"合计（共 {len(detail)} 株，其中 {priced} 株计入价格）", bold=True)
    put(cells[-1], f"{total_value:,.2f}", bold=True, right=True)


class ReportArgs(RegionArgs):
    """Arguments of :class:`ReportTool`."""

    title: Optional[str] = Field(None, description="报告标题；省略则自动生成")


class ReportTool(BaseTool):
    """Generate a Word document describing the trees within a range."""

    name = "generate_report"
    description = "输入统计范围，以Word文档文字形式输出该范围内苗木的统计结果（数量、构成、冠幅、资产价值）。需先提取。"
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
            raise ToolError("指定范围内没有苗木，无法生成报告。")
        summary = species_summary(trees, pricing)
        region = region_label(args, "地块")
        title = args.title or f"{region}苗木统计报告"
        total = int(summary["count"].sum())
        total_value = float(summary["value"].sum(skipna=True))
        total_area = float(summary["total_crown_area_m2"].sum())
        priced_total = int(summary["priced_count"].sum())
        detail = tree_detail(trees, list(summary["species"]))
        cur = pricing.currency
        scope = f"{region}" + (f"（范围 bbox={args.bbox}）" if args.bbox else "") + (
            f"（按矢量 {args.region_vector_path} 裁剪）" if args.region_vector_path else ""
        )

        doc = Document()
        apply_report_font(doc)
        heading = doc.add_heading(title, level=0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(f"报告日期：{datetime.now():%Y年%m月%d日}　数据来源：{image.name}")

        doc.add_heading("一、统计概况", level=1)
        doc.add_paragraph(
            f"统计范围为{scope}。经遥感影像智能提取，范围内共识别苗木 {total} 株，"
            f"涉及 {len(summary)} 个树种，冠幅投影总面积约 {total_area:,.1f} 平方米。"
        )

        doc.add_heading("二、树种构成", level=1)
        for r in summary.itertuples():
            doc.add_paragraph(
                f"{r.species}：{int(r.count)} 株，占比 {r.share_pct:.1f}%，"
                f"平均等效冠幅约 {r.mean_crown_d_m:.2f} 米，冠幅总面积约 {r.total_crown_area_m2:,.1f} 平方米。",
                style="List Bullet",
            )

        doc.add_heading("三、资产价值", level=1)
        if summary["has_formula"].any():
            doc.add_paragraph(
                f"估值方式：{MODE_TEXT[mode]}。{region}内 {priced_total} 株苗木计入价格，"
                f"总价格 {total_value:,.2f} {cur}。"
            )
        else:
            doc.add_paragraph(f"{region}内的树种均{NO_FORMULA}，未做资产评估。")
        for r in summary.itertuples():
            if not r.has_formula:
                text = f"{r.species}：{NO_FORMULA}，未计入总价值。"
            elif r.priced_count == 0:
                text = f"{r.species}：{NO_HEIGHT}，未计入总价值。"
            else:
                text = (
                    f"{r.species}：平均单价约 {r.avg_price:,.2f} {cur}/株，价值 {r.value:,.2f} {cur}"
                    f"（{pricing.formula_text(r.species, mode == 'dem')}）。"
                )
            doc.add_paragraph(text, style="List Bullet")

        doc.add_heading("四、单棵树明细", level=1)
        doc.add_paragraph(
            "下表逐棵列出等效冠幅" + ("、树高" if mode == "dem" else "") + f"和价格，并给出总价格 {total_value:,.2f} {cur}。"
        )
        _add_tree_table(doc, detail, mode == "dem", cur, total_value, priced_total)

        doc.add_heading("五、说明", level=1)
        doc.add_paragraph("价格公式中 W 为等效冠幅（厘米）；H 为树高（厘米），取树冠范围内 DEM−DTM 的最高值，其中 DTM 由 DEM 经渐进形态学滤波（PMF）得到。")
        doc.add_paragraph("本报告数据由模型自动提取与计算得出，仅供参考，重要决策前请结合现场核查。")

        out_dir = ctx.services.settings.output_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{re.sub(r'[\\/:*?\"<>|]', '_', title)}_{stamp}.docx"
        doc.save(out_path)
        logger.info("Report written: %s", out_path)
        value_part = f"，总价格 {total_value:,.2f} {cur}" if summary["has_formula"].any() else ""
        return ToolResult(
            message=f"已生成报告「{title}」（{total} 株{value_part}，含逐棵明细），可在下方下载。",
            files=[str(out_path)],
            data={"total": total, "total_value": total_value},
        )
