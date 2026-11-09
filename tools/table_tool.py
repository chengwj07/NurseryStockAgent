"""Tool 5: fill a built-in report template and export an Excel table."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import Field

from services.analytics import species_summary, tree_detail
from services.valuation import NO_FORMULA, NO_HEIGHT
from tools._common import collect_trees, ensure_priced, region_label, resolve_image
from tools.base import BaseTool, RegionArgs, ToolContext, ToolError, ToolResult
from tools.valuation_tool import MODE_TEXT

logger = logging.getLogger(__name__)

_THIN = Side(style="thin")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


class TableArgs(RegionArgs):
    """Arguments of :class:`TableTool`."""

    template: str = Field(
        ...,
        description="Report template: asset_ledger (area nursery asset ledger: lists crown width / height / price of every tree and the total) or plot_census (plot census report)",
    )


def _is_number(value: Any) -> bool:
    """True for finite int/float values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value))


class TableTool(BaseTool):
    """Render an Excel report from a JSON template (``templates/table_templates.json``)."""

    name = "generate_table"
    description = (
        "Export an Excel report from a built-in template: asset_ledger = area nursery asset ledger (lists the crown width, "
        "tree height (with a DEM) and price of every tree and computes the total price), plot_census = plot census report. "
        "Requires prior extraction."
    )
    args_schema = TableArgs

    def run(self, ctx: ToolContext, args: TableArgs) -> ToolResult:
        """Generate the workbook.

        Args:
            ctx: Tool context.
            args: Validated arguments.

        Returns:
            Result containing the generated ``.xlsx`` path.

        Raises:
            ToolError: If the template name is unknown or no tree is in range.
        """
        path = ctx.services.settings.templates_dir / "table_templates.json"
        templates: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        tpl = templates.get(args.template)
        if tpl is None:
            raise ToolError(f"Unknown report template {args.template!r}; available: {', '.join(templates)}")

        mode = "single"
        if tpl.get("needs_price"):
            mode = ensure_priced(ctx, resolve_image(ctx, args.image_id))
        image, trees = collect_trees(ctx, args)
        if trees.empty:
            raise ToolError("No trees in the selected range; cannot generate the table.")
        summary = species_summary(trees, ctx.services.pricing)
        summary.insert(0, "index", range(1, len(summary) + 1))
        species_rows = summary.to_dict("records")
        for rec in species_rows:  # species that cannot be valued show a reason instead of a number
            for field in ("avg_price", "value"):
                if isinstance(rec.get(field), float) and math.isnan(rec[field]):
                    rec[field] = NO_FORMULA if not rec["has_formula"] else NO_HEIGHT
        tree_rows = tree_detail(trees, list(summary["species"])).to_dict("records")
        for rec in tree_rows:
            if math.isnan(rec["price"]):
                rec["price"] = rec["price_text"] or NO_FORMULA
            if math.isnan(rec["height_m"]):
                rec["height_m"] = "—"
        sources = {"species": species_rows, "trees": tree_rows}

        region = region_label(args, tpl.get("default_region_name", "Area"))
        fmt = {
            "region_name": region,
            "region_display": (args.region_name or "").strip() or "Not specified",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "image_name": image.name,
            "mode_text": MODE_TEXT[mode],
        }
        title = tpl["title"].format_map(fmt)

        out_dir = ctx.services.settings.output_dir / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{re.sub(r'[\\/:*?\"<>|]', '_', title)}_{stamp}.xlsx"
        self._write_workbook(out_path, tpl, title, fmt, sources, mode)
        logger.info("Table written: %s", out_path)

        total_text = ""
        if tpl.get("needs_price"):
            total = sum(float(r["price"]) for r in tree_rows if _is_number(r["price"]))
            total_text = f", total price {total:,.2f} {ctx.services.pricing.currency}"
        return ToolResult(
            message=f"Generated “{title}” ({len(tree_rows)} trees, {len(species_rows)} species{total_text}); download it below.",
            files=[str(out_path)],
            data={"template": args.template, "rows": len(tree_rows)},
        )

    @staticmethod
    def _write_workbook(
        out_path: Path,
        tpl: dict[str, Any],
        title: str,
        fmt: dict[str, str],
        sources: dict[str, list[dict[str, Any]]],
        mode: str,
    ) -> None:
        """Write the template-driven workbook (one worksheet per ``sheets`` entry).

        Args:
            out_path: Target file.
            tpl: Template definition.
            title: Resolved report title.
            fmt: Placeholder values.
            sources: Rows per source name (``"species"`` / ``"trees"``).
            mode: ``"single"`` or ``"dem"``; columns with ``requires: dem`` are
                dropped in single-image mode.
        """
        wb = Workbook()
        wb.remove(wb.active)
        for sheet in tpl["sheets"]:
            cols = [c for c in sheet["columns"] if c.get("requires", mode) == mode]
            rows = sources[sheet["source"]]
            ws = wb.create_sheet(sheet["name"])
            TableTool._write_sheet(ws, tpl, title, fmt, cols, rows, sheet.get("total"))
        wb.save(out_path)

    @staticmethod
    def _write_sheet(
        ws: Worksheet,
        tpl: dict[str, Any],
        title: str,
        fmt: dict[str, str],
        cols: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        total: dict[str, Any] | None,
    ) -> None:
        """Fill one worksheet: title, meta block, header, rows, total row."""
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(cols))
        ws.cell(1, 1, title).font = Font(size=16, bold=True)
        ws.cell(1, 1).alignment = Alignment(horizontal="center")

        row_no = 2
        for key, value in tpl.get("meta", []):
            ws.cell(row_no, 1, key).font = Font(bold=True)
            ws.cell(row_no, 2, value.format_map(fmt))
            row_no += 1

        header_row = row_no + 1
        for c, col in enumerate(cols, 1):
            cell = ws.cell(header_row, c, col["header"])
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="DDEBF7")
            cell.alignment = Alignment(horizontal="center")
            cell.border = _BORDER
            ws.column_dimensions[get_column_letter(c)].width = col.get("width", 14)

        for r, record in enumerate(rows, header_row + 1):
            for c, col in enumerate(cols, 1):
                cell = ws.cell(r, c, record.get(col["field"]))
                cell.border = _BORDER
                if "format" in col and _is_number(record.get(col["field"])):
                    cell.number_format = col["format"]
        ws.freeze_panes = ws.cell(header_row + 1, 1)

        if total:
            r = header_row + len(rows) + 1
            ws.cell(r, 1, total["label"].format(n=len(rows))).font = Font(bold=True)
            for c, col in enumerate(cols, 1):
                cell = ws.cell(r, c)
                cell.border = _BORDER
                if col["field"] in total["sum_fields"]:
                    cell.value = sum(float(x[col["field"]]) for x in rows if _is_number(x[col["field"]]))
                    cell.font = Font(bold=True)
                    if "format" in col:
                        cell.number_format = col["format"]
