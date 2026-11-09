"""Tool 3: value the extracted trees with the per-species price formulas."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from agent.state import ValuationResult
from services.analytics import species_summary
from services.valuation import NO_FORMULA, NO_HEIGHT
from tools._common import collect_trees, ensure_priced, region_label, resolve_image
from tools.base import BaseTool, RegionArgs, ToolContext, ToolResult

MODE_TEXT: dict[str, str] = {
    "dem": "original image + DEM (price from equivalent crown width W and tree height H, in cm; H is the highest DEM−DTM value inside the crown)",
    "single": "original image only (price from equivalent crown width W, in cm)",
}


class ValueArgs(RegionArgs):
    """Arguments of :class:`ValuationTool`."""

    species: Optional[str] = Field(None, description="Value only this species; omit to value all extracted species")


class ValuationTool(BaseTool):
    """Compute the price of every tree and the asset value of a region."""

    name = "evaluate_value"
    description = (
        "Asset valuation: compute the price of every tree with the species formula and summarise "
        "(uses crown width W and tree height H when a DEM is attached, otherwise W only). "
        "Only Buddhist Pine, Red Loropetalum, Sasanqua Camellia and Glossy Privet are supported; "
        "other species return 'No valuation formula'. Requires prior extraction."
    )
    args_schema = ValueArgs

    def run(self, ctx: ToolContext, args: ValueArgs) -> ToolResult:
        """Compute per-tree prices, store them and summarise.

        Args:
            ctx: Tool context.
            args: Validated arguments.

        Returns:
            Result with a Markdown table and the total value.
        """
        pricing = ctx.services.pricing
        image = resolve_image(ctx, args.image_id)
        if args.species:
            name = pricing.resolve(args.species) or args.species.strip()
            if not pricing.has_formula(name):
                return ToolResult(
                    message=f"**{name}**: {NO_FORMULA} (currently supported: {', '.join(pricing.supported_species())}).",
                    data={"supported": False},
                )

        mode = ensure_priced(ctx, image)
        _, trees = collect_trees(ctx, args, args.species)
        label = region_label(args)
        if trees.empty:
            return ToolResult(message=f"No matching trees in {label}; nothing to value.")
        summary = species_summary(trees, pricing)
        if not summary["has_formula"].any():
            names = ", ".join(summary["species"])
            return ToolResult(
                message=f"The extracted species ({names}): {NO_FORMULA} (currently supported: {', '.join(pricing.supported_species())}).",
                data={"supported": False},
            )

        total = float(summary["value"].sum(skipna=True))
        ctx.session.last_valuation = ValuationResult(label, summary, total)
        cur = pricing.currency
        rows: list[str] = []
        for r in summary.itertuples():
            if not r.has_formula:
                avg = val = NO_FORMULA
            elif r.priced_count == 0:
                avg = val = NO_HEIGHT
            else:
                avg, val = f"{r.avg_price:,.2f}", f"{r.value:,.2f}"
            rows.append(f"| {r.species} | {int(r.count)} | {avg} | {val} |")
        table = f"| Species | Count | Avg. unit price ({cur}/tree) | Value ({cur}) |\n|---|---|---|---|\n" + "\n".join(rows)

        notes = ""
        unvalued = int(((summary["has_formula"]) & (summary["count"] > summary["priced_count"])).sum())
        if unvalued:
            notes += "\n\n> Some trees have no height data (outside the DEM) and are not included in the value."
        if (~summary["has_formula"]).any():
            notes += f"\n\n> {', '.join(summary.loc[~summary['has_formula'], 'species'])}: {NO_FORMULA}; not included in the total."
        return ToolResult(
            message=(
                f"Nursery asset valuation of {label} ({MODE_TEXT[mode]}):\n\n{table}\n\n**Total: {total:,.2f} {cur}**{notes}\n\n"
                "Hover over a tree on the map to see its price."
            ),
            data={"total_value": total, "mode": mode},
        )
