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
    "dem": "原图 + DEM（价格由等效冠幅 W 与树高 H 计算，单位厘米；H 为冠幅内 DEM−DTM 的最高值）",
    "single": "仅原图（价格由等效冠幅 W 计算，单位厘米）",
}


class ValueArgs(RegionArgs):
    """Arguments of :class:`ValuationTool`."""

    species: Optional[str] = Field(None, description="只评估该树种；省略则评估已提取的全部树种")


class ValuationTool(BaseTool):
    """Compute the price of every tree and the asset value of a region."""

    name = "evaluate_value"
    description = (
        "资产评估：按树种公式逐棵计算价格并汇总（有DEM时用冠幅W和树高H，否则只用冠幅W）。"
        "仅支持罗汉松、红花继木、茶梅、亮晶女贞，其他树种返回'无评估公式'。需先提取。"
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
                    message=f"**{name}**{NO_FORMULA}（目前仅支持：{'、'.join(pricing.supported_species())}）。",
                    data={"supported": False},
                )

        mode = ensure_priced(ctx, image)
        _, trees = collect_trees(ctx, args, args.species)
        label = region_label(args)
        if trees.empty:
            return ToolResult(message=f"{label}内没有符合条件的苗木，无法评估。")
        summary = species_summary(trees, pricing)
        if not summary["has_formula"].any():
            names = "、".join(summary["species"])
            return ToolResult(
                message=f"已提取的树种（{names}）{NO_FORMULA}（目前仅支持：{'、'.join(pricing.supported_species())}）。",
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
        table = f"| 树种 | 数量(株) | 平均单价({cur}/株) | 价值({cur}) |\n|---|---|---|---|\n" + "\n".join(rows)

        notes = ""
        unvalued = int(((summary["has_formula"]) & (summary["count"] > summary["priced_count"])).sum())
        if unvalued:
            notes += f"\n\n> 部分树木无高度数据（不在DEM范围内），未计入价值。"
        if (~summary["has_formula"]).any():
            notes += f"\n\n> {'、'.join(summary.loc[~summary['has_formula'], 'species'])}{NO_FORMULA}，未计入合计。"
        return ToolResult(
            message=(
                f"{label}苗木资产评估（{MODE_TEXT[mode]}）：\n\n{table}\n\n**合计：{total:,.2f} {cur}**{notes}\n\n"
                "鼠标指向地图上的树可查看单棵价格。"
            ),
            data={"total_value": total, "mode": mode},
        )
