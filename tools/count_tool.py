"""Tool 2: count extracted trees."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from tools._common import collect_trees, region_label
from tools.base import BaseTool, RegionArgs, ToolContext, ToolResult


class CountArgs(RegionArgs):
    """Arguments of :class:`CountTool`."""

    species: Optional[str] = Field(None, description="只统计该树种；省略则统计全部树种")


class CountTool(BaseTool):
    """Count trees per species."""

    name = "count_trees"
    description = "统计已提取苗木的数量（总数和分树种数量；每个连通域计一株），可按树种和范围过滤。需先提取。"
    args_schema = CountArgs

    def run(self, ctx: ToolContext, args: CountArgs) -> ToolResult:
        """Count trees.

        Args:
            ctx: Tool context.
            args: Validated arguments.

        Returns:
            Result with total and per-species counts.
        """
        _, trees = collect_trees(ctx, args, args.species)
        label = region_label(args)
        if trees.empty:
            return ToolResult(message=f"{label}内没有符合条件的苗木（0 株）。", data={"total": 0, "by_species": {}})
        counts = trees["species"].value_counts()
        lines = "\n".join(f"- {name}：{num} 株" for name, num in counts.items())
        return ToolResult(
            message=f"{label}共统计到 **{len(trees)}** 株苗木：\n{lines}",
            data={"total": int(len(trees)), "by_species": {k: int(v) for k, v in counts.items()}},
        )
