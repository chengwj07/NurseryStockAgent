"""Tool 2: count extracted trees."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from tools._common import collect_trees, region_label
from tools.base import BaseTool, RegionArgs, ToolContext, ToolResult


class CountArgs(RegionArgs):
    """Arguments of :class:`CountTool`."""

    species: Optional[str] = Field(None, description="Count only this species; omit to count all species")


class CountTool(BaseTool):
    """Count trees per species."""

    name = "count_trees"
    description = (
        "Count the extracted trees (total and per species; each connected component counts as one tree). "
        "Can filter by species and range. Requires prior extraction."
    )
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
            return ToolResult(message=f"No matching trees in {label} (0 trees).", data={"total": 0, "by_species": {}})
        counts = trees["species"].value_counts()
        lines = "\n".join(f"- {name}: {num} trees" for name, num in counts.items())
        return ToolResult(
            message=f"Counted **{len(trees)}** trees in {label}:\n{lines}",
            data={"total": int(len(trees)), "by_species": {k: int(v) for k, v in counts.items()}},
        )
