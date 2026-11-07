"""Preset plans: named, parameterised tool sequences.

The planner may answer with ``{"preset": name, "params": {...}}``; the preset
expands to concrete steps without further LLM work. Add your own with
:meth:`PresetRegistry.register`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PlanStep(BaseModel):
    """One tool call.

    Attributes:
        tool: Tool name.
        args: Tool arguments.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class PresetPlan:
    """A preset plan definition.

    Attributes:
        name: Unique name used by the planner.
        description: When to use it (shown to the LLM).
        params: Parameter name -> description.
        builder: Function turning params into steps.
    """

    name: str
    description: str
    params: dict[str, str]
    builder: Callable[[dict[str, Any]], list[PlanStep]]


class PresetRegistry:
    """Registry of preset plans."""

    def __init__(self) -> None:
        """Create a registry pre-filled with the built-in presets."""
        self._items: dict[str, PresetPlan] = {}
        for preset in _builtin_presets():
            self.register(preset)

    def register(self, preset: PresetPlan) -> None:
        """Register (or replace) a preset.

        Args:
            preset: Preset definition.
        """
        self._items[preset.name] = preset

    def __contains__(self, name: object) -> bool:
        """Whether a preset exists."""
        return name in self._items

    def expand(self, name: str, params: dict[str, Any]) -> list[PlanStep]:
        """Expand a preset into steps.

        Args:
            name: Preset name.
            params: Parameters supplied by the planner.

        Returns:
            Concrete steps.
        """
        return self._items[name].builder(params)

    def catalog(self) -> str:
        """Render the preset list for the planner prompt."""
        lines = []
        for p in self._items.values():
            params = "、".join(f"{k}({v})" for k, v in p.params.items())
            lines.append(f"* {p.name}: {p.description} 参数: {params}")
        return "\n".join(lines)


def _pick(params: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Copy the non-empty entries of ``keys`` from ``params``."""
    return {k: params[k] for k in keys if params.get(k) not in (None, "")}


_REGION_KEYS = ("image_id", "region_name", "bbox", "region_vector_path")


def _extract_count(p: dict[str, Any]) -> list[PlanStep]:
    return [
        PlanStep(tool="extract_trees", args=_pick(p, "type_name", "image_id")),
        PlanStep(tool="count_trees", args=_pick(p, *_REGION_KEYS)),
    ]


def _value_pipeline(p: dict[str, Any]) -> list[PlanStep]:
    value_args = _pick(p, *_REGION_KEYS)
    if p.get("type_name"):
        value_args["species"] = p["type_name"]
    return _extract_count(p) + [PlanStep(tool="evaluate_value", args=value_args)]


def _full_census(p: dict[str, Any]) -> list[PlanStep]:
    region = _pick(p, *_REGION_KEYS)
    return _value_pipeline(p) + [
        PlanStep(tool="generate_table", args={**region, "template": "plot_census"}),
        PlanStep(tool="generate_table", args={**region, "template": "asset_ledger"}),
        PlanStep(tool="generate_report", args=region),
    ]


def _builtin_presets() -> list[PresetPlan]:
    common = {"type_name": "苗木类型名", "image_id": "可选", "region_name": "可选", "bbox": "可选"}
    return [
        PresetPlan("extract_and_count", "提取某类苗木并统计数量", common, _extract_count),
        PresetPlan("extract_and_value", "提取某类苗木、统计数量并做资产评估", common, _value_pipeline),
        PresetPlan("full_census", "一键完成提取、统计、资产评估、两张报表和Word报告", common, _full_census),
    ]


def default_presets() -> PresetRegistry:
    """Build the default preset registry.

    Returns:
        Registry with built-in presets.
    """
    return PresetRegistry()

