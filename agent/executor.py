"""Executor: runs plan steps by invoking LangChain tools. No LLM involved."""

from __future__ import annotations

import logging
from typing import Iterator

from pydantic import BaseModel

from agent.presets import PlanStep
from agent.registry import ToolRegistry
from tools.base import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class StepResult(BaseModel):
    """Outcome of one executed step.

    Attributes:
        step: The executed step.
        result: Tool result.
    """

    step: PlanStep
    result: ToolResult


class Executor:
    """Executes steps sequentially and stops at the first failure."""

    def __init__(self, tools: ToolRegistry) -> None:
        """Initialise the executor.

        Args:
            tools: Tool registry.
        """
        self._tools = tools

    def run(self, steps: list[PlanStep], ctx: ToolContext) -> Iterator[StepResult]:
        """Execute steps lazily.

        Args:
            steps: Ordered steps.
            ctx: Tool context.

        Yields:
            One :class:`StepResult` per executed step (the last one may be a failure).
        """
        for step in steps:
            args = {k: v for k, v in step.args.items() if v is not None}
            logger.info("Execute %s %s", step.tool, args)
            try:
                tool = self._tools.get(step.tool).as_langchain_tool(ctx)
                out = tool.invoke(args)
                result = out if isinstance(out, ToolResult) else ToolResult(message=str(out))
            except Exception as exc:  # noqa: BLE001 - tool failures must not crash the app
                logger.exception("Tool %s crashed", step.tool)
                result = ToolResult(success=False, message=f"Tool {step.tool} failed: {exc}")
            yield StepResult(step=step, result=result)
            if not result.success:
                return
