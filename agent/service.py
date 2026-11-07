"""Agent service: planner -> executor loop with optional re-planning."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Literal

from agent.executor import Executor, StepResult
from agent.planner import Planner
from agent.state import SessionState
from config import Settings
from services.container import Services
from tools.base import ToolContext

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """Progress event streamed to the UI.

    Attributes:
        kind: ``"plan"``, ``"step"`` or ``"done"``.
        text: Markdown text to append to the chat answer.
        files: Generated files (only on ``"done"``).
    """

    kind: Literal["plan", "step", "done"]
    text: str = ""
    files: list[str] = field(default_factory=list)


class AgentService:
    """Entry point of the agent."""

    def __init__(
        self, planner: Planner, executor: Executor, services: Services, settings: Settings
    ) -> None:
        """Initialise the service.

        Args:
            planner: Plan generator.
            executor: Step runner.
            services: Shared services.
            settings: Application settings.
        """
        self._planner = planner
        self._executor = executor
        self._services = services
        self._settings = settings

    def _history_text(self, session: SessionState) -> str:
        """Format the last turns for the prompt."""
        turns = session.history[-2 * self._settings.history_turns :]
        return "\n".join(f"{role}: {text}" for role, text in turns)

    def stream(self, message: str, session: SessionState) -> Iterator[AgentEvent]:
        """Handle one user message.

        Args:
            message: User text.
            session: Session state (mutated by tools).

        Yields:
            :class:`AgentEvent` objects; the last one has ``kind == "done"``.
        """
        ctx = ToolContext(session=session, services=self._services)
        history = self._history_text(session)
        feedback = ""
        files: list[str] = []
        summary_parts: list[str] = []

        try:
            plan = self._planner.plan(message, session, history)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Planner failed")
            yield AgentEvent("done", f"⚠️ 无法调用大模型：{exc}")
            return

        for attempt in range(self._settings.max_replans + 1):
            steps = self._planner.expand(plan)
            if not steps:
                reply = plan.reply or "好的。"
                session.history += [("用户", message), ("助手", reply)]
                yield AgentEvent("done", reply)
                return

            names = " → ".join(s.tool for s in steps)
            yield AgentEvent("plan", f"🧭 **执行计划**：{names}\n\n")

            failed: StepResult | None = None
            for idx, sr in enumerate(self._executor.run(steps, ctx), 1):
                icon = "✅" if sr.result.success else "❌"
                text = f"{icon} **{idx}. {sr.step.tool}**\n{sr.result.message}\n\n"
                summary_parts.append(sr.result.message)
                files += sr.result.files
                yield AgentEvent("step", text)
                if not sr.result.success:
                    failed = sr
            if failed is None:
                break
            if attempt < self._settings.max_replans:
                feedback = (
                    f"【上一次计划在工具 {failed.step.tool} 失败：{failed.result.message}。"
                    "请据此重新规划，必要时先补充缺失的前置步骤。】"
                )
                yield AgentEvent("plan", "🔁 执行失败，正在重新规划…\n\n")
                plan = self._planner.plan(message, session, history, feedback)
            else:
                break

        session.history += [("用户", message), ("助手", " ".join(summary_parts)[:300])]
        yield AgentEvent("done", "", files)
