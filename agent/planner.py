"""Planner: the LLM only turns natural language into a tool-call plan (JSON)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError, model_validator

from agent.llm_factory import LLMProvider
from agent.presets import PlanStep, PresetRegistry
from agent.registry import ToolRegistry
from agent.state import SessionState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = """你是"苗木资产管理智能体"的【规划器(Planner)】。你只负责：①理解用户意图 ②把需求拆解成对工具的调用计划。\
你不做任何计算，不编造任何数字；统计、估值、制表等结果全部由工具产出。

## 可用工具
{tool_catalog}

## 预制计划（用户需求与之吻合时优先使用）
{preset_catalog}

## 输出格式（只输出一个JSON对象，不要解释，不要Markdown代码块）
情况A 使用预制计划: {{"preset": "<预制计划名>", "params": {{...}}}}
情况B 自定义计划: {{"steps": [{{"tool": "<工具名>", "args": {{...}}}}, ...]}}
情况C 闲聊 / 信息不足需追问 / 确实无法完成: {{"reply": "<对用户说的话>"}}

## 规则
1. 只能使用上面列出的工具名；args 只能包含该工具列出的参数；不确定的可选参数直接省略。
2. 工具间通过会话状态传递数据（提取结果会被后续统计/估值/制表/报告自动读取），不要在args里传结果。
3. 统计、估值、制表、报告之前必须已有提取结果：会话状态里已提取过该树种则不要重复提取，否则先加入 extract_trees 步骤。
4. 非预制需求：优先用已有工具组合解决，不要轻易拒绝；确实做不到时才用情况C说明原因。
5. 没有已加载图像时使用情况C提醒用户先上传图像。
"""

HUMAN_PROMPT: str = """## 会话状态
{session_state}

## 最近对话
{history}

## 用户请求
{user_input}
{feedback}"""


class PlanError(ValueError):
    """The LLM output is not a valid plan."""


class Plan(BaseModel):
    """Structured planner output.

    Attributes:
        preset: Name of a preset plan, if used.
        params: Preset parameters.
        steps: Custom tool-call steps.
        reply: Direct reply (chit-chat / clarification).
    """

    preset: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    steps: list[PlanStep] = Field(default_factory=list)
    reply: Optional[str] = None

    @model_validator(mode="after")
    def _non_empty(self) -> "Plan":
        """Require at least one of preset / steps / reply."""
        if not (self.preset or self.steps or self.reply):
            raise ValueError("计划必须包含 preset、steps 或 reply 之一")
        return self


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from LLM text.

    Handles ``<think>`` blocks and Markdown fences.

    Args:
        text: Raw model output.

    Returns:
        The parsed object.

    Raises:
        PlanError: If no JSON object is found.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise PlanError("输出中没有找到JSON对象")


class Planner:
    """Builds plans with an LLM chain (prompt | llm | parser)."""

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        presets: PresetRegistry,
        max_parse_retries: int = 1,
    ) -> None:
        """Initialise the planner.

        Args:
            llm: Chat model provider.
            tools: Tool registry (for the prompt catalog and validation).
            presets: Preset plan registry.
            max_parse_retries: Extra attempts when the output is invalid.
        """
        self._llm = llm
        self._tools = tools
        self._presets = presets
        self._retries = max_parse_retries
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )

    def validate(self, plan: Plan) -> None:
        """Check tool and preset names.

        Args:
            plan: Parsed plan.

        Raises:
            PlanError: If an unknown tool or preset is referenced.
        """
        if plan.preset and plan.preset not in self._presets:
            raise PlanError(f"未知预制计划 {plan.preset!r}")
        for step in plan.steps:
            if step.tool not in self._tools:
                raise PlanError(f"未知工具 {step.tool!r}")

    def expand(self, plan: Plan) -> list[PlanStep]:
        """Resolve a plan to concrete steps.

        Args:
            plan: Validated plan.

        Returns:
            Ordered tool-call steps (empty for a pure reply).
        """
        if plan.preset:
            return self._presets.expand(plan.preset, plan.params)
        return plan.steps

    def plan(
        self, user_input: str, session: SessionState, history: str, feedback: str = ""
    ) -> Plan:
        """Ask the LLM for a plan.

        Args:
            user_input: Latest user message.
            session: Current session (summarised into the prompt).
            history: Formatted recent conversation.
            feedback: Extra text, e.g. the error of a failed previous plan.

        Returns:
            A validated plan; falls back to a clarifying reply on failure.
        """
        chain = self._prompt | self._llm.get() | StrOutputParser()
        note = ""
        for attempt in range(self._retries + 1):
            raw = chain.invoke(
                {
                    "tool_catalog": self._tools.catalog(),
                    "preset_catalog": self._presets.catalog(),
                    "session_state": session.describe(),
                    "history": history or "（无）",
                    "user_input": user_input,
                    "feedback": feedback + note,
                }
            )
            logger.debug("Planner raw output (attempt %d): %s", attempt, raw)
            try:
                plan = Plan.model_validate(extract_json(raw))
                self.validate(plan)
                logger.info("Plan: preset=%s steps=%s", plan.preset, [s.tool for s in plan.steps])
                return plan
            except (PlanError, ValidationError, ValueError) as exc:
                logger.warning("Invalid plan (attempt %d): %s", attempt, exc)
                note = f"\n【你上一次的输出不合法：{exc}。请严格按格式重新输出一个JSON对象。】"
        return Plan(reply="抱歉，我没能理解这个需求，能换一种说法或说得更具体一些吗？")
