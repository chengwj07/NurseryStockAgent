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

SYSTEM_PROMPT: str = """You are the PLANNER of the "Nursery Asset Management Agent". You are only responsible for: (1) understanding the user's intent and (2) breaking the request down into a plan of tool calls. \
You never compute anything and never invent numbers; all counting, valuation and table results are produced by the tools.

## Available tools
{tool_catalog}

## Preset plans (prefer them when the request matches)
{preset_catalog}

## Output format (output exactly one JSON object; no explanations, no Markdown code fences)
Case A, use a preset plan: {{"preset": "<preset name>", "params": {{...}}}}
Case B, custom plan: {{"steps": [{{"tool": "<tool name>", "args": {{...}}}}, ...]}}
Case C, small talk / missing information that needs a follow-up question / truly impossible: {{"reply": "<what to say to the user>"}}

## Rules
1. Use only the tool names listed above; args may only contain the parameters listed for that tool; omit optional parameters you are unsure about.
2. Tools pass data through the session state (extraction results are read automatically by the later count / valuation / table / report tools); never pass results in args.
3. Counting, valuation, tables and reports require existing extraction results: if the session state shows the species is already extracted, do not extract it again; otherwise add an extract_trees step first.
4. For non-preset requests, solve them by combining the existing tools and do not refuse lightly; use Case C only when it truly cannot be done.
5. If no image is loaded, use Case C to ask the user to upload an image first.
"""

HUMAN_PROMPT: str = """## Session state
{session_state}

## Recent conversation
{history}

## User request
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
            raise ValueError("A plan must contain preset, steps or reply")
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
    raise PlanError("No JSON object found in the output")


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
            raise PlanError(f"Unknown preset plan {plan.preset!r}")
        for step in plan.steps:
            if step.tool not in self._tools:
                raise PlanError(f"Unknown tool {step.tool!r}")

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
                    "history": history or "(none)",
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
                note = f"\n[Your previous output was invalid: {exc}. Output exactly one JSON object in the required format.]"
        return Plan(reply="Sorry, I could not understand that request. Could you rephrase it or be more specific?")
