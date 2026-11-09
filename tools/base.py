"""Tool abstractions: every executable capability derives from :class:`BaseTool`."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from agent.state import SessionState
from services.container import Services

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """A user-facing, expected failure (e.g. no image loaded)."""


class ToolResult(BaseModel):
    """Uniform tool output.

    Attributes:
        success: Whether the tool succeeded.
        message: Markdown text shown to the user.
        data: Small machine-readable payload.
        files: Generated files offered for download.
    """

    success: bool = True
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    files: list[str] = Field(default_factory=list)


@dataclass
class ToolContext:
    """Runtime context handed to a tool.

    Attributes:
        session: Mutable session state.
        services: Shared services.
    """

    session: SessionState
    services: Services


class RegionArgs(BaseModel):
    """Common arguments selecting an image and a statistics range."""

    image_id: Optional[str] = Field(None, description="Image id; omit to use the current image")
    region_name: Optional[str] = Field(None, description="Area / plot name, used in report titles")
    bbox: Optional[list[float]] = Field(
        None, description="Statistics range [minx,miny,maxx,maxy] in lon/lat (EPSG:4326); omit for the whole image"
    )
    region_vector_path: Optional[str] = Field(
        None, description="Path of a range vector file (shp/geojson); omit for no vector clipping"
    )

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, value: Optional[list[float]]) -> Optional[list[float]]:
        """Validate that ``bbox`` has four numbers."""
        if value is not None and len(value) != 4:
            raise ValueError("bbox must be [minx, miny, maxx, maxy]")
        return value


class BaseTool(ABC):
    """Base class of all tools.

    To add a tool, create ``tools/<anything>_tool.py`` with a subclass; the
    :class:`agent.registry.ToolRegistry` discovers it automatically.

    Attributes:
        name: Unique tool name used by the planner.
        description: What the tool does (shown to the LLM).
        args_schema: Pydantic model of the arguments.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, ctx: ToolContext, args: Any) -> ToolResult:
        """Execute the tool.

        Args:
            ctx: Runtime context.
            args: Validated arguments (instance of ``args_schema``).

        Returns:
            The tool result.

        Raises:
            ToolError: For expected, user-facing failures.
        """

    def as_langchain_tool(self, ctx: ToolContext) -> StructuredTool:
        """Wrap the tool as a LangChain ``StructuredTool`` bound to a context.

        Args:
            ctx: Runtime context to bind.

        Returns:
            A structured tool whose ``invoke`` returns a :class:`ToolResult`.
        """

        def _call(**kwargs: Any) -> ToolResult:
            try:
                return self.run(ctx, self.args_schema(**kwargs))
            except ToolError as exc:
                logger.warning("Tool %s failed: %s", self.name, exc)
                return ToolResult(success=False, message=str(exc))

        return StructuredTool.from_function(
            func=_call,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )
