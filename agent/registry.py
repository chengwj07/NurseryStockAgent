"""Tool registry with auto-discovery."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Iterator

from tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Holds all available tools by name."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register (or replace) a tool.

        Args:
            tool: Tool instance.
        """
        if tool.name in self._tools:
            logger.warning("Tool %s is replaced", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def discover(self, package: str = "tools") -> None:
        """Import every ``*_tool`` module of a package and register its tools.

        Args:
            package: Dotted package name to scan.
        """
        pkg = importlib.import_module(package)
        for info in pkgutil.iter_modules(pkg.__path__):
            if not info.name.endswith("_tool"):
                continue
            module = importlib.import_module(f"{package}.{info.name}")
            for obj in vars(module).values():
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseTool)
                    and obj is not BaseTool
                    and not inspect.isabstract(obj)
                    and obj.__module__ == module.__name__
                ):
                    self.register(obj())

    def get(self, name: str) -> BaseTool:
        """Return a tool by name.

        Args:
            name: Tool name.

        Returns:
            The tool.

        Raises:
            KeyError: If the tool is unknown.
        """
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        """Whether a tool name is registered."""
        return name in self._tools

    def __iter__(self) -> Iterator[BaseTool]:
        """Iterate over tools."""
        return iter(self._tools.values())

    def catalog(self) -> str:
        """Render the tool list for the planner prompt.

        Returns:
            Multi-line description of every tool and its arguments.
        """
        blocks: list[str] = []
        for tool in self:
            props = tool.args_schema.model_json_schema().get("properties", {})
            required = set(tool.args_schema.model_json_schema().get("required", []))
            args = []
            for key, prop in props.items():
                typ = prop.get("type") or "|".join(
                    str(t.get("type", "any")) for t in prop.get("anyOf", []) if t.get("type") != "null"
                )
                flag = "必填" if key in required else "可选"
                args.append(f"    - {key} ({typ}, {flag}): {prop.get('description', '')}")
            blocks.append(f"* {tool.name}: {tool.description}\n" + "\n".join(args))
        return "\n".join(blocks)
