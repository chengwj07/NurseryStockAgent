"""Entry point: ``python app.py``."""

from __future__ import annotations

import logging

import gradio as gr
import uvicorn
from fastapi import FastAPI

from agent.executor import Executor
from agent.llm_factory import LLMProvider
from agent.planner import Planner
from agent.presets import default_presets
from agent.registry import ToolRegistry
from agent.service import AgentService
from config import Settings, get_settings
from logging_config import setup_logging
from services.container import Services
from ui.api import register_routes
from ui.layout import build_demo

logger = logging.getLogger(__name__)


def build_agent(settings: Settings, services: Services, llm: LLMProvider | None = None) -> AgentService:
    """Wire registry, planner and executor into an :class:`AgentService`.

    Args:
        settings: Application settings.
        services: Shared services.
        llm: Optional LLM provider override (tests).

    Returns:
        The agent service.
    """
    registry = ToolRegistry()
    registry.discover("tools")
    presets = default_presets()
    planner = Planner(llm or LLMProvider(settings), registry, presets)
    return AgentService(planner, Executor(registry), services, settings)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI app with the Gradio UI mounted at ``/``.

    Args:
        settings: Optional settings override.

    Returns:
        The ASGI application.
    """
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.output_dir.parent / "logs")
    services = Services(settings)
    agent = build_agent(settings, services)
    app = FastAPI(title="Nursery Asset Agent")
    register_routes(app, services)
    demo = build_demo(services, agent)
    return gr.mount_gradio_app(app, demo, path="/", allowed_paths=[str(settings.output_dir)])


if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)
