"""Gradio UI: title / (image manager | map) / agent chat."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import gradio as gr

from agent.service import AgentService
from agent.state import SessionState
from services.container import Services
from ui.map_view import build_map_html

logger = logging.getLogger(__name__)

TITLE_MD: str = """# 🌳 浙农苗木贷智能体
**苗木资产管理智能体** · 基于遥感影像的苗木 **提取 → 统计 → 估值 → 报表 / 报告** 一体化助手。上传影像（GeoTIFF，或 PNG/JPG）后，用自然语言下达指令，例如：
「提取罗汉松」「统计数量」「评估价值」「输出区域苗木资产核算表」「生成地块的统计报告」。"""

EXAMPLES: list[str] = [
    "提取罗汉松，并统计数量",
    "评估当前图像中苗木的价值",
    "输出区域苗木资产核算表",
    "输出地块普查报表",
    "生成地块的苗木统计报告",
]


def _image_choices(session: SessionState) -> list[tuple[str, str]]:
    """Radio choices ``(label, image_id)`` of the loaded images."""
    return [
        (f"{img.name}（含DEM）" if img.dem_path else img.name, img.image_id)
        for img in session.images.values()
    ]


def _make_chatbot() -> gr.Chatbot:
    """Create a messages-format Chatbot on Gradio 5.x and 6.x.

    Gradio 6 removed the ``type`` argument (messages is the only format).

    Returns:
        The chatbot component.
    """
    try:
        return gr.Chatbot(type="messages", height=360, label="对话")
    except TypeError:
        return gr.Chatbot(height=360, label="对话")


def build_demo(services: Services, agent: AgentService) -> gr.Blocks:
    """Build the Gradio Blocks app.

    Args:
        services: Shared services.
        agent: Agent entry point.

    Returns:
        The (not yet launched) Blocks app.
    """
    base_url = services.settings.public_base_url

    def render_map(session: SessionState) -> str:
        cfg = services.settings
        return build_map_html(
            session, services.image_store, base_url,
            init_view=(cfg.map_init_lat, cfg.map_init_lon, cfg.map_init_zoom),
        )

    def on_upload(files: Optional[list[str]], session: SessionState) -> tuple[Any, ...]:
        """Register uploaded GeoTIFFs."""
        for f in files or []:
            try:
                record = services.image_store.register(Path(f))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Upload failed")
                gr.Warning(f"加载失败：{exc}")
                continue
            session.images[record.image_id] = record
            session.current_image_id = record.image_id
        session.touch()
        return (
            session,
            gr.update(choices=_image_choices(session), value=session.current_image_id),
            render_map(session),
        )

    def on_upload_dem(files: Any, session: SessionState) -> tuple[Any, ...]:
        """Attach a DEM to the current image (enables the DEM valuation formulas)."""
        iid = session.current_image_id
        if not files:
            return session, gr.update()
        if iid is None or iid not in session.images:
            gr.Warning("请先上传并选择一张原图，再上传DEM。")
            return session, gr.update()
        gr.Info("正在由 DEM 生成 DTM（渐进形态学滤波），大影像可能需要几分钟…")
        path = Path(files if isinstance(files, str) else files[-1])
        try:
            services.image_store.attach_dem(session.images[iid], path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("DEM upload failed")
            gr.Warning(f"DEM 关联失败：{exc}")
            return session, gr.update()
        trees = session.trees.get(iid)
        if trees is not None:  # cached heights / prices belong to the old DEM state
            session.trees[iid] = trees.drop(columns=["height_m", "price", "price_text"], errors="ignore")
        session.last_valuation = None
        gr.Info("已关联DEM（已生成DTM），之后的资产评估将使用「原图+DEM」公式；已评估过的请重新评估。")
        return session, gr.update(choices=_image_choices(session), value=iid)

    def on_select(image_id: Optional[str], session: SessionState) -> tuple[Any, ...]:
        """Switch the current image and zoom to it."""
        if image_id in session.images:
            session.current_image_id = image_id
        return session, render_map(session)

    def on_remove(session: SessionState) -> tuple[Any, ...]:
        """Remove the selected image and its results."""
        iid = session.current_image_id
        if iid:
            session.images.pop(iid, None)
            session.trees.pop(iid, None)
            services.image_store.remove(iid)
            session.current_image_id = next(iter(session.images), None)
        return (
            session,
            gr.update(choices=_image_choices(session), value=session.current_image_id),
            render_map(session),
        )

    def on_send(
        message: str, chat: list[dict[str, str]], session: SessionState
    ) -> Iterator[tuple[Any, ...]]:
        """Stream the agent's progress into the chat."""
        message = (message or "").strip()
        if not message:
            yield chat, "", session, gr.update(), gr.update()
            return
        chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": "⏳ 正在解析意图…"}]
        yield chat, "", session, gr.update(), gr.update()
        answer = ""
        files: list[str] = []
        version = session.version
        for event in agent.stream(message, session):
            answer += event.text
            files = event.files or files
            chat[-1] = {"role": "assistant", "content": answer or "⏳ 处理中…"}
            map_update: Any = gr.update()
            if session.version != version:
                version = session.version
                map_update = render_map(session)
            yield chat, "", session, map_update, (files or gr.update())
        yield chat, "", session, render_map(session), (files or gr.update())

    with gr.Blocks(title="浙农苗木贷智能体") as demo:
        session_state = gr.State(lambda: SessionState())

        # ---- Top: title ----
        gr.Markdown(TITLE_MD)

        # ---- Middle: image manager | map ----
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### 图像管理")
                upload = gr.UploadButton(
                    "📤 上传图像 (GeoTIFF / PNG / JPG)",
                    file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"],
                    file_count="multiple",
                )
                image_list = gr.Radio(choices=[], label="已加载图像（点击切换当前图像）")
                dem_upload = gr.UploadButton(
                    "⛰ 为当前图像上传DEM（可选）", file_types=[".tif", ".tiff"], file_count="single"
                )
                remove_btn = gr.Button("🗑 移除当前图像", size="sm")
            with gr.Column(scale=4):
                map_html = gr.HTML(render_map(SessionState()))

        # ---- Bottom: agent chat ----
        gr.Markdown("### 智能体对话")
        chatbot = _make_chatbot()
        with gr.Row():
            textbox = gr.Textbox(
                placeholder="输入指令，如：提取罗汉松并统计数量", show_label=False, scale=8, lines=1
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)
            clear_btn = gr.Button("🧹 清空聊天记录", scale=2)
        gr.Examples(EXAMPLES, inputs=textbox, label="示例指令")
        out_files = gr.File(label="生成的文件（报表 / 报告）", file_count="multiple", interactive=False)

        # ---- Events ----
        upload.upload(on_upload, [upload, session_state], [session_state, image_list, map_html])
        dem_upload.upload(on_upload_dem, [dem_upload, session_state], [session_state, image_list])
        image_list.input(on_select, [image_list, session_state], [session_state, map_html])
        remove_btn.click(on_remove, [session_state], [session_state, image_list, map_html])
        chat_io = dict(
            fn=on_send,
            inputs=[textbox, chatbot, session_state],
            outputs=[chatbot, textbox, session_state, map_html, out_files],
        )
        textbox.submit(**chat_io)
        send_btn.click(**chat_io)

        def on_clear(session: SessionState) -> tuple[Any, ...]:
            """Clear chat window, input box, download list and the planner's memory."""
            session.history.clear()
            return [], "", None

        clear_btn.click(on_clear, [session_state], [chatbot, textbox, out_files], queue=False)

    demo.queue()
    return demo
