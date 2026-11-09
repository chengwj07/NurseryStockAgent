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

TITLE_MD: str = """# 🌳 Nursery Asset Management Agent
**Nursery Asset Management Agent** · An all-in-one assistant for remote-sensing based nursery stock: **extract → count → value → tables / reports**. Upload an image (GeoTIFF, or PNG/JPG) and give commands in natural language, for example:
“Extract podocarpus macrophyllus”, “Count the trees”, “Evaluate the value”, “Output the area nursery asset ledger”, “Generate the plot statistics report”."""

EXAMPLES: list[str] = [
    "Extract podocarpus macrophyllus and count the trees",
    "Evaluate the value of the trees in the current image",
    "Output the area nursery asset ledger",
    "Output the plot census report",
    "Generate the plot nursery statistics report",
]


def _image_choices(session: SessionState) -> list[tuple[str, str]]:
    """Radio choices ``(label, image_id)`` of the loaded images."""
    return [
        (f"{img.name} (with DEM)" if img.dem_path else img.name, img.image_id)
        for img in session.images.values()
    ]


def _make_chatbot() -> gr.Chatbot:
    """Create a messages-format Chatbot on Gradio 5.x and 6.x.

    Gradio 6 removed the ``type`` argument (messages is the only format).

    Returns:
        The chatbot component.
    """
    try:
        return gr.Chatbot(type="messages", height=360, label="Conversation")
    except TypeError:
        return gr.Chatbot(height=360, label="Conversation")


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
                gr.Warning(f"Failed to load: {exc}")
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
            gr.Warning("Upload and select an original image first, then upload the DEM.")
            return session, gr.update()
        gr.Info("Deriving the DTM from the DEM (progressive morphological filter); large images may take a few minutes…")
        path = Path(files if isinstance(files, str) else files[-1])
        try:
            services.image_store.attach_dem(session.images[iid], path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("DEM upload failed")
            gr.Warning(f"Failed to attach the DEM: {exc}")
            return session, gr.update()
        trees = session.trees.get(iid)
        if trees is not None:  # cached heights / prices belong to the old DEM state
            session.trees[iid] = trees.drop(columns=["height_m", "price", "price_text"], errors="ignore")
        session.last_valuation = None
        gr.Info("DEM attached (DTM generated). Later valuations use the “original image + DEM” formulas; re-run any valuation you already did.")
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
        chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": "⏳ Parsing your request…"}]
        yield chat, "", session, gr.update(), gr.update()
        answer = ""
        files: list[str] = []
        version = session.version
        for event in agent.stream(message, session):
            answer += event.text
            files = event.files or files
            chat[-1] = {"role": "assistant", "content": answer or "⏳ Working…"}
            map_update: Any = gr.update()
            if session.version != version:
                version = session.version
                map_update = render_map(session)
            yield chat, "", session, map_update, (files or gr.update())
        yield chat, "", session, render_map(session), (files or gr.update())

    with gr.Blocks(title="Zhenong Nursery-Loan Agent") as demo:
        session_state = gr.State(lambda: SessionState())

        # ---- Top: title ----
        gr.Markdown(TITLE_MD)

        # ---- Middle: image manager | map ----
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Image Management")
                upload = gr.UploadButton(
                    "📤 Upload Image (GeoTIFF / PNG / JPG)",
                    file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"],
                    file_count="multiple",
                )
                image_list = gr.Radio(choices=[], label="Loaded images (click to switch the current image)")
                dem_upload = gr.UploadButton(
                    "⛰ Upload DEM for the current image (optional)", file_types=[".tif", ".tiff"], file_count="single"
                )
                remove_btn = gr.Button("🗑 Remove current image", size="sm")
            with gr.Column(scale=4):
                map_html = gr.HTML(render_map(SessionState()))

        # ---- Bottom: agent chat ----
        gr.Markdown("### Agent Chat")
        chatbot = _make_chatbot()
        with gr.Row():
            textbox = gr.Textbox(
                placeholder="Type a command, e.g. Extract podocarpus macrophyllus and count the trees", show_label=False, scale=8, lines=1
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("🧹 Clear chat history", scale=2)
        gr.Examples(EXAMPLES, inputs=textbox, label="Example commands")
        out_files = gr.File(label="Generated files (tables / reports)", file_count="multiple", interactive=False)

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
