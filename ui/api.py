"""FastAPI routes: XYZ tiles and GeoJSON layers consumed by the map iframe."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from services.container import Services
from services.tiles import render_tile

logger = logging.getLogger(__name__)


def register_routes(app: FastAPI, services: Services) -> None:
    """Attach tile / layer routes (and CORS) to an app.

    Args:
        app: FastAPI application.
        services: Shared services.
    """
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

    @app.get("/tiles/{image_id}/{z}/{x}/{y}.png")
    def tile(image_id: str, z: int, x: int, y: int) -> Response:
        record = services.image_store.get(image_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown image")
        try:
            content = render_tile(record, z, x, y)
        except Exception:
            logger.exception("Tile render failed: %s z=%d x=%d y=%d", image_id, z, x, y)
            raise HTTPException(status_code=500, detail="tile render failed")
        return Response(content, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.get("/layers/{layer_id}.geojson")
    def layer(layer_id: str) -> FileResponse:
        path = services.layer_store.path_of(layer_id)
        if path is None:
            raise HTTPException(status_code=404, detail="unknown layer")
        return FileResponse(path, media_type="application/geo+json")
