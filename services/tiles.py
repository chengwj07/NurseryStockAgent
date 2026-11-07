"""XYZ tile rendering for GeoTIFFs (rio-tiler)."""

from __future__ import annotations

import base64

from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader

from agent.state import ImageRecord

TRANSPARENT_PNG: bytes = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def render_tile(record: ImageRecord, z: int, x: int, y: int) -> bytes:
    """Render one 256px PNG tile.

    Args:
        record: Image to render.
        z: Zoom level.
        x: Tile column.
        y: Tile row.

    Returns:
        PNG bytes (a transparent pixel if the tile is outside the raster).
    """
    indexes = (1, 2, 3) if record.band_count >= 3 else (1,)
    try:
        with Reader(str(record.path)) as src:
            img = src.tile(x, y, z, indexes=indexes, tilesize=256)
    except TileOutsideBounds:
        return TRANSPARENT_PNG
    if record.rescale is not None:
        rescaled = img.rescale(in_range=(record.rescale,) * len(indexes))
        if rescaled is not None:
            img = rescaled
    return bytes(img.render(img_format="PNG"))
