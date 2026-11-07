"""Helpers shared by several tools."""

from __future__ import annotations

from typing import Optional

import geopandas as gpd

from agent.state import ImageRecord
from services.geo_utils import apply_region
from services.valuation import price_trees
from tools.base import RegionArgs, ToolContext, ToolError


def resolve_image(ctx: ToolContext, image_id: Optional[str]) -> ImageRecord:
    """Return the requested (or current) image.

    Args:
        ctx: Tool context.
        image_id: Explicit id or ``None`` for the current image.

    Returns:
        The image record.

    Raises:
        ToolError: If no usable image exists.
    """
    iid = image_id or ctx.session.current_image_id
    if iid is None or iid not in ctx.session.images:
        raise ToolError("没有可用的图像，请先在左侧上传并选择一张图像。")
    return ctx.session.images[iid]


def collect_trees(
    ctx: ToolContext, args: RegionArgs, species: Optional[str] = None
) -> tuple[ImageRecord, gpd.GeoDataFrame]:
    """Fetch extracted trees for an image, filtered by species and region.

    Args:
        ctx: Tool context.
        args: Image / region arguments.
        species: Optional species filter.

    Returns:
        ``(image, trees)``; ``trees`` may be empty after filtering.

    Raises:
        ToolError: If nothing has been extracted for the image yet.
    """
    image = resolve_image(ctx, args.image_id)
    trees = ctx.session.trees.get(image.image_id)
    if trees is None or trees.empty:
        raise ToolError("当前图像还没有提取结果，请先提取苗木（extract_trees）。")
    if species:
        canonical = ctx.services.pricing.resolve(species) or species
        trees = trees[trees["species"] == canonical]
    if args.bbox is not None or args.region_vector_path:
        trees = apply_region(trees, args.bbox, args.region_vector_path)
    return image, trees


def region_label(args: RegionArgs, default: str = "区域") -> str:
    """Return a display name for the statistics range.

    Args:
        args: Image / region arguments.
        default: Fallback label.

    Returns:
        Region name.
    """
    return (args.region_name or "").strip() or default


def ensure_priced(ctx: ToolContext, image: ImageRecord) -> str:
    """Value every extracted tree of an image and refresh the map layers.

    Adds ``price`` / ``price_text`` to ``session.trees`` and rewrites the
    GeoJSON of the image's layers so the map tooltip shows the price.

    Args:
        ctx: Tool context.
        image: The image whose trees are valued.

    Returns:
        ``"dem"`` (original image + DEM formulas) or ``"single"``.

    Raises:
        ToolError: If nothing has been extracted for the image yet.
    """
    trees = ctx.session.trees.get(image.image_id)
    if trees is None or trees.empty:
        raise ToolError("当前图像还没有提取结果，请先提取苗木（extract_trees）。")
    priced, mode = price_trees(trees, ctx.services.pricing, image.ndsm_path)
    ctx.session.trees[image.image_id] = priced
    for layer in ctx.session.layers:
        if layer.image_id == image.image_id and layer.species:
            ctx.services.layer_store.write(layer.layer_id, priced[priced["species"] == layer.species])
    ctx.session.touch()
    return mode
