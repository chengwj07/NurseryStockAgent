"""Raster post-processing of a merged segmentation mask.

Pipeline (used for counting): merged binary mask of one species ->
fill small holes -> remove small debris -> connected-component labelling ->
one polygon per component.  The number of components is the tree count.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

_EIGHT = np.ones((3, 3), dtype=bool)  # 8-connectivity for objects (4 for background)


@dataclass(frozen=True)
class CleanParams:
    """Thresholds of the cleaning step.

    Attributes:
        min_object_area_m2: Components smaller than this are debris and removed.
        max_hole_area_m2: Enclosed background regions up to this size are filled.
        max_pixels: Masks larger than this are block-reduced before processing.
    """

    min_object_area_m2: float = 1.0
    max_hole_area_m2: float = 1.0
    max_pixels: int = 250_000_000


@dataclass
class CleanStats:
    """Numbers reported after cleaning.

    Attributes:
        raw_components: Connected components before any cleaning.
        filled_holes: Number of small holes filled.
        removed_debris: Components removed as debris.
        final_count: Components kept (= tree count).
        scale: Block-reduction factor that was applied (1 = full resolution).
    """

    raw_components: int
    filled_holes: int
    removed_debris: int
    final_count: int
    scale: int


def pixel_area_m2(crs: CRS, transform: Affine, height: int) -> float:
    """Approximate ground area of one pixel.

    Args:
        crs: Raster CRS.
        transform: Full-resolution affine transform.
        height: Raster height in pixels (to find the centre latitude).

    Returns:
        Pixel area in square metres.
    """
    if crs.is_geographic:
        lat = transform.f + transform.e * height / 2.0
        dx = abs(transform.a) * 111_320.0 * math.cos(math.radians(lat))
        dy = abs(transform.e) * 110_540.0
        return dx * dy
    try:
        factor = float(crs.linear_units_factor[1])
    except Exception:  # noqa: BLE001 - unknown units: assume metres
        factor = 1.0
    return abs(transform.a) * factor * abs(transform.e) * factor


def scale_transform(transform: Affine, scale: int) -> Affine:
    """Return ``transform`` for a mask whose pixels are ``scale`` times larger.

    Written out explicitly because ``Affine * Affine`` (affine < 3) and
    ``Affine @ Affine`` (affine >= 2.4 / 3) are not available in all versions.

    Args:
        transform: Full-resolution transform.
        scale: Integer block factor.

    Returns:
        The transform of the reduced raster (same origin, larger pixels).
    """
    t = transform
    return Affine(t.a * scale, t.b * scale, t.c, t.d * scale, t.e * scale, t.f)


def fill_small_holes(mask: np.ndarray, max_area_px: int) -> tuple[np.ndarray, int]:
    """Fill enclosed background regions not larger than ``max_area_px``.

    Background touching the image border is never filled, and large enclosed
    background (e.g. a ring of trees) is kept.

    Args:
        mask: Boolean mask.
        max_area_px: Largest hole (pixels) that is filled.

    Returns:
        ``(filled_mask, number_of_filled_holes)``.
    """
    if max_area_px < 1 or not mask.any():
        return mask, 0
    labels, n = ndimage.label(~mask)  # 4-connected background
    if n == 0:
        return mask, 0
    sizes = np.bincount(labels.ravel())
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    fill = sizes <= max_area_px
    fill[0] = False
    fill[border] = False
    if not fill.any():
        return mask, 0
    return mask | fill[labels], int(fill.sum())


def clean_and_label(mask: np.ndarray, min_area_px: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Remove small components and label the rest.

    Args:
        mask: Boolean mask.
        min_area_px: Components with fewer pixels are removed.

    Returns:
        ``(labels, sizes, raw_components, removed)``: ``labels`` is int32 with
        ids ``1..N`` (0 = background); ``sizes[i]`` is the pixel count of id ``i``.
    """
    labels, raw = ndimage.label(mask, structure=_EIGHT)
    if raw == 0:
        return labels.astype(np.int32), np.zeros(1, dtype=np.int64), 0, 0
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(min_area_px, 1)
    keep[0] = False
    new_ids = np.cumsum(keep) * keep
    labels = new_ids.astype(np.int32)[labels]
    new_sizes = np.bincount(labels.ravel(), minlength=int(keep.sum()) + 1)
    return labels, new_sizes, int(raw), int(raw - keep.sum())


def reduce_mask(mm: np.ndarray, scale: int, strip_rows: int = 256) -> np.ndarray:
    """Block-reduce a (possibly memory-mapped) mask with a max filter.

    Args:
        mm: 2-D uint8/bool array of shape ``(H, W)``.
        scale: Integer reduction factor.
        strip_rows: Output rows processed at a time (limits memory).

    Returns:
        Boolean array of shape ``(ceil(H/scale), ceil(W/scale))``.
    """
    height, width = mm.shape
    out_h, out_w = -(-height // scale), -(-width // scale)
    out = np.zeros((out_h, out_w), dtype=bool)
    for r0 in range(0, out_h, strip_rows):
        r1 = min(out_h, r0 + strip_rows)
        block = np.asarray(mm[r0 * scale : min(height, r1 * scale)])
        pad_h = (r1 - r0) * scale - block.shape[0]
        pad_w = out_w * scale - width
        if pad_h or pad_w:
            block = np.pad(block, ((0, pad_h), (0, pad_w)))
        out[r0:r1] = block.reshape(r1 - r0, scale, out_w, scale).max(axis=(1, 3)) > 0
    return out


def labels_to_polygons(labels: np.ndarray, transform: Affine, simplify_tol: float) -> list[BaseGeometry]:
    """Vectorise labelled components, one geometry per label id.

    Args:
        labels: int32 label raster.
        transform: Affine transform of ``labels``.
        simplify_tol: Douglas-Peucker tolerance in CRS units.

    Returns:
        Geometries ordered by label id (1..N).
    """
    parts: dict[int, list[BaseGeometry]] = defaultdict(list)
    for geom, value in shapes(labels, mask=labels > 0, transform=transform, connectivity=8):
        parts[int(value)].append(shape(geom))
    result: list[BaseGeometry] = []
    for label_id in sorted(parts):
        geom = unary_union(parts[label_id])
        result.append(geom.simplify(simplify_tol, preserve_topology=True) if simplify_tol > 0 else geom)
    return result


def mask_to_trees(
    mask_source: np.ndarray,
    transform: Affine,
    crs: CRS,
    params: CleanParams,
) -> tuple[list[BaseGeometry], list[float], CleanStats]:
    """Full counting pipeline on a merged mask.

    Args:
        mask_source: Full-resolution 2-D mask (array or ``np.memmap``).
        transform: Full-resolution affine transform.
        crs: Raster CRS.
        params: Cleaning thresholds.

    Returns:
        ``(polygons, areas_m2, stats)``; one polygon / area per counted tree.
    """
    height, width = mask_source.shape
    pix_area = pixel_area_m2(crs, transform, height)
    scale = 1
    if height * width > params.max_pixels:
        scale = math.ceil(math.sqrt(height * width / params.max_pixels))
        logger.warning("Mask %dx%d too large, reducing by %d for counting", width, height, scale)
        mask = reduce_mask(mask_source, scale)
    else:
        mask = np.asarray(mask_source) > 0
    tf = scale_transform(transform, scale)
    area_per_px = pix_area * scale * scale

    hole_px = int(round(params.max_hole_area_m2 / area_per_px)) if area_per_px > 0 else 0
    min_px = int(round(params.min_object_area_m2 / area_per_px)) if area_per_px > 0 else 1
    mask, filled = fill_small_holes(mask, hole_px)
    labels, sizes, raw, removed = clean_and_label(mask, min_px)
    count = int(labels.max()) if labels.size else 0

    tol = 0.5 * min(abs(tf.a), abs(tf.e))
    polygons = labels_to_polygons(labels, tf, tol) if count else []
    areas = [float(sizes[i] * area_per_px) for i in range(1, count + 1)]
    stats = CleanStats(raw, filled, removed, count, scale)
    logger.info("Counting: %s", stats)
    return polygons, areas, stats
