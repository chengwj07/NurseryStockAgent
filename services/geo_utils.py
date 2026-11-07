"""Spatial helpers: region filtering."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import geopandas as gpd

logger = logging.getLogger(__name__)


def apply_region(
    trees: gpd.GeoDataFrame,
    bbox: Optional[Sequence[float]] = None,
    vector_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Restrict trees to a region given as bbox and/or a vector file.

    A tree is kept if its representative point lies inside the region.

    Args:
        trees: Tree polygons in EPSG:4326.
        bbox: ``[minx, miny, maxx, maxy]`` in longitude/latitude.
        vector_path: Path of a Shapefile/GeoJSON whose union is the region.

    Returns:
        The filtered GeoDataFrame (a copy).
    """
    result = trees
    points = result.geometry.representative_point()
    mask = points.notna()
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        mask &= (points.x >= minx) & (points.x <= maxx)
        mask &= (points.y >= miny) & (points.y <= maxy)
    if vector_path:
        region = gpd.read_file(vector_path).to_crs(4326)
        mask &= points.within(region.union_all())
    logger.info("Region filter kept %d / %d trees", int(mask.sum()), len(result))
    return result[mask].copy()
