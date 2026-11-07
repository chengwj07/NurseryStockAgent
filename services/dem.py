"""Raster sampling per tree crown."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

logger = logging.getLogger(__name__)


def max_value_per_tree(trees: gpd.GeoDataFrame, raster_path: Path) -> np.ndarray:
    """Highest raster value (band 1) inside every tree polygon.

    Used with the nDSM raster: the result is the tree height H.

    Args:
        trees: Tree polygons (any CRS; EPSG:4326 in this app).
        raster_path: Raster to sample.

    Returns:
        Float array aligned with ``trees``; ``NaN`` where the polygon does not
        overlap valid pixels.
    """
    out = np.full(len(trees), np.nan, dtype=float)
    if trees.empty:
        return out
    with rasterio.open(raster_path) as ds:
        geoms = trees.geometry.to_crs(ds.crs)
        for i, geom in enumerate(geoms):
            try:
                arr, _ = rio_mask(ds, [geom], crop=True, all_touched=True, filled=False, indexes=1)
            except ValueError:  # polygon does not overlap the raster
                continue
            values = np.asarray(arr.compressed(), dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                out[i] = float(values.max())
    logger.info("Per-tree max: %d / %d trees covered", int(np.isfinite(out).sum()), len(out))
    return out
