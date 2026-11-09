"""DTM extraction from a DEM with the Progressive Morphological Filter (PMF).

Pipeline: DEM (absolute elevation) -> block-minimum grid -> PMF ground mask ->
interpolated DTM -> normalised surface ``nDSM = max(DEM - DTM, 0)`` written at
the full DEM resolution. The height of a tree is the highest nDSM value inside
its crown polygon.

PMF follows Zhang et al. (2003): the surface is opened with windows of growing
size; a pixel is non-ground if the opening lowers it by more than the
elevation-difference threshold of that window size.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator

from services.postprocess import pixel_area_m2

logger = logging.getLogger(__name__)

NODATA: float = -9999.0


@dataclass(frozen=True)
class PmfParams:
    """Parameters of the progressive morphological filter.

    Attributes:
        cell_size_m: Grid size (m) the filter runs at. The DEM is block-minimum
            reduced to this size (never up-sampled).
        max_window_m: Largest window (m); must exceed the largest object that
            should be removed (tree crowns, buildings are *not* removed unless
            smaller).
        slope: Terrain slope (rise/run) tolerated by the threshold.
        initial_dh_m: Elevation difference threshold of the first window (m).
        max_dh_m: Upper bound of the threshold (m).
    """

    cell_size_m: float = 1.0
    max_window_m: float = 20.0
    slope: float = 0.15
    initial_dh_m: float = 0.15
    max_dh_m: float = 3.0


def window_sizes(cell_m: float, max_window_m: float) -> list[int]:
    """Window sizes in cells: 3, 5, 9, 17, 33 ... until ``max_window_m``.

    Args:
        cell_m: Grid size in metres.
        max_window_m: Largest window in metres.

    Returns:
        Odd window sizes (cells), at least one.
    """
    sizes: list[int] = []
    k = 1
    while True:
        size = 2**k + 1
        sizes.append(size)
        if size * cell_m >= max_window_m:
            return sizes
        k += 1


def _fill_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid pixels by the value of the nearest valid pixel."""
    if valid.all():
        return values
    if not valid.any():
        raise ValueError("The DEM has no valid pixels")
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(idx)]


def pmf_ground_mask(z: np.ndarray, cell_m: float, params: PmfParams) -> np.ndarray:
    """Classify pixels as ground / non-ground with PMF.

    Args:
        z: 2-D elevation grid (NaN = no data).
        cell_m: Grid size in metres.
        params: Filter parameters.

    Returns:
        Boolean array, ``True`` for ground pixels (invalid pixels are False).
    """
    valid = np.isfinite(z)
    surface = _fill_nearest(np.where(valid, z, np.nan).astype(np.float32), valid)
    ground = valid.copy()
    limit = max(3, (min(z.shape) // 2) * 2 + 1)
    previous = 1
    for index, size in enumerate(window_sizes(cell_m, params.max_window_m)):
        size = min(size, limit)
        opened = ndimage.grey_opening(surface, size=(size, size))
        if index == 0:
            threshold = params.initial_dh_m
        else:
            threshold = min(params.max_dh_m, params.slope * (size - previous) * cell_m + params.initial_dh_m)
        ground &= (surface - opened) <= threshold
        surface, previous = opened, size
    logger.info("PMF: %.1f%% ground pixels", 100.0 * ground.sum() / max(valid.sum(), 1))
    return ground


def interpolate_dtm(z: np.ndarray, ground: np.ndarray, max_points: int = 200_000) -> np.ndarray:
    """Build a DTM by interpolating the ground pixels.

    Linear (Delaunay) interpolation on a decimated set of ground pixels, with
    nearest-ground fallback outside the convex hull.

    Args:
        z: Elevation grid.
        ground: Ground mask from :func:`pmf_ground_mask`.
        max_points: Maximum number of ground points used for interpolation.

    Returns:
        DTM grid (float32) of the same shape as ``z``.

    Raises:
        ValueError: If there are no ground pixels.
    """
    ground = ground & np.isfinite(z)
    if not ground.any():
        raise ValueError("No ground points left after filtering; check the DEM or adjust the PMF parameters")
    dtm = np.where(ground, z, np.nan).astype(np.float32)
    todo = ~ground
    if not todo.any():
        return dtm
    idx = ndimage.distance_transform_edt(todo, return_distances=False, return_indices=True)
    fill = z[tuple(idx)][todo].astype(np.float64)
    try:
        ys, xs = np.nonzero(ground)
        step = len(ys) // max_points + 1
        ys, xs = ys[::step], xs[::step]
        if len(ys) >= 4:
            interp = LinearNDInterpolator(np.column_stack([ys, xs]), z[ys, xs])
            targets = np.argwhere(todo)
            vals = np.concatenate(
                [interp(targets[i : i + 500_000]) for i in range(0, len(targets), 500_000)]
            )
            fill = np.where(np.isfinite(vals), vals, fill)
    except Exception:  # noqa: BLE001 - nearest-ground fill is an acceptable fallback
        logger.warning("Linear DTM interpolation failed, using nearest ground", exc_info=True)
    dtm[todo] = fill
    return dtm


def _block_min(ds: rasterio.DatasetReader, factor: int, strip_rows: int = 256) -> np.ndarray:
    """Block-minimum reduction of band 1 (NaN where a block has no valid pixel)."""
    height, width = ds.height, ds.width
    out_h, out_w = -(-height // factor), -(-width // factor)
    out = np.full((out_h, out_w), np.nan, dtype=np.float32)
    for r0 in range(0, out_h, strip_rows):
        r1 = min(out_h, r0 + strip_rows)
        window = Window(0, r0 * factor, width, min(height, r1 * factor) - r0 * factor)
        data = ds.read(1, window=window, masked=True)
        arr = np.where(np.ma.getmaskarray(data) | ~np.isfinite(data.data), np.inf, data.data).astype(np.float32)
        pad_h, pad_w = (r1 - r0) * factor - arr.shape[0], out_w * factor - width
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=np.inf)
        block = arr.reshape(r1 - r0, factor, out_w, factor).min(axis=(1, 3))
        block[np.isinf(block)] = np.nan
        out[r0:r1] = block
    return out


def build_ndsm(dem_path: Path, out_path: Path, params: PmfParams, strip_rows: int = 512) -> None:
    """Create ``nDSM = max(DEM - DTM, 0)`` at the DEM's resolution.

    Args:
        dem_path: DEM GeoTIFF with absolute elevations (band 1).
        out_path: Output GeoTIFF (float32, nodata -9999).
        params: PMF parameters.
        strip_rows: Rows processed at a time.

    Raises:
        ValueError: If the DEM is rotated or has no usable pixels.
    """
    with rasterio.open(dem_path) as ds:
        tf = ds.transform
        if tf.b != 0 or tf.d != 0:
            raise ValueError("Rotated DEMs are not supported")
        cell = math.sqrt(pixel_area_m2(ds.crs, tf, ds.height))
        factor = max(1, int(round(params.cell_size_m / cell)))
        logger.info("DEM cell %.3f m, PMF grid factor %d", cell, factor)
        coarse = _block_min(ds, factor)
        ground = pmf_ground_mask(coarse, cell * factor, params)
        dtm = interpolate_dtm(coarse, ground)

        profile = {
            "driver": "GTiff", "height": ds.height, "width": ds.width, "count": 1, "dtype": "float32",
            "crs": ds.crs, "transform": tf, "nodata": NODATA, "tiled": True,
            "blockxsize": 256, "blockysize": 256, "compress": "deflate",
        }
        cols = (np.arange(ds.width) + 0.5) / factor - 0.5
        with rasterio.open(out_path, "w", **profile) as out:
            for r0 in range(0, ds.height, strip_rows):
                rows = min(strip_rows, ds.height - r0)
                window = Window(0, r0, ds.width, rows)
                dem = ds.read(1, window=window, masked=True)
                invalid = np.ma.getmaskarray(dem) | ~np.isfinite(dem.data)
                rr, cc = np.meshgrid((np.arange(r0, r0 + rows) + 0.5) / factor - 0.5, cols, indexing="ij")
                dtm_strip = ndimage.map_coordinates(dtm, [rr, cc], order=1, mode="nearest")
                height = np.clip(dem.data.astype(np.float32) - dtm_strip.astype(np.float32), 0, None)
                out.write(np.where(invalid, NODATA, height).astype(np.float32), 1, window=window)
    logger.info("nDSM written: %s", out_path.name)
