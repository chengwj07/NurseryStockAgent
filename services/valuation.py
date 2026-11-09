"""Per-tree valuation: attaches ``price`` / ``price_text`` columns."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
import numpy as np

from services.dem import max_value_per_tree
from services.pricing import PricingModel

logger = logging.getLogger(__name__)

NO_FORMULA: str = "No valuation formula"
NO_HEIGHT: str = "No height data"

Mode = Literal["single", "dem"]


def price_trees(
    trees: gpd.GeoDataFrame, pricing: PricingModel, ndsm_path: Optional[Path]
) -> tuple[gpd.GeoDataFrame, Mode]:
    """Value every tree.

    Uses the original-image + DEM formulas when ``ndsm_path`` is given (the tree
    height H = highest nDSM value inside the crown, cached in ``height_m``),
    otherwise the original-image-only formulas.

    Args:
        trees: Trees with ``species`` and ``crown_diameter_m``.
        pricing: Price model.
        ndsm_path: nDSM (DEM - DTM) raster of the image, or ``None``.

    Returns:
        ``(trees_copy, mode)``; the copy has ``price`` (float, NaN if not
        valued) and ``price_text`` (display string) columns.
    """
    out = trees.copy()
    use_dem = ndsm_path is not None
    if use_dem:
        if "height_m" not in out.columns:
            out["height_m"] = np.nan
        missing = out["height_m"].isna().to_numpy()
        if missing.any():
            out.loc[missing, "height_m"] = max_value_per_tree(out[missing], ndsm_path)  # type: ignore[arg-type]
        heights = out["height_m"].to_numpy(dtype=float)
    else:
        heights = np.full(len(out), np.nan)

    prices: list[float] = []
    texts: list[str] = []
    for species, width, height in zip(out["species"], out["crown_diameter_m"], heights):
        if not pricing.has_formula(species):
            prices.append(np.nan)
            texts.append(NO_FORMULA)
            continue
        value = pricing.tree_price(species, float(width), float(height), use_dem)
        if value is None:
            prices.append(np.nan)
            texts.append(NO_HEIGHT)
        else:
            prices.append(value)
            texts.append(f"{value:,.2f} {pricing.currency}")
    out["price"] = prices
    out["price_text"] = texts
    mode: Mode = "dem" if use_dem else "single"
    logger.info("Priced %d trees (mode=%s)", len(out), mode)
    return out, mode
