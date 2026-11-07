"""Aggregation of tree polygons into per-species statistics."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from services.pricing import PricingModel

SUMMARY_COLUMNS: list[str] = [
    "species",
    "count",
    "share_pct",
    "mean_crown_d_m",
    "total_crown_area_m2",
    "priced_count",
    "avg_price",
    "value",
    "has_formula",
]


def species_summary(trees: gpd.GeoDataFrame, pricing: PricingModel) -> pd.DataFrame:
    """Aggregate trees per species.

    Args:
        trees: GeoDataFrame with ``species``, ``crown_diameter_m``,
            ``crown_area_m2`` and optionally ``price`` (see
            :func:`services.valuation.price_trees`).
        pricing: Price model (to flag species without a formula).

    Returns:
        DataFrame with :data:`SUMMARY_COLUMNS`, sorted by count (descending).
        ``value`` / ``avg_price`` are NaN for species that could not be valued.
    """
    if trees.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    price = trees["price"].to_numpy(dtype=float) if "price" in trees.columns else np.full(len(trees), np.nan)
    df = pd.DataFrame(
        {
            "species": trees["species"].to_numpy(),
            "crown_d": trees["crown_diameter_m"].to_numpy(),
            "crown_area": trees["crown_area_m2"].to_numpy(),
            "price": price,
        }
    )
    grouped = (
        df.groupby("species")
        .agg(
            count=("species", "size"),
            mean_crown_d_m=("crown_d", "mean"),
            total_crown_area_m2=("crown_area", "sum"),
            priced_count=("price", "count"),
            value=("price", "sum"),
        )
        .reset_index()
    )
    grouped.loc[grouped["priced_count"] == 0, "value"] = np.nan
    grouped["avg_price"] = grouped["value"] / grouped["priced_count"].replace(0, np.nan)
    grouped["share_pct"] = grouped["count"] / grouped["count"].sum() * 100.0
    grouped["has_formula"] = [pricing.has_formula(s) for s in grouped["species"]]
    grouped = grouped.sort_values("count", ascending=False).reset_index(drop=True)
    return grouped[SUMMARY_COLUMNS]


TREE_COLUMNS: list[str] = ["index", "species", "crown_d_m", "height_m", "price", "price_text"]


def tree_detail(trees: gpd.GeoDataFrame, species_order: list[str]) -> pd.DataFrame:
    """One row per tree for the asset ledger and the report.

    Rows are grouped by species (in ``species_order``) and, inside a species,
    sorted by price (descending, unvalued last). The numbering is therefore
    identical in the Excel ledger and the Word report.

    Args:
        trees: Trees with ``species``, ``crown_diameter_m`` and optionally
            ``height_m``, ``price`` and ``price_text``.
        species_order: Species names in display order.

    Returns:
        DataFrame with :data:`TREE_COLUMNS` (``index`` starts at 1).
    """
    n = len(trees)
    nan = np.full(n, np.nan)
    df = pd.DataFrame(
        {
            "species": trees["species"].to_numpy(),
            "crown_d_m": trees["crown_diameter_m"].to_numpy(dtype=float),
            "height_m": trees["height_m"].to_numpy(dtype=float) if "height_m" in trees.columns else nan,
            "price": trees["price"].to_numpy(dtype=float) if "price" in trees.columns else nan,
            "price_text": trees["price_text"].to_numpy() if "price_text" in trees.columns else [""] * n,
        }
    )
    rank = {name: i for i, name in enumerate(species_order)}
    df["_rank"] = df["species"].map(rank).fillna(len(rank))
    df = df.sort_values(["_rank", "price"], ascending=[True, False], na_position="last", kind="stable")
    df = df.drop(columns="_rank").reset_index(drop=True)
    df.insert(0, "index", np.arange(1, n + 1))
    return df[TREE_COLUMNS]
