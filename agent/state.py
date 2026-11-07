"""Per-session state shared by the planner, executor and tools."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd


@dataclass
class ImageRecord:
    """A loaded raster image.

    Attributes:
        image_id: Unique id (used in tile URLs and tool arguments).
        name: Display name (original file name).
        path: Path of the stored GeoTIFF.
        bounds: ``(minx, miny, maxx, maxy)`` in EPSG:4326.
        dtype: Raster data type of band 1.
        band_count: Number of bands.
        rescale: ``(low, high)`` display stretch for non-uint8 data.
        dem_path: Associated DEM GeoTIFF (enables the original image + DEM valuation).
        dem_name: Display name of the DEM.
        ndsm_path: nDSM (DEM - PMF-DTM) raster used to read tree heights.
        georeferenced: ``False`` for PNG/JPG or TIFF without CRS; such images get a
            pseudo local coordinate system and are shown without a basemap.
    """

    image_id: str
    name: str
    path: Path
    bounds: tuple[float, float, float, float]
    dtype: str
    band_count: int
    rescale: Optional[tuple[float, float]] = None
    georeferenced: bool = True
    dem_path: Optional[Path] = None
    dem_name: Optional[str] = None
    ndsm_path: Optional[Path] = None


@dataclass
class LayerRecord:
    """A vector layer shown on the map.

    Attributes:
        layer_id: Unique id (used in the GeoJSON URL).
        name: Display name in the layer control.
        color: Hex stroke/fill colour.
        image_id: Image the layer was extracted from.
        species: Species shown by the layer.
    """

    layer_id: str
    name: str
    color: str
    image_id: Optional[str] = None
    species: Optional[str] = None


@dataclass
class ValuationResult:
    """Last valuation produced by the value tool.

    Attributes:
        region_name: Label of the valued region.
        table: Per-species summary table.
        total_value: Sum of all species values.
    """

    region_name: str
    table: pd.DataFrame
    total_value: float


@dataclass
class SessionState:
    """All mutable data of one browser session.

    Attributes:
        session_id: Random id for logging.
        images: Loaded images by id.
        current_image_id: Image used when a tool call omits ``image_id``.
        trees: Extracted tree polygons per image id (EPSG:4326).
        layers: Vector layers shown on the map.
        last_valuation: Result of the last valuation.
        history: ``(role, text)`` chat turns for the planner prompt.
        version: Incremented whenever the map content changes.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    images: dict[str, ImageRecord] = field(default_factory=dict)
    current_image_id: Optional[str] = None
    trees: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    layers: list[LayerRecord] = field(default_factory=list)
    last_valuation: Optional[ValuationResult] = None
    history: list[tuple[str, str]] = field(default_factory=list)
    version: int = 0

    def touch(self) -> None:
        """Mark the map content as changed."""
        self.version += 1

    def describe(self) -> str:
        """Summarise the state for the planner prompt.

        Returns:
            Multi-line Chinese description.
        """
        if not self.images:
            return "- 尚未加载任何图像"
        lines: list[str] = []
        for image in self.images.values():
            mark = "（当前）" if image.image_id == self.current_image_id else ""
            gdf = self.trees.get(image.image_id)
            if gdf is None or gdf.empty:
                extracted = "尚无提取结果"
            else:
                counts = gdf["species"].value_counts()
                extracted = "已提取: " + "、".join(f"{k}×{v}" for k, v in counts.items())
            dem = "；已关联DEM" if image.dem_path else "；未关联DEM"
            lines.append(f"- 图像 id={image.image_id} 名称={image.name}{mark}；{extracted}{dem}")
        lines.append(f"- 已有估值结果: {'是' if self.last_valuation else '否'}")
        return "\n".join(lines)
