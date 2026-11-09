"""Global stores for uploaded images and generated vector layers."""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from agent.state import ImageRecord, LayerRecord
from services.dtm import PmfParams, build_ndsm

logger = logging.getLogger(__name__)

PALETTE: list[str] = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
]


class ImageStore:
    """Registry of uploaded images (shared by tile server and tools).

    GeoTIFFs with a CRS are copied as is. Anything else (PNG/JPG/BMP/WebP, or a
    TIFF without CRS) is converted into a tiled GeoTIFF in a pseudo projected
    system (EPSG:3857, origin at 0,0, ``pixel_size_m`` per pixel) so that the
    tile server, segmentation and statistics work unchanged.
    """

    PLAIN_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})

    def __init__(
        self, upload_dir: Path, pixel_size_m: float = 1.0, pmf: PmfParams | None = None
    ) -> None:
        """Initialise the store.

        Args:
            upload_dir: Directory where uploaded files are stored.
            pixel_size_m: Pixel size assumed for non-georeferenced images.
            pmf: Parameters of the DEM -> DTM filter.
        """
        self._pmf = pmf or PmfParams()
        self._dir = upload_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pixel = pixel_size_m
        self._items: dict[str, ImageRecord] = {}
        self._lock = threading.Lock()

    def register(self, src_path: Path, display_name: Optional[str] = None) -> ImageRecord:
        """Store an image and read its metadata.

        Args:
            src_path: Path of the uploaded file.
            display_name: Optional display name (defaults to the file name).

        Returns:
            The new :class:`ImageRecord`.

        Raises:
            ValueError: If the file type is not supported.
        """
        image_id = uuid.uuid4().hex[:8]
        suffix = src_path.suffix.lower()
        if suffix in self.PLAIN_SUFFIXES:
            dst = self._dir / f"{image_id}_{src_path.stem}.tif"
            self._convert_plain(src_path, dst)
            georeferenced = False
        elif suffix in {".tif", ".tiff"}:
            with rasterio.open(src_path) as probe:
                georeferenced = probe.crs is not None
            if georeferenced:
                dst = self._dir / f"{image_id}_{src_path.name}"
                shutil.copy2(src_path, dst)
            else:
                dst = self._dir / f"{image_id}_{src_path.stem}.tif"
                self._convert_untagged_tiff(src_path, dst)
        else:
            raise ValueError(f"Unsupported image format {suffix or src_path.name} (supported: tif/tiff/png/jpg/bmp/webp)")

        self._ensure_overviews(dst)
        with rasterio.open(dst) as ds:
            bounds = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
            dtype = ds.dtypes[0]
            rescale = None if dtype == "uint8" else self._estimate_rescale(ds)
            record = ImageRecord(
                image_id=image_id,
                name=display_name or src_path.name,
                path=dst,
                bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
                dtype=dtype,
                band_count=ds.count,
                rescale=rescale,
                georeferenced=georeferenced,
            )
        with self._lock:
            self._items[image_id] = record
        logger.info(
            "Registered image %s (%s, %s, %d bands, georeferenced=%s)",
            image_id, record.name, dtype, record.band_count, georeferenced,
        )
        return record

    def attach_dem(self, record: ImageRecord, dem_src: Path) -> ImageRecord:
        """Associate a DEM with an image (enables the DEM valuation mode).

        The DTM is derived from the DEM with PMF and the nDSM (DEM - DTM) is stored.

        Args:
            record: The original image.
            dem_src: Uploaded DEM file.

        Returns:
            The updated record.

        Raises:
            ValueError: If the DEM cannot be used with this image.
        """
        if not record.georeferenced:
            raise ValueError("This image has no geographic coordinates, so a DEM cannot be attached")
        if dem_src.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError("The DEM must be a GeoTIFF (.tif / .tiff)")
        with rasterio.open(dem_src) as ds:
            if ds.crs is None:
                raise ValueError("The DEM has no coordinate system")
            b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        x0, y0, x1, y1 = record.bounds
        if b[0] >= x1 or b[2] <= x0 or b[1] >= y1 or b[3] <= y0:
            raise ValueError("The DEM does not overlap the current image")
        dst = self._dir / f"{record.image_id}_dem_{dem_src.name}"
        ndsm = self._dir / f"{record.image_id}_ndsm.tif"
        shutil.copy2(dem_src, dst)
        try:
            build_ndsm(dst, ndsm, self._pmf)
        except Exception as exc:  # noqa: BLE001
            dst.unlink(missing_ok=True)
            ndsm.unlink(missing_ok=True)
            logger.exception("DEM -> DTM failed")
            raise ValueError(f"Failed to derive the DTM from the DEM: {exc}") from exc
        for old in (record.dem_path, record.ndsm_path):
            if old and old not in (dst, ndsm):
                old.unlink(missing_ok=True)
        record.dem_path, record.dem_name, record.ndsm_path = dst, dem_src.name, ndsm
        logger.info("Attached DEM %s to image %s", dem_src.name, record.image_id)
        return record

    @staticmethod
    def _ensure_overviews(path: Path) -> None:
        """Build internal overviews for large rasters so low-zoom tiles are fast.

        Without overviews every low-zoom tile has to read the full-resolution
        data, which makes big images appear to never load.

        Args:
            path: Stored GeoTIFF (modified in place; it is our own copy).
        """
        try:
            with rasterio.open(path, "r+") as ds:
                longest = max(ds.width, ds.height)
                if ds.overviews(1) or longest < 2048:
                    return
                levels, factor = [], 2
                while longest // factor >= 256:
                    levels.append(factor)
                    factor *= 2
                logger.info("Building overviews %s for %s ...", levels, path.name)
                ds.build_overviews(levels, Resampling.average)
        except Exception:  # noqa: BLE001 - overviews are an optimisation only
            logger.warning("Could not build overviews for %s", path.name, exc_info=True)

    def _profile(self, width: int, height: int, count: int, dtype: str) -> dict[str, object]:
        """GeoTIFF profile with the pseudo georeferencing."""
        return {
            "driver": "GTiff", "width": width, "height": height, "count": count, "dtype": dtype,
            "crs": "EPSG:3857",
            "transform": from_origin(0.0, height * self._pixel, self._pixel, self._pixel),
            "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate",
        }

    def _convert_plain(self, src_path: Path, dst: Path) -> None:
        """Convert a PNG/JPG/... into a pseudo-georeferenced GeoTIFF.

        Args:
            src_path: Source image.
            dst: Target GeoTIFF path.
        """
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None  # remote sensing images can be huge
        with Image.open(src_path) as img:
            arr = np.asarray(img.convert("RGB"))
        height, width, _ = arr.shape
        with rasterio.open(dst, "w", **self._profile(width, height, 3, "uint8")) as out:
            out.write(np.moveaxis(arr, 2, 0))

    def _convert_untagged_tiff(self, src_path: Path, dst: Path) -> None:
        """Re-write a TIFF without CRS with the pseudo georeferencing.

        Args:
            src_path: Source TIFF.
            dst: Target GeoTIFF path.
        """
        with rasterio.open(src_path) as src:
            profile = self._profile(src.width, src.height, src.count, src.dtypes[0])
            with rasterio.open(dst, "w", **profile) as out:
                for _, window in out.block_windows(1):
                    out.write(src.read(window=window), window=window)

    @staticmethod
    def _estimate_rescale(ds: rasterio.DatasetReader) -> tuple[float, float]:
        """Estimate a 2-98 percentile display stretch from a decimated read.

        Args:
            ds: Open dataset.

        Returns:
            ``(low, high)`` stretch values.
        """
        indexes = [1, 2, 3] if ds.count >= 3 else [1]
        shape = (len(indexes), min(ds.height, 1024), min(ds.width, 1024))
        arr = ds.read(indexes, out_shape=shape, resampling=Resampling.nearest, masked=True)
        values = arr.compressed()
        if values.size == 0:
            return 0.0, 255.0
        low, high = np.percentile(values, [2, 98])
        return float(low), float(max(high, low + 1e-6))

    def get(self, image_id: str) -> Optional[ImageRecord]:
        """Look up an image.

        Args:
            image_id: Image id.

        Returns:
            The record or ``None``.
        """
        with self._lock:
            return self._items.get(image_id)

    def remove(self, image_id: str) -> None:
        """Remove an image from the registry (the file is deleted too).

        Args:
            image_id: Image id.
        """
        with self._lock:
            record = self._items.pop(image_id, None)
        if record is not None:
            record.path.unlink(missing_ok=True)
            for extra in (record.dem_path, record.ndsm_path):
                if extra:
                    extra.unlink(missing_ok=True)
            logger.info("Removed image %s", image_id)


class LayerStore:
    """Writes vector layers to GeoJSON files served by the API."""

    def __init__(self, layer_dir: Path) -> None:
        """Initialise the store.

        Args:
            layer_dir: Directory for GeoJSON files.
        """
        self._dir = layer_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._lock = threading.Lock()

    def save(self, gdf: gpd.GeoDataFrame, name: str) -> LayerRecord:
        """Persist a GeoDataFrame as a GeoJSON layer.

        Args:
            gdf: Features in EPSG:4326.
            name: Display name.

        Returns:
            The new :class:`LayerRecord`.
        """
        layer_id = uuid.uuid4().hex[:10]
        with self._lock:
            color = PALETTE[self._counter % len(PALETTE)]
            self._counter += 1
        self.write(layer_id, gdf)
        logger.info("Saved layer %s (%s, %d features)", layer_id, name, len(gdf))
        return LayerRecord(layer_id=layer_id, name=name, color=color)

    #: Columns written to GeoJSON (``price_text`` feeds the map tooltip).
    EXPORT_COLUMNS: tuple[str, ...] = ("species", "crown_area_m2", "crown_diameter_m", "price_text")

    def write(self, layer_id: str, gdf: gpd.GeoDataFrame) -> None:
        """(Re)write the GeoJSON file of a layer.

        Args:
            layer_id: Layer id.
            gdf: Features in EPSG:4326.
        """
        keep = [c for c in self.EXPORT_COLUMNS if c in gdf.columns]
        gdf[keep + ["geometry"]].to_file(self._dir / f"{layer_id}.geojson", driver="GeoJSON")

    def path_of(self, layer_id: str) -> Optional[Path]:
        """Return the file of a layer.

        Args:
            layer_id: Layer id (must be alphanumeric).

        Returns:
            Path to the GeoJSON or ``None`` if unknown.
        """
        if not layer_id.isalnum():
            return None
        path = self._dir / f"{layer_id}.geojson"
        return path if path.exists() else None
