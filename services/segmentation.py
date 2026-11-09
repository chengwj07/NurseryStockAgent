"""Segmentation backends.

Both backends follow the same counting logic: the text-prompt masks of **all**
windows of one species are merged into a single raster; small holes are filled,
small debris is removed and connected components are counted (see
:mod:`services.postprocess`). Merging by raster union also removes the
duplicates created by overlapping windows.
"""

from __future__ import annotations

import logging
import tempfile
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.affinity import affine_transform
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from config import Settings
from services.postprocess import CleanParams, CleanStats, mask_to_trees

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Output of a backend.

    Attributes:
        trees: One polygon per counted tree (EPSG:4326) with
            ``crown_area_m2`` and ``crown_diameter_m``.
        stats: Cleaning statistics (raw components, holes, debris, final count).
        raw_instances: Number of SAM3 instances before merging.
    """

    trees: gpd.GeoDataFrame
    stats: CleanStats
    raw_instances: int


def _finalize(geoms: list[BaseGeometry], areas: list[float], crs: CRS) -> gpd.GeoDataFrame:
    """Build the standard result GeoDataFrame in EPSG:4326.

    Args:
        geoms: Component polygons in ``crs``.
        areas: Ground area of each component in square metres.
        crs: CRS of ``geoms``.

    Returns:
        GeoDataFrame with ``crown_area_m2`` and ``crown_diameter_m``.
    """
    if not geoms:
        return gpd.GeoDataFrame(
            {"crown_area_m2": [], "crown_diameter_m": []}, geometry=[], crs="EPSG:4326"
        )
    area = np.asarray(areas, dtype=float)
    gdf = gpd.GeoDataFrame(
        {"crown_area_m2": area, "crown_diameter_m": 2.0 * np.sqrt(area / np.pi)},
        geometry=geoms,
        crs=crs,
    )
    return gdf.to_crs(4326)


def _clean_params(settings: Settings) -> CleanParams:
    """Extract the cleaning thresholds from the settings."""
    return CleanParams(
        min_object_area_m2=settings.min_object_area_m2,
        max_hole_area_m2=settings.max_hole_area_m2,
        max_pixels=settings.mask_max_pixels,
    )


class SegmentationBackend(ABC):
    """Interface of every segmentation backend."""

    @abstractmethod
    def segment(
        self, image_path: Path, prompt: str, rescale: Optional[tuple[float, float]] = None
    ) -> SegmentationResult:
        """Segment and count all instances matching a text prompt.

        Args:
            image_path: GeoTIFF path.
            prompt: Text prompt, e.g. ``"buddhist pine"``.
            rescale: Display stretch for non-uint8 rasters.

        Returns:
            The cleaned, counted result.
        """


class MockBackend(SegmentationBackend):
    """Random circular crowns run through the real counting pipeline (no GPU)."""

    def __init__(self, settings: Settings) -> None:
        """Initialise.

        Args:
            settings: Application settings (cleaning thresholds).
        """
        self._params = _clean_params(settings)

    def segment(
        self, image_path: Path, prompt: str, rescale: Optional[tuple[float, float]] = None
    ) -> SegmentationResult:
        """See :meth:`SegmentationBackend.segment`."""
        rng = np.random.default_rng(zlib.crc32(prompt.encode("utf-8")))
        with rasterio.open(image_path) as src:
            tf, width, height, crs = src.transform, src.width, src.height, src.crs
        coeffs = [tf.a, tf.b, tf.d, tf.e, tf.c, tf.f]
        circles: list[BaseGeometry] = []
        for _ in range(int(rng.integers(40, 120))):
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            radius = rng.uniform(0.008, 0.02) * width
            circles.append(affine_transform(Point(cx, cy).buffer(radius, 8), coeffs))
        mask = rasterize([(g, 1) for g in circles], out_shape=(height, width), transform=tf, dtype="uint8")
        polygons, areas, stats = mask_to_trees(mask, tf, crs, self._params)
        return SegmentationResult(_finalize(polygons, areas, crs), stats, len(circles))


class Sam3Backend(SegmentationBackend):
    """Sliding-window SAM3 text-prompt segmentation for large GeoTIFFs.

    Note:
        The SAM3 calls in :meth:`_load` / :meth:`_infer` follow the official
        repository interface. Replace them with your own inference pipeline if
        it differs.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise (the model is loaded lazily on first use).

        Args:
            settings: Application settings.
        """
        self._checkpoint = settings.sam3_checkpoint
        self._size = settings.tile_size
        self._overlap = settings.tile_overlap
        self._min_score = settings.min_score
        self._params = _clean_params(settings)
        self._tmp_dir = settings.data_dir / "tmp"
        self._processor: object = None

    def _load(self) -> None:
        """Load SAM3 once."""
        if self._processor is not None:
            return
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
        from sam3.model_builder import build_sam3_image_model  # type: ignore

        kwargs = {"checkpoint_path": self._checkpoint} if self._checkpoint else {}
        logger.info("Loading SAM3 model...")
        self._processor = Sam3Processor(build_sam3_image_model(**kwargs))

    def _infer(self, tile: np.ndarray, prompt: str) -> list[tuple[np.ndarray, float]]:
        """Run SAM3 on one RGB tile.

        Args:
            tile: ``(H, W, 3)`` uint8 array.
            prompt: Text prompt.

        Returns:
            List of ``(bool mask (H, W), score)``.
        """
        import torch
        from PIL import Image

        processor = self._processor
        # The official SAM3 mixes bfloat16 and float32 weights/activations; the
        # reference notebooks run everything under bf16 autocast. Without it
        # you get "mat1 and mat2 must have the same dtype (BFloat16 and Float)".
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(tile))  # type: ignore[attr-defined]
            out = processor.set_text_prompt(state=state, prompt=prompt)  # type: ignore[attr-defined]

        results: list[tuple[np.ndarray, float]] = []
        for mask, score in zip(out["masks"], out["scores"]):
            score_f = float(score)
            if score_f < self._min_score:
                continue
            m = mask.squeeze().detach()
            if m.dtype != torch.bool:  # bf16 tensors cannot go to numpy directly
                m = m.float() > 0.5
            results.append((m.cpu().numpy(), score_f))
        return results

    @staticmethod
    def _starts(length: int, size: int, step: int) -> list[int]:
        """Window start offsets along one axis (last window aligned to the edge)."""
        if length <= size:
            return [0]
        starts = list(range(0, length - size, step))
        starts.append(length - size)
        return starts

    def _windows(self, width: int, height: int) -> Iterator[Window]:
        """Yield sliding windows covering the raster."""
        step = max(self._size - self._overlap, 1)
        for row in self._starts(height, self._size, step):
            for col in self._starts(width, self._size, step):
                yield Window(col, row, min(self._size, width - col), min(self._size, height - row))

    @staticmethod
    def _read_rgb(
        src: rasterio.DatasetReader, window: Window, rescale: Optional[tuple[float, float]]
    ) -> np.ndarray:
        """Read a window as ``(H, W, 3)`` uint8."""
        if src.count >= 3:
            arr = src.read([1, 2, 3], window=window)
        else:
            arr = np.repeat(src.read([1], window=window), 3, axis=0)
        arr = np.moveaxis(arr, 0, -1)
        if arr.dtype != np.uint8:
            low, high = rescale or (float(arr.min()), float(arr.max()))
            arr = (np.clip((arr.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _fit_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Resize a mask to ``shape`` (nearest) if SAM3 returned another size."""
        if mask.shape == shape:
            return mask
        from PIL import Image

        resized = Image.fromarray(mask.astype(np.uint8)).resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(resized) > 0

    def segment(
        self, image_path: Path, prompt: str, rescale: Optional[tuple[float, float]] = None
    ) -> SegmentationResult:
        """Segment all windows, merge masks on disk, then clean and count trees."""
        self._load()
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        raw_instances = 0

        with rasterio.open(image_path) as src, tempfile.TemporaryDirectory(dir=self._tmp_dir) as tmp:
            height, width = src.height, src.width
            merged = np.memmap(
                Path(tmp) / "merged.u8",
                dtype=np.uint8,
                mode="w+",
                shape=(height, width),
            )
            try:
                windows = list(self._windows(width, height))
                logger.info(
                    "SAM3 on %s with prompt %r: %d windows",
                    image_path.name, prompt, len(windows),
                )
                for idx, window in enumerate(windows, 1):
                    tile = self._read_rgb(src, window, rescale)
                    win_mask = np.zeros(
                        (int(window.height), int(window.width)), dtype=bool
                    )
                    for mask, _score in self._infer(tile, prompt):
                        win_mask |= self._fit_mask(mask, win_mask.shape)
                        raw_instances += 1

                    r0, c0 = int(window.row_off), int(window.col_off)
                    region = merged[
                        r0 : r0 + win_mask.shape[0],
                        c0 : c0 + win_mask.shape[1],
                    ]
                    try:
                        np.bitwise_or(region, win_mask.view(np.uint8), out=region)
                    finally:
                        # A memmap slice keeps the file mapping alive on Windows.
                        del region

                    logger.debug(
                        "window %d/%d done, %d instances so far",
                        idx, len(windows), raw_instances,
                    )

                merged.flush()
                polygons, areas, stats = mask_to_trees(
                    merged, src.transform, src.crs, self._params
                )
                crs = src.crs
            finally:
                # Close the mapping before TemporaryDirectory removes merged.u8.
                # No views of merged may be used after this point.
                mapping = merged._mmap
                if mapping is not None and not mapping.closed:
                    mapping.close()
                del merged

        logger.info(
            "SAM3 finished: %d instances -> %d trees",
            raw_instances, stats.final_count,
        )
        return SegmentationResult(_finalize(polygons, areas, crs), stats, raw_instances)


def build_segmenter(settings: Settings) -> SegmentationBackend:
    """Create the configured segmentation backend.

    Args:
        settings: Application settings.

    Returns:
        A :class:`SegmentationBackend` instance.
    """
    if settings.seg_backend == "mock":
        logger.warning("Using MockBackend: results are random!")
        return MockBackend(settings)
    return Sam3Backend(settings)
