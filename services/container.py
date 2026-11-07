"""Dependency container shared by tools and the UI."""

from __future__ import annotations

import threading
from typing import Optional

from config import Settings
from services.dtm import PmfParams
from services.pricing import PricingModel
from services.segmentation import SegmentationBackend, build_segmenter
from services.storage import ImageStore, LayerStore


class Services:
    """Holds long-lived services (created once at start-up)."""

    def __init__(self, settings: Settings) -> None:
        """Create all services.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.pricing = PricingModel.from_json(settings.templates_dir / "price_table.json")
        self.image_store = ImageStore(
            settings.data_dir / "uploads",
            settings.plain_image_pixel_size_m,
            PmfParams(
                cell_size_m=settings.pmf_cell_size_m,
                max_window_m=settings.pmf_max_window_m,
                slope=settings.pmf_slope,
                initial_dh_m=settings.pmf_initial_dh_m,
                max_dh_m=settings.pmf_max_dh_m,
            ),
        )
        self.layer_store = LayerStore(settings.output_dir / "layers")
        self._segmenter: Optional[SegmentationBackend] = None
        self._lock = threading.Lock()

    @property
    def segmenter(self) -> SegmentationBackend:
        """The segmentation backend (created lazily)."""
        with self._lock:
            if self._segmenter is None:
                self._segmenter = build_segmenter(self.settings)
            return self._segmenter
