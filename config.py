"""Application configuration loaded from environment variables / ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Global settings. Every field can be overridden by ``NURSERY_<FIELD>``.

    Attributes:
        host: Bind host of the web server.
        port: Bind port of the web server.
        public_base_url: Only needed behind a proxy/sub-path. Empty (default)
            means the map iframe uses URLs relative to the page, which works for
            localhost, LAN and remote access alike.
        map_init_lat: Latitude of the initial map view (before any image).
        map_init_lon: Longitude of the initial map view.
        map_init_zoom: Zoom of the initial map view (7 ~ province extent).
        data_dir: Directory for uploaded images.
        output_dir: Directory for generated layers, tables and reports.
        templates_dir: Directory with price table and table templates.
        log_level: Python logging level name.
        llm_mode: ``"api"`` (OpenAI-compatible endpoint) or ``"llamacpp"``.
        seg_backend: ``"sam3"`` or ``"mock"`` (no GPU, random crowns).
    """

    model_config = SettingsConfigDict(
        env_prefix="NURSERY_", env_file=BASE_DIR / ".env", extra="ignore"
    )

    host: str = "0.0.0.0"
    port: int = 7860
    public_base_url: str = ""
    map_init_lat: float = 29.2   # initial map view (province scale, Zhejiang)
    map_init_lon: float = 120.1
    map_init_zoom: float = 7.0
    data_dir: Path = BASE_DIR / "data"
    output_dir: Path = BASE_DIR / "outputs"
    templates_dir: Path = BASE_DIR / "templates"
    log_level: str = "INFO"

    # --- LLM ---
    llm_mode: Literal["api", "llamacpp"] = "api"
    llamacpp_model_path: str = ""
    llamacpp_n_ctx: int = 8192
    llamacpp_n_gpu_layers: int = -1
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    api_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # --- Segmentation ---
    seg_backend: Literal["sam3", "mock"] = "sam3"
    sam3_checkpoint: str = ""
    tile_size: int = 1024
    tile_overlap: int = 128
    min_score: float = 0.5
    # PMF (DEM -> DTM) for tree height
    pmf_cell_size_m: float = 1.0
    pmf_max_window_m: float = 20.0
    pmf_slope: float = 0.15
    pmf_initial_dh_m: float = 0.15
    pmf_max_dh_m: float = 3.0
    # Counting: merged-raster cleaning thresholds (ground area, m^2)
    min_object_area_m2: float = 1.0   # components smaller than this are debris
    max_hole_area_m2: float = 1.0     # enclosed holes up to this size are filled
    mask_max_pixels: int = 250_000_000  # larger masks are block-reduced for counting
    plain_image_pixel_size_m: float = 1.0  # pixel size assumed for PNG/JPG/untagged TIFF

    # --- Agent ---
    history_turns: int = 6
    max_replans: int = 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton.

    Returns:
        The application settings.
    """
    return Settings()
