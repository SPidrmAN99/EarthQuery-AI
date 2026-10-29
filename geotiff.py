"""
Safe rasterio wrappers used by EarthQuery.

The uploaded source file is always opened read-only. No source raster is
modified by the application.
"""

from __future__ import annotations

import numpy as np
import rasterio


class GeoTiffReadError(Exception):
    """Raised when a raster file cannot be opened or read cleanly."""


def open_dataset(path: str):
    try:
        return rasterio.open(path, "r")
    except Exception as exc:
        raise GeoTiffReadError(
            f"Could not open '{path}' as a raster file: {exc}"
        ) from exc


def read_bands(dataset) -> np.ndarray:
    try:
        return dataset.read()
    except Exception as exc:
        raise GeoTiffReadError(f"Could not read raster bands: {exc}") from exc
