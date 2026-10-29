"""
Raster preview renderer.

The source raster is never modified. A separate PNG is generated for the
browser and VLM. Single-band rasters are rendered as grayscale; RGB/
multiband rasters use their first three bands. Non-display-ready values
are percentile-stretched for visibility.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

MAX_PREVIEW_DIMENSION = 2048


def _percentile_stretch(
    band: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)

    lo, hi = np.percentile(valid, [low, high])
    if hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
        if hi <= lo:
            hi = lo + 1.0

    clean = np.nan_to_num(band, nan=lo, posinf=hi, neginf=lo).astype(np.float64)
    stretched = np.clip((clean - lo) / (hi - lo), 0.0, 1.0)
    return np.round(stretched * 255).astype(np.uint8)


def render_preview_png(
    bands: np.ndarray,
    nodata: float | None = None,
    filename: str | None = None,
) -> tuple[bytes, str]:
    """
    Convert raster bands to a model/browser-safe PNG.

    Single-band rasters become grayscale RGB PNGs. Three or more bands use
    the first three bands as RGB. Values outside uint8/display-ready form
    are percentile-stretched.
    """
    if bands.ndim != 3 or bands.shape[0] == 0:
        raise ValueError("Expected a (bands, height, width) array.")

    working = bands.astype(np.float64)
    if nodata is not None:
        working = np.where(working == nodata, np.nan, working)

    band_count = working.shape[0]

    if band_count == 1:
        gray = _percentile_stretch(working[0])
        stacked = np.stack([gray, gray, gray], axis=-1)
    else:
        source = working[:3]
        channels = [_percentile_stretch(source[i]) for i in range(source.shape[0])]
        while len(channels) < 3:
            channels.append(channels[-1].copy())
        stacked = np.stack(channels[:3], axis=-1)

    image = Image.fromarray(stacked, mode="RGB")

    if max(image.size) > MAX_PREVIEW_DIMENSION:
        image.thumbnail((MAX_PREVIEW_DIMENSION, MAX_PREVIEW_DIMENSION), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    png_bytes = buffer.getvalue()
    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    return png_bytes, data_url
