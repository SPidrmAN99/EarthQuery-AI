"""
Deterministic raster metadata and analysis-context extraction.

The LLM is never asked to infer basic raster/GIS facts that can be
obtained directly from the source file. This module collects them first.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover
    Transformer = None


_ACQUISITION_TAG_CANDIDATES = (
    "TIFFTAG_DATETIME",
    "ACQUISITIONDATETIME",
    "ACQUISITION_DATE",
    "ACQUISITIONDATE",
    "DATE",
    "DATETIME",
)

_SENSOR_TAG_CANDIDATES = (
    "PLATFORM",
    "SATELLITE",
    "SENSOR",
    "INSTRUMENT",
)


def _first_present(tags: dict, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        value = tags.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _parse_filename(filename: str) -> dict[str, Any]:
    """
    Parse common Sentinel-1 naming fields without claiming that every
    filename follows the same convention.
    """
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit("\\", 1)[-1]
    stem = re.sub(r"\.(tif|tiff|png|jpe?g)$", "", stem, flags=re.I)

    result: dict[str, Any] = {
        "source_filename": filename,
        "mission_family": None,
        "platform": None,
        "acquisition_datetime_utc": None,
        "acquisition_date": None,
        "acquisition_time_utc": None,
        "acquisition_timestamp_source": None,
        "acquisition_mode": None,
        "product_type": None,
        "product_level": None,
        "resolution_class": None,
        "polarization": None,
        "naming_convention_match": None,
    }

    # Sentinel-1 SAFE-style abbreviated product pattern:
    # S1A_IW_GRDH_1SDV_20170613T165043_...
    m = re.search(
        r"^(S1[AB])_([A-Z]{2})_([A-Z]{4})_([0-9A-Z]{4})_"
        r"(\d{8}T\d{6})_.*?_([A-Z]{2})$",
        stem,
        flags=re.I,
    )
    if m:
        platform, mode, product, levelpol, stamp, pol = m.groups()
        result.update(
            {
                "mission_family": "Sentinel-1",
                "platform": platform.upper(),
                "acquisition_mode": mode.upper(),
                "product_type": product.upper(),
                "product_level": levelpol[:2].upper(),
                "resolution_class": (
                    "High Resolution" if "H" in product.upper() else None
                ),
                "polarization": pol.upper(),
                "naming_convention_match": "Sentinel-1 abbreviated product",
            }
        )
        try:
            dt = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            result["acquisition_datetime_utc"] = dt.isoformat().replace("+00:00", "Z")
            result["acquisition_date"] = dt.date().isoformat()
            result["acquisition_time_utc"] = dt.time().isoformat()
            result["acquisition_timestamp_source"] = "filename"
        except ValueError:
            pass
        return result

    # Generic timestamp fallback.
    stamp_match = re.search(r"(\d{8}T\d{6})", stem)
    if stamp_match:
        try:
            dt = datetime.strptime(
                stamp_match.group(1), "%Y%m%dT%H%M%S"
            ).replace(tzinfo=timezone.utc)
            result["acquisition_datetime_utc"] = dt.isoformat().replace("+00:00", "Z")
            result["acquisition_date"] = dt.date().isoformat()
            result["acquisition_time_utc"] = dt.time().isoformat()
            result["acquisition_timestamp_source"] = "filename"
        except ValueError:
            pass

    return result


def extract_metadata(dataset, filename: str, file_size_bytes: int) -> dict[str, Any]:
    """Return defensively extracted metadata from the rasterio dataset."""
    try:
        dataset_tags = dict(dataset.tags())
    except Exception:
        dataset_tags = {}

    band_tags: list[dict] = []
    band_descriptions: list[str | None] = []
    for i in range(1, dataset.count + 1):
        try:
            band_tags.append(dict(dataset.tags(i)))
        except Exception:
            band_tags.append({})
        try:
            band_descriptions.append(dataset.descriptions[i - 1])
        except Exception:
            band_descriptions.append(None)

    crs_str = None
    epsg = None
    crs_wkt = None
    try:
        if dataset.crs:
            crs_str = dataset.crs.to_string()
            epsg = dataset.crs.to_epsg()
            crs_wkt = dataset.crs.to_wkt()
    except Exception:
        pass

    transform = None
    pixel_resolution = None
    try:
        if dataset.transform and not dataset.transform.is_identity:
            t = dataset.transform
            transform = [t.a, t.b, t.c, t.d, t.e, t.f]
            pixel_resolution = {"x": abs(float(t.a)), "y": abs(float(t.e))}
    except Exception:
        pass

    bounds = None
    try:
        b = dataset.bounds
        bounds = {
            "left": float(b.left),
            "bottom": float(b.bottom),
            "right": float(b.right),
            "top": float(b.top),
        }
    except Exception:
        pass

    color_interp = None
    try:
        color_interp = [str(ci) for ci in dataset.colorinterp]
    except Exception:
        pass

    file_name_info = _parse_filename(filename)

    return {
        "filename": filename,
        "file_size_bytes": file_size_bytes,
        "driver": getattr(dataset, "driver", None),
        "width": int(dataset.width),
        "height": int(dataset.height),
        "band_count": int(dataset.count),
        "dtype": str(dataset.dtypes[0]) if dataset.dtypes else None,
        "is_georeferenced": crs_str is not None,
        "crs": crs_str,
        "epsg": epsg,
        "crs_wkt": crs_wkt,
        "transform": transform,
        "pixel_resolution": pixel_resolution,
        "bounds": bounds,
        "nodata": _json_safe(dataset.nodata),
        "band_descriptions": band_descriptions,
        "color_interpretation": color_interp,
        "dataset_tags": _json_safe(dataset_tags),
        "band_tags": _json_safe(band_tags),
        "acquisition_datetime": _first_present(dataset_tags, _ACQUISITION_TAG_CANDIDATES)
        or file_name_info.get("acquisition_datetime_utc"),
        "sensor_platform": _first_present(dataset_tags, _SENSOR_TAG_CANDIDATES)
        or file_name_info.get("platform"),
        "filename_product_info": file_name_info,
    }


def _band_statistics(bands: np.ndarray, nodata: float | None) -> list[dict[str, Any]]:
    result = []
    for idx in range(bands.shape[0]):
        band = np.asarray(bands[idx], dtype=np.float64)
        valid = np.isfinite(band)
        if nodata is not None:
            valid &= band != nodata
        values = band[valid]
        if values.size == 0:
            result.append(
                {
                    "band": idx + 1,
                    "valid_pixels": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                    "p02": None,
                    "p50": None,
                    "p98": None,
                }
            )
            continue

        p02, p50, p98 = np.percentile(values, [2, 50, 98])
        result.append(
            {
                "band": idx + 1,
                "valid_pixels": int(values.size),
                "min": _safe_float(values.min()),
                "max": _safe_float(values.max()),
                "mean": _safe_float(values.mean()),
                "std": _safe_float(values.std()),
                "p02": _safe_float(p02),
                "p50": _safe_float(p50),
                "p98": _safe_float(p98),
            }
        )
    return result


def _geospatial_context(metadata: dict[str, Any]) -> dict[str, Any]:
    bounds = metadata.get("bounds")
    crs = metadata.get("crs")
    if not bounds:
        return {
            "area_m2": None,
            "extent_native_units": None,
            "center_wgs84": None,
            "bounds_wgs84": None,
            "coordinate_conversion": None,
        }

    width_native = abs(bounds["right"] - bounds["left"])
    height_native = abs(bounds["top"] - bounds["bottom"])

    context = {
        "area_m2": _safe_float(width_native * height_native),
        "extent_native_units": {
            "width": _safe_float(width_native),
            "height": _safe_float(height_native),
        },
        "center_wgs84": None,
        "bounds_wgs84": None,
        "coordinate_conversion": None,
    }

    if not crs or Transformer is None:
        context["coordinate_conversion"] = "Unavailable"
        return context

    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        cx = (bounds["left"] + bounds["right"]) / 2
        cy = (bounds["bottom"] + bounds["top"]) / 2
        lon, lat = transformer.transform(cx, cy)
        context["center_wgs84"] = {
            "latitude": _safe_float(lat),
            "longitude": _safe_float(lon),
            "crs": "EPSG:4326",
        }

        corners = [
            (bounds["left"], bounds["bottom"]),
            (bounds["left"], bounds["top"]),
            (bounds["right"], bounds["bottom"]),
            (bounds["right"], bounds["top"]),
        ]
        converted = []
        for x, y in corners:
            lon_c, lat_c = transformer.transform(x, y)
            converted.append(
                {
                    "latitude": _safe_float(lat_c),
                    "longitude": _safe_float(lon_c),
                }
            )
        context["bounds_wgs84"] = converted
        context["coordinate_conversion"] = f"{crs} → EPSG:4326 using pyproj"
    except Exception as exc:
        context["coordinate_conversion"] = f"Failed: {exc}"

    return context



def _extract_band_wavelengths(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract wavelength-like values when explicitly present in raster tags."""
    candidates = (
        "WAVELENGTH", "WAVELENGTH_NM", "CENTER_WAVELENGTH", "CENTRAL_WAVELENGTH",
        "CENTRAL_WAVELENGTH_NM", "BAND_WAVELENGTH", "WAVELENGTH_UM", "CENTER_WAVELENGTH_NM",
    )
    result = []
    for idx, tags in enumerate(metadata.get("band_tags") or [], start=1):
        value = None
        key = None
        for candidate in candidates:
            if candidate in tags and str(tags[candidate]).strip():
                value = tags[candidate]
                key = candidate
                break
        result.append({"band": idx, "wavelength": _json_safe(value), "source_tag": key})
    return result


def _spectral_context(bands: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    """Deterministic radiometric/spectral profile; never invents wavelength semantics."""
    stats = _band_statistics(bands, metadata.get("nodata"))
    wavelengths = _extract_band_wavelengths(metadata)
    profiles = []
    for s, w in zip(stats, wavelengths):
        profiles.append({
            "band": s["band"],
            "wavelength": w.get("wavelength"),
            "wavelength_source": w.get("source_tag"),
            "min": s["min"],
            "max": s["max"],
            "mean": s["mean"],
            "std": s["std"],
            "p02": s["p02"],
            "p50": s["p50"],
            "p98": s["p98"],
            "dynamic_range": (
                _safe_float(s["max"] - s["min"])
                if s["max"] is not None and s["min"] is not None else None
            ),
        })
    return {
        "analysis_type": "deterministic_bandwise_radiometric_profile",
        "band_profiles": profiles,
        "wavelengths": wavelengths,
        "note": "Wavelengths are reported only when present in source metadata; numerical band statistics are computed from source raster values after excluding NoData/non-finite values.",
    }

def build_analysis_context(
    metadata: dict[str, Any],
    bands: np.ndarray,
    filename: str,
    renderer_notes: str = "",
) -> dict[str, Any]:
    """Build the deterministic technical context supplied to the VLM."""
    product = metadata.get("filename_product_info") or {}
    band_count = int(bands.shape[0])

    is_sentinel1 = product.get("mission_family") == "Sentinel-1"
    polarization = product.get("polarization")

    if is_sentinel1:
        image_type = "Sentinel-1 SAR raster"
        modality = "Synthetic Aperture Radar (SAR)"
        if band_count == 1 and polarization:
            representation = (
                f"single-band {polarization}-polarized SAR raster; "
                "stored values represent radar-derived image intensity/backscatter "
                "according to the source product preprocessing"
            )
        else:
            representation = "SAR raster"
    elif band_count == 1:
        image_type = "single-band raster"
        modality = "raster intensity/scalar field"
        representation = "single-band raster visualization"
    else:
        image_type = "multiband raster"
        modality = "multispectral/colour raster (exact semantics depend on band metadata)"
        representation = "multiband raster visualization"

    context = {
        "source": {
            "filename": filename,
            "input_type": metadata.get("driver"),
            "file_size_bytes": metadata.get("file_size_bytes"),
        },
        "image_type": image_type,
        "modality": modality,
        "representation": representation,
        "product": product,
        "raster": {
            "dimensions": {
                "width_px": metadata.get("width"),
                "height_px": metadata.get("height"),
                "total_pixels": (
                    metadata.get("width", 0) * metadata.get("height", 0)
                    if metadata.get("width") is not None and metadata.get("height") is not None
                    else None
                ),
            },
            "band_count": metadata.get("band_count"),
            "dtype": metadata.get("dtype"),
            "pixel_resolution": metadata.get("pixel_resolution"),
            "nodata": metadata.get("nodata"),
            "band_descriptions": metadata.get("band_descriptions"),
            "color_interpretation": metadata.get("color_interpretation"),
            "statistics": _band_statistics(bands, metadata.get("nodata")),
            "spectral_analysis": _spectral_context(bands, metadata),
        },
        "geospatial": _geospatial_context(metadata),
        "crs": {
            "name": metadata.get("crs"),
            "epsg": metadata.get("epsg"),
            "transform": metadata.get("transform"),
            "bounds": metadata.get("bounds"),
        },
        "tags": {
            "dataset": metadata.get("dataset_tags"),
            "bands": metadata.get("band_tags"),
        },
        "interpretation_guidance": {
            "visual_preview": "The attached image is a rendered preview of the source raster.",
            "renderer": "The preview uses percentile contrast stretching for non-display-ready raster data; this changes visual brightness but not the source values.",
            "sar_note": (
                "For Sentinel-1 SAR, brightness is radar backscatter response, not RGB colour. "
                "Speckle is a normal property of coherent SAR imaging."
                if is_sentinel1
                else None
            ),
            "polarization": polarization,
        },
    }

    # Keep the complete deterministic metadata available to the model.
    context["source_metadata"] = metadata
    return context
