"""EarthQuery v6 deterministic agentic orchestration.

This module deliberately does not call an LLM. It classifies the request and
builds the exact structured input that a future remote-sensing model will
receive.
"""
from __future__ import annotations

import re
from typing import Any


ANNOTATION_CUES = re.compile(
    r"\b(annotate|annotation|mark|outline|bounding box|bbox|segment|segmentation|locate|highlight|detect|identify .* object|where .* (building|road|water|tree|field))\b",
    re.I,
)
CHANGE_CUES = re.compile(
    r"\b(before|after|changed|change|difference over time|temporal|new construction|removed|increase|decrease|what changed|has .* changed)\b",
    re.I,
)


def _bounds_area(b: dict[str, Any] | None) -> float:
    if not b:
        return 0.0
    return abs(float(b["right"]) - float(b["left"])) * abs(float(b["top"]) - float(b["bottom"]))


def _intersection_area(a: dict[str, Any], b: dict[str, Any]) -> float:
    left = max(a["left"], b["left"])
    right = min(a["right"], b["right"])
    bottom = max(a["bottom"], b["bottom"])
    top = min(a["top"], b["top"])
    if right <= left or top <= bottom:
        return 0.0
    return (right-left) * (top-bottom)


def compare_footprints(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ba = a.get("bounds")
    bb = b.get("bounds")
    if not ba or not bb or not a.get("crs") or not b.get("crs"):
        return {"relationship": "unknown", "reason": "Both rasters must have comparable georeferencing."}
    if a.get("crs") != b.get("crs"):
        return {"relationship": "unknown", "reason": "Rasters use different CRS definitions; reprojection is required before exact footprint comparison."}
    ia = _intersection_area(ba, bb)
    aa = _bounds_area(ba)
    ab = _bounds_area(bb)
    union = aa + ab - ia
    iou = ia / union if union else 0.0
    min_cover = ia / min(aa, ab) if min(aa, ab) else 0.0
    if iou >= 0.75 or min_cover >= 0.90:
        rel = "same_location"
    else:
        rel = "different_locations"
    return {
        "relationship": rel,
        "intersection_over_union": round(iou, 6),
        "smaller_footprint_coverage": round(min_cover, 6),
        "crs": a.get("crs"),
    }


def classify_query(question: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    q = question.strip()
    count = len(sources)
    annotation = bool(ANNOTATION_CUES.search(q))

    if count == 1:
        query_type = "single image annotation" if annotation else "single image vqa"
        operations = ["annotation"] if annotation else ["vqa"]
        return {
            "query_type": query_type,
            "input": "single_image",
            "operations": operations,
            "location_relationship": None,
            "classification_basis": "single raster input; annotation cues detected" if annotation else "single raster input; no annotation cues detected",
        }

    if count == 2:
        relationship = compare_footprints(sources[0].get("metadata", {}), sources[1].get("metadata", {}))
        if relationship["relationship"] == "same_location" or CHANGE_CUES.search(q):
            scenario = "before_after_same_location"
            prefix = "change_detection"
        else:
            scenario = "different_locations"
            prefix = "cross_modal_analysis"
        prefix_label = prefix.replace("_", " ")
        if annotation:
            query_type = f"{prefix_label} vqa and annotation"
            operations = [prefix, "vqa", "annotation"]
        else:
            query_type = f"{prefix_label} vqa"
            operations = [prefix, "vqa"]
        return {
            "query_type": query_type,
            "input": "multi_image",
            "operations": operations,
            "scenario": scenario,
            "location_relationship": relationship,
            "classification_basis": "geospatial footprint comparison plus query wording",
        }

    return {"query_type": "unsupported", "input": f"{count}_images", "operations": [], "classification_basis": "maximum two rasters"}


def _source_block(source: dict[str, Any]) -> str:
    ctx = source["analysis_context"]
    return (
        f"FILE: {source['filename']}\n"
        f"IMAGE TYPE: {ctx.get('image_type')}\n"
        f"MODALITY: {ctx.get('modality')}\n"
        f"PLATFORM: {(ctx.get('product') or {}).get('platform')}\n"
        f"POLARIZATION: {(ctx.get('product') or {}).get('polarization')}\n"
        f"DIMENSIONS: {(ctx.get('raster') or {}).get('dimensions')}\n"
        f"CRS/EPSG: {ctx.get('crs', {}).get('name')} / {ctx.get('crs', {}).get('epsg')}\n"
        f"PIXEL RESOLUTION: {(ctx.get('raster') or {}).get('pixel_resolution')}\n"
        f"WGS84 CENTER: {(ctx.get('geospatial') or {}).get('center_wgs84')}\n"
        f"WGS84 BOUNDS: {(ctx.get('geospatial') or {}).get('bounds_wgs84')}\n"
        f"ACQUISITION: {(ctx.get('source_metadata') or {}).get('acquisition_datetime')}\n"
        f"SPECTRAL/RADIOMETRIC ANALYSIS: {(ctx.get('raster') or {}).get('spectral_analysis')}"
    )


def build_model_input(question: str, sources: list[dict[str, Any]], classification: dict[str, Any]) -> str:
    """Human-readable orchestration payload supplied to the active model."""
    blocks = []
    for i, source in enumerate(sources, start=1):
        blocks.append(f"SOURCE {i}\n{_source_block(source)}")
    relationship = classification.get("location_relationship")
    return (
        "Agent orchestration:\n"
        + "\n\n".join(blocks)
        + "\n\nAgentic orchestration:\n"
        + f"query type: {classification.get('query_type')}\n"
        + f"input: {classification.get('input')}\n"
        + f"operations: {', '.join(classification.get('operations', []))}\n"
        + (f"scenario: {classification.get('scenario')}\n" if classification.get('scenario') else "")
        + (f"location relationship: {relationship}\n" if relationship else "")
        + f"question: {question}\n"
        + "\nmodel_input_type: earthquery_remote_sensing_v1\n"
        + "model_status: MODEL CALLED — ORCHESTRATION CONTEXT SENT TO AZURE OPENAI"
    )
