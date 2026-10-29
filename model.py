"""EarthQuery v6 Azure OpenAI model integration and usage accounting."""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import APIError, APITimeoutError, OpenAI

AZURE_ENDPOINT = "https://satquery-ai-vlm2-resource.openai.azure.com/openai/v1/"
DEPLOYMENT_NAME = "gpt-4.1-mini"
REQUEST_TIMEOUT = 120

# Azure pricing calculator values supplied for GPT-4.1 mini, UAE North,
# Global deployment, Standard (On-Demand). All values are INR / 1,000 tokens.
INPUT_PRICE_PER_1K_INR = 0.03822
CACHED_INPUT_PRICE_PER_1K_INR = 0.00955
OUTPUT_PRICE_PER_1K_INR = 0.15287

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "AZURE_OPENAI_API_KEY environment variable is not set."
    )

client = OpenAI(
    api_key=API_KEY,
    base_url=AZURE_ENDPOINT,
    timeout=REQUEST_TIMEOUT,
)


EARTHQUERY_SYSTEM_PROMPT = """
You are EarthQuery's primary visual and geospatial reasoning engine.

EarthQuery is a remote-sensing and geospatial AI system. Its purpose is to
understand supplied satellite imagery, remote-sensing imagery, GeoTIFFs,
raster datasets, ordinary images, and related scientific/geospatial data and
to answer questions about them accurately. You are not merely a generic
chatbot looking at pictures.

Understand the nature of the supplied data before interpreting its visual
appearance.

FILE AND DATA UNDERSTANDING

A supplied file may be an ordinary image, JPEG, PNG, WEBP, TIFF/GeoTIFF,
a single-band scientific raster, a multispectral raster, SAR imagery, an
optical satellite image, PDF, text, JSON, CSV, spreadsheet, or another
structured/scientific file.

Do not assume every image is an RGB photograph. A TIFF may contain numerical
raster values rather than photographic RGB pixels. A GeoTIFF may contain CRS,
coordinate system, geographic bounds, affine transform, pixel resolution,
raster dimensions, band information, NoData values, datatype, spatial
reference, acquisition/product information, and other tags.

A scientific raster may represent a spectral band, reflectance, temperature,
elevation, radar backscatter, or another physical quantity. Exact semantics
must come from the supplied metadata and established sensor/product context;
do not invent them.

REMOTE-SENSING REASONING

When analysing remote-sensing imagery, consider sensor characteristics,
spectral band/wavelength, spatial resolution, pixel structure, brightness,
texture, shape, spatial arrangement, surrounding context, shadows,
land-cover patterns, geographic information, acquisition information, and
raster statistics when supplied.

For SAR imagery, brightness represents radar backscatter behaviour rather
than ordinary visible brightness. For single-band imagery, grayscale is a
visualisation of one measurement, not necessarily a black-and-white photo.
For multispectral imagery, bands can represent different portions of the
electromagnetic spectrum.

REASONING METHOD

For every question:
1. Understand what the supplied file represents.
2. Understand exactly what the user is asking.
3. Identify evidence relevant to that question.
4. Examine visual evidence where available.
5. Examine supplied technical metadata and file content.
6. Consider spatial relationships and remote-sensing context.
7. Distinguish direct observation from interpretation.
8. Distinguish interpretation from inference.
9. Consider image resolution and data limitations.
10. Answer using the strongest available evidence.

Do not fabricate information. If something cannot be reliably determined,
state the limitation. For counts, attempt a count but acknowledge resolution
or ambiguity. For measurements, do not invent measurements. If metadata
provides an exact value, distinguish metadata from visual observation.

CONVERSATION MEMORY

This is a continuing EarthQuery session. Remember information already
established in the conversation. Resolve references such as "the image",
"that building", "the previous question", "the bright region", or "the B08
band" using the conversation history instead of repeatedly asking for
information already established.

AGENTIC ORCHESTRATION

The backend may provide deterministic query classification and geospatial
relationship analysis. Treat that information as supporting context, not as
an unquestionable conclusion. Use the actual image and metadata as evidence.

Do not reveal private chain-of-thought or hidden reasoning. Provide the
conclusion and a concise explanation of the evidence supporting it.
""".strip()


@dataclass
class UsageResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    normal_input_tokens: int
    input_cost_inr: float
    cached_input_cost_inr: float
    output_cost_inr: float
    query_cost_inr: float
    latency_seconds: float


def _usage_value(usage: Any, name: str) -> int:
    return int(getattr(usage, name, 0) or 0)


def calculate_usage(usage: Any) -> dict[str, Any]:
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)

    cached_tokens = int(
        getattr(prompt_details, "cached_tokens", 0) or 0
    ) if prompt_details else 0

    reasoning_tokens = int(
        getattr(completion_details, "reasoning_tokens", 0) or 0
    ) if completion_details else 0

    normal_input_tokens = max(prompt_tokens - cached_tokens, 0)

    input_cost_inr = (
        normal_input_tokens / 1000.0
    ) * INPUT_PRICE_PER_1K_INR

    cached_input_cost_inr = (
        cached_tokens / 1000.0
    ) * CACHED_INPUT_PRICE_PER_1K_INR

    output_cost_inr = (
        completion_tokens / 1000.0
    ) * OUTPUT_PRICE_PER_1K_INR

    query_cost_inr = (
        input_cost_inr
        + cached_input_cost_inr
        + output_cost_inr
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "normal_input_tokens": normal_input_tokens,
        "input_cost_inr": input_cost_inr,
        "cached_input_cost_inr": cached_input_cost_inr,
        "output_cost_inr": output_cost_inr,
        "query_cost_inr": query_cost_inr,
    }


def create_session_log() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = log_dir / f"earthquery_usage_{timestamp}.csv"

    fields = [
        "timestamp",
        "query_number",
        "query",
        "model",
        "file_names",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "normal_input_tokens",
        "input_cost_inr",
        "cached_input_cost_inr",
        "output_cost_inr",
        "query_cost_inr",
        "cumulative_tokens",
        "cumulative_cost_inr",
        "latency_seconds",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()

    return path


def append_usage_log(
    log_file: Path,
    *,
    query_number: int,
    query: str,
    file_names: list[str],
    usage: dict[str, Any],
    cumulative_tokens: int,
    cumulative_cost_inr: float,
    latency_seconds: float,
) -> None:
    fields = [
        "timestamp",
        "query_number",
        "query",
        "model",
        "file_names",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "normal_input_tokens",
        "input_cost_inr",
        "cached_input_cost_inr",
        "output_cost_inr",
        "query_cost_inr",
        "cumulative_tokens",
        "cumulative_cost_inr",
        "latency_seconds",
    ]

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query_number": query_number,
        "query": query,
        "model": DEPLOYMENT_NAME,
        "file_names": " | ".join(file_names),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "normal_input_tokens": usage["normal_input_tokens"],
        "input_cost_inr": f"{usage['input_cost_inr']:.8f}",
        "cached_input_cost_inr": f"{usage['cached_input_cost_inr']:.8f}",
        "output_cost_inr": f"{usage['output_cost_inr']:.8f}",
        "query_cost_inr": f"{usage['query_cost_inr']:.8f}",
        "cumulative_tokens": cumulative_tokens,
        "cumulative_cost_inr": f"{cumulative_cost_inr:.8f}",
        "latency_seconds": f"{latency_seconds:.3f}",
    }

    with log_file.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writerow(row)


def call_earthquery(
    messages: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], float]:
    """Call the configured Azure OpenAI deployment and return answer, usage, latency."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=1400,
        )
    except (APITimeoutError, APIError):
        raise

    latency = time.perf_counter() - start
    answer = response.choices[0].message.content or ""

    if response.usage is None:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "normal_input_tokens": 0,
            "input_cost_inr": 0.0,
            "cached_input_cost_inr": 0.0,
            "output_cost_inr": 0.0,
            "query_cost_inr": 0.0,
        }
    else:
        usage = calculate_usage(response.usage)

    return answer, usage, latency
