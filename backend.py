"""EarthQuery AI v6 backend — map-first remote-sensing workstation with Azure VLM."""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import APIError, APITimeoutError

from agentic import build_model_input, classify_query
from geotiff import GeoTiffReadError, open_dataset, read_bands
from metadata import build_analysis_context, extract_metadata
from model import (
    DEPLOYMENT_NAME,
    EARTHQUERY_SYSTEM_PROMPT,
    append_usage_log,
    call_earthquery,
    create_session_log,
)
from renderer import render_preview_png


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
    allowed_origins: list[str] = field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "*").split(",")
    )
    upload_dir: str = os.getenv("UPLOAD_DIR", "uploads")


settings = Settings()
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("earthquery")

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

app = FastAPI(
    title="EarthQuery AI",
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class Source:
    filename: str
    saved_path: Path
    metadata: dict[str, Any]
    analysis_context: dict[str, Any]
    preview_image: str


@dataclass
class Session:
    sources: list[Source] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    log_file: Path | None = None
    query_count: int = 0
    cumulative_tokens: int = 0
    cumulative_cost_inr: float = 0.0


SESSIONS: dict[str, Session] = {}


def _validate_upload(filename: str, size_bytes: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )
    if size_bytes <= 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )
    if size_bytes > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_mb} MB.",
        )


def _process_source(original_name: str, raw_bytes: bytes) -> Source:
    _validate_upload(original_name, len(raw_bytes))
    safe_name = Path(original_name).name
    saved_path = Path(settings.upload_dir) / f"{uuid.uuid4().hex}_{safe_name}"

    try:
        saved_path.write_bytes(raw_bytes)
        with open_dataset(str(saved_path)) as dataset:
            metadata = extract_metadata(
                dataset,
                safe_name,
                len(raw_bytes),
            )
            bands = read_bands(dataset)
            context = build_analysis_context(
                metadata,
                bands,
                safe_name,
                renderer_notes=render_preview_png.__doc__ or "",
            )
            _, preview = render_preview_png(
                bands,
                nodata=dataset.nodata,
                filename=safe_name,
            )

        return Source(
            safe_name,
            saved_path,
            metadata,
            context,
            preview,
        )

    except GeoTiffReadError:
        saved_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Could not analyse '{safe_name}': {exc}",
        ) from exc


def _source_response(source: Source) -> dict[str, Any]:
    ctx = source.analysis_context
    return {
        "filename": source.filename,
        "metadata": source.metadata,
        "analysis_context": ctx,
        "preview_image": source.preview_image,
        "map": {
            "georeferenced": bool(
                ctx.get("geospatial", {}).get("bounds_wgs84")
            ),
            "bounds_wgs84": ctx.get("geospatial", {}).get("bounds_wgs84"),
            "center_wgs84": ctx.get("geospatial", {}).get("center_wgs84"),
        },
    }


def _build_initial_model_message(sources: list[Source]) -> dict[str, Any]:
    """Build the one-time file/context message sent at session start."""
    blocks: list[str] = []

    for index, source in enumerate(sources, start=1):
        blocks.append(
            f"SOURCE {index}\n"
            f"FILE: {source.filename}\n"
            f"DETERMINISTIC EARTHQUERY ANALYSIS CONTEXT:\n"
            f"{json.dumps(source.analysis_context, indent=2, ensure_ascii=False, default=str)}"
        )

    text = (
        "The following source file(s) are the active EarthQuery analysis "
        "inputs for this session. Keep them in conversation context and use "
        "the supplied metadata together with the visual evidence.\n\n"
        + "\n\n".join(blocks)
    )

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": text,
        }
    ]

    for index, source in enumerate(sources, start=1):
        content.append(
            {
                "type": "text",
                "text": f"Visual evidence for SOURCE {index}: {source.filename}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": source.preview_image,
                },
            }
        )

    return {
        "role": "user",
        "content": content,
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse("index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "application": "EarthQuery AI",
        "version": "6.0.0",
        "model_calls_enabled": True,
        "model": DEPLOYMENT_NAME,
        "active_sessions": len(SESSIONS),
    }


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> JSONResponse:
    if not files or len(files) > 2:
        raise HTTPException(
            status_code=400,
            detail="EarthQuery accepts one or two images per analysis.",
        )

    session_id = uuid.uuid4().hex
    session = Session(log_file=create_session_log())

    try:
        for file in files:
            source = _process_source(
                file.filename or "uploaded_raster",
                await file.read(),
            )
            session.sources.append(source)

    except Exception:
        for source in session.sources:
            source.saved_path.unlink(missing_ok=True)
        if session.log_file:
            # The session log is intentionally retained for auditability even
            # when upload processing fails.
            logger.exception("EarthQuery session upload failed: %s", session.log_file)
        raise

    # The system prompt is sent exactly once at session creation. The source
    # metadata and rendered visual evidence are also sent once here. Later
    # queries only append the user's question and the model's answer.
    session.messages = [
        {
            "role": "system",
            "content": EARTHQUERY_SYSTEM_PROMPT,
        },
        _build_initial_model_message(session.sources),
    ]

    SESSIONS[session_id] = session

    logger.info(
        "EarthQuery v6 session %s created; log=%s; sources=%s",
        session_id,
        session.log_file,
        [source.filename for source in session.sources],
    )

    return JSONResponse(
        {
            "session_id": session_id,
            "source_count": len(session.sources),
            "sources": [_source_response(s) for s in session.sources],
        }
    )


@app.post("/ask")
async def ask(
    session_id: str = Form(...),
    question: str = Form(...),
) -> JSONResponse:
    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    session = SESSIONS.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="No active analysis session. Upload an image first.",
        )

    source_dicts = [
        {
            "filename": source.filename,
            "metadata": source.metadata,
            "analysis_context": source.analysis_context,
        }
        for source in session.sources
    ]

    classification = classify_query(
        question,
        source_dicts,
    )

    # Deterministic orchestration is supplied to the model as context. The
    # model still has access to the actual images and metadata already stored
    # in the session's initial message.
    orchestration = build_model_input(
        question,
        source_dicts,
        classification,
    )

    user_message = {
        "role": "user",
        "content": (
            "Use the active EarthQuery source image(s), their deterministic "
            "metadata, and the following agentic orchestration context to "
            "answer the user\'s question. Do not merely repeat the orchestration.\n\n"
            + orchestration
        ),
    }

    session.messages.append(user_message)

    try:
        answer, usage, latency = call_earthquery(
            session.messages
        )

    except APITimeoutError as exc:
        session.messages.pop()
        raise HTTPException(
            status_code=504,
            detail="Azure OpenAI request timed out.",
        ) from exc

    except APIError as exc:
        session.messages.pop()
        logger.exception("Azure OpenAI API error")
        raise HTTPException(
            status_code=502,
            detail=f"Azure OpenAI request failed: {exc}",
        ) from exc

    except Exception as exc:
        session.messages.pop()
        logger.exception("EarthQuery model call failed")
        raise HTTPException(
            status_code=500,
            detail=f"EarthQuery model call failed: {exc}",
        ) from exc

    # Only successful model responses become part of conversation memory.
    session.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    session.history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )

    session.query_count += 1
    session.cumulative_tokens += usage["total_tokens"]
    session.cumulative_cost_inr += usage["query_cost_inr"]

    if session.log_file is not None:
        append_usage_log(
            session.log_file,
            query_number=session.query_count,
            query=question,
            file_names=[source.filename for source in session.sources],
            usage=usage,
            cumulative_tokens=session.cumulative_tokens,
            cumulative_cost_inr=session.cumulative_cost_inr,
            latency_seconds=latency,
        )

    logger.info(
        "EarthQuery query %s: tokens=%s cost_inr=%.8f latency=%.3fs log=%s",
        session.query_count,
        usage["total_tokens"],
        usage["query_cost_inr"],
        latency,
        session.log_file,
    )

    # Deliberately do NOT return usage, token counts, pricing, latency,
    # cumulative accounting, or the CSV path to the frontend.
    return JSONResponse(
        {
            "answer": answer,
            "classification": classification,
            "model_called": True,
        }
    )


@app.post("/reset")
async def reset(session_id: str = Form(...)) -> JSONResponse:
    session = SESSIONS.pop(session_id, None)

    if session:
        for source in session.sources:
            source.saved_path.unlink(missing_ok=True)

        logger.info(
            "EarthQuery session %s reset; queries=%s; log=%s",
            session_id,
            session.query_count,
            session.log_file,
        )

    return JSONResponse({"status": "cleared"})
