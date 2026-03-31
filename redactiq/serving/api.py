"""FastAPI server for RedactIQ.

Provides REST endpoints for single and batch document redaction,
health checks, and metrics. Designed for deployment behind
a reverse proxy or directly on an Intel Xeon instance.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from redactiq.redaction.pipeline import RedactionPipeline
from redactiq.utils.config import load_config
from redactiq.utils.file_parser import extract_text
from redactiq.utils.models import (
    BatchRedactionRequest,
    HealthResponse,
    RedactionRequest,
    RedactionResult,
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "redactiq_requests_total",
    "Total redaction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "redactiq_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
PII_DETECTED = Counter(
    "redactiq_pii_detected_total",
    "Total PII entities detected",
    ["entity_type", "source"],
)
ANOMALIES_FLAGGED = Counter(
    "redactiq_anomalies_total",
    "Total anomalies flagged",
)

# Global state
_pipeline: RedactionPipeline | None = None
_config: dict[str, Any] = {}
_rate_limit_window: dict[str, list[float]] = defaultdict(list)
_last_rate_limit_cleanup: float = 0.0

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global _pipeline, _config
    _config = load_config()

    _pipeline = RedactionPipeline(_config)

    # Only load LLM if explicitly enabled
    llm_enabled = _config.get("detection", {}).get("llm", {}).get("enabled", False)
    if llm_enabled:
        try:
            _pipeline.load_models()
        except Exception as e:
            logger.warning(f"Could not load LLM models: {e}. Running rules-only mode.")

    # Load anomaly model if a saved model exists
    anomaly_path = _PROJECT_ROOT / "models" / "anomaly_model.pkl"
    if anomaly_path.exists():
        try:
            _pipeline.anomaly_detector.load(anomaly_path)
            _pipeline.anomaly_detector.load_embedder()
        except Exception as e:
            logger.warning(f"Could not load anomaly model: {e}")

    logger.info("RedactIQ server started")
    yield
    logger.info("RedactIQ server shutting down")


app = FastAPI(
    title="RedactIQ",
    description="Smart PII Redaction Assistant with Hybrid Detection and Anomaly Detection",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_rate_limit(client_ip: str) -> bool:
    """Sliding-window rate limiter (per minute). Returns True if allowed."""
    global _last_rate_limit_cleanup
    rate_limit = _config.get("serving", {}).get("rate_limit", 100)
    if rate_limit <= 0:
        return True

    now = time.time()

    # Periodic cleanup of stale IPs to prevent unbounded memory growth
    if now - _last_rate_limit_cleanup > 300:
        _last_rate_limit_cleanup = now
        stale_ips = [
            ip for ip, timestamps in _rate_limit_window.items()
            if all(now - t >= 60 for t in timestamps)
        ]
        for ip in stale_ips:
            del _rate_limit_window[ip]

    window = _rate_limit_window[client_ip]
    _rate_limit_window[client_ip] = [t for t in window if now - t < 60]

    if len(_rate_limit_window[client_ip]) >= rate_limit:
        return False

    _rate_limit_window[client_ip].append(now)
    return True


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=_pipeline is not None and _pipeline.detector.llm_detector._llm is not None,
        anomaly_model_loaded=_pipeline is not None and _pipeline.anomaly_detector.ready,
    )


@app.post("/redact", response_model=RedactionResult)
async def redact_document(request: RedactionRequest, raw_request: Request):
    """Redact PII from a single document."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    client_ip = raw_request.client.host if raw_request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    start = time.perf_counter()

    try:
        result = _pipeline.process(
            text=request.text,
            mode=request.mode,
            detect_anomalies=request.detect_anomalies,
            entity_types=request.entity_types,
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/redact", status="error").inc()
        logger.error(f"Redaction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.perf_counter() - start
    REQUEST_COUNT.labels(endpoint="/redact", status="ok").inc()
    REQUEST_LATENCY.labels(endpoint="/redact").observe(elapsed)

    for entity in result.entities:
        PII_DETECTED.labels(
            entity_type=entity.entity_type.value,
            source=entity.source.value,
        ).inc()

    ANOMALIES_FLAGGED.inc(len(result.anomaly_flags))
    _audit_log(request.text, result)

    return result


@app.post("/redact/batch")
async def redact_batch(request: BatchRedactionRequest, raw_request: Request):
    """Redact PII from multiple documents (per-document error isolation)."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    client_ip = raw_request.client.host if raw_request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        results = _pipeline.process_batch(
            texts=request.documents,
            mode=request.mode,
            detect_anomalies=request.detect_anomalies,
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/redact/batch", status="error").inc()
        logger.error(f"Batch redaction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    REQUEST_COUNT.labels(endpoint="/redact/batch", status="ok").inc()
    return {"results": [r.model_dump() for r in results]}


@app.post("/redact/file")
async def redact_file(
    raw_request: Request,
    file: UploadFile = File(...),
    mode: str = Form("mask"),
    detect_anomalies: bool = Form(True),
):
    """Redact PII from an uploaded file.

    Supported formats: **PDF, DOCX, XLSX, CSV, TXT, LOG, JSON, MD, TSV**.

    - **PDF/DOCX**: Text is extracted page-by-page / paragraph-by-paragraph.
    - **XLSX**: Each sheet is processed separately.
    - **CSV**: Each cell is scanned and redacted individually.
    - **Text files**: Entire content is redacted.

    Returns JSON with redacted content per page/section, detected entities,
    and anomaly flags.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    client_ip = raw_request.client.host if raw_request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    raw_bytes = await file.read()
    filename = file.filename or "upload.txt"
    start = time.perf_counter()

    try:
        parsed = extract_text(raw_bytes, filename)
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/redact/file", status="error").inc()
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    file_format = parsed["format"]
    all_entities: list[dict] = []
    total_anomalies = 0
    redacted_pages: list[dict] = []

    try:
        if file_format == "csv":
            # CSV: cell-by-cell redaction
            redacted_content, all_entities, total_anomalies = _redact_csv(
                parsed["pages"][0]["text"], mode, detect_anomalies
            )
            redacted_pages.append({"page": "CSV", "redacted_text": redacted_content})
        else:
            # All other formats: page-by-page redaction
            for page_info in parsed["pages"]:
                page_text = page_info["text"]
                if not page_text.strip():
                    continue

                result = _pipeline.process(
                    text=page_text,
                    mode=mode,
                    detect_anomalies=detect_anomalies,
                )
                redacted_pages.append({
                    "page": page_info["page"],
                    "redacted_text": result.redacted_text,
                })
                for e in result.entities:
                    all_entities.append({
                        "type": e.entity_type.value,
                        "text": e.text,
                        "confidence": round(e.confidence, 3),
                        "source": e.source.value,
                        "page": page_info["page"],
                    })
                total_anomalies += len(result.anomaly_flags)

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/redact/file", status="error").inc()
        logger.error(f"File redaction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Build combined redacted content
    if file_format == "csv":
        combined_redacted = redacted_pages[0]["redacted_text"] if redacted_pages else ""
    else:
        combined_redacted = "\n\n".join(
            f"--- {p['page']} ---\n{p['redacted_text']}" for p in redacted_pages
        )

    elapsed = time.perf_counter() - start
    REQUEST_COUNT.labels(endpoint="/redact/file", status="ok").inc()
    REQUEST_LATENCY.labels(endpoint="/redact/file").observe(elapsed)

    return {
        "filename": filename,
        "format": file_format,
        "redacted_content": combined_redacted,
        "redacted_pages": redacted_pages,
        "total_entities_found": len(all_entities),
        "total_anomalies_flagged": total_anomalies,
        "processing_time_ms": round(elapsed * 1000, 2),
        "entities": all_entities,
    }


def _redact_csv(content: str, mode: str, detect_anomalies: bool) -> tuple[str, list, int]:
    """Process a CSV file cell by cell and return redacted CSV content.

    Anomaly detection runs once on all non-empty cell texts (batched)
    rather than per-cell to reduce overhead.
    """
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)

    all_entities = []
    redacted_rows = []

    # Pass 1: detect + redact each cell (lightweight, no anomaly detection per cell)
    for row in rows:
        redacted_row = []
        for cell in row:
            cell_text = cell.strip()
            if not cell_text:
                redacted_row.append(cell)
                continue

            redacted, cell_entities = _pipeline.redact_cell(cell_text, mode=mode)
            redacted_row.append(redacted)

            for e in cell_entities:
                all_entities.append({
                    "type": e.entity_type.value,
                    "text": e.text,
                    "confidence": round(e.confidence, 3),
                    "source": e.source.value,
                })

        redacted_rows.append(redacted_row)

    # Pass 2: run anomaly detection once on all redacted cells (batched)
    total_anomalies = 0
    if detect_anomalies and _pipeline.anomaly_detector.ready:
        all_texts = [
            cell.strip()
            for row in redacted_rows
            for cell in row
            if cell.strip()
        ]
        if all_texts:
            flags = _pipeline.anomaly_detector.detect(all_texts)
            total_anomalies = len(flags)

    # Write back to CSV string
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(redacted_rows)
    return output.getvalue(), all_entities, total_anomalies


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


def _audit_log(original_text: str, result: RedactionResult):
    """Append an audit entry. Logs input hash (not raw PII) for traceability.

    Wrapped in try/except so audit failures never crash requests.
    """
    try:
        audit_file = _config.get("monitoring", {}).get("audit_log", "logs/audit.jsonl")
        Path(audit_file).parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_hash": hashlib.sha256(original_text.encode()).hexdigest()[:16],
            "input_length": len(original_text),
            "entities_found": len(result.entities),
            "anomalies_flagged": len(result.anomaly_flags),
            "processing_time_ms": result.processing_time_ms,
            "entity_types": [e.entity_type.value for e in result.entities],
            "entity_sources": [e.source.value for e in result.entities],
        }

        with open(audit_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Audit log write failed: {e}")


def main():
    """Entry point for the server."""
    config = load_config()
    serving = config.get("serving", {})
    uvicorn.run(
        "redactiq.serving.api:app",
        host=serving.get("host", "0.0.0.0"),
        port=serving.get("port", 8000),
        workers=serving.get("workers", 4),
        log_level=config.get("monitoring", {}).get("log_level", "info").lower(),
    )


if __name__ == "__main__":
    main()
