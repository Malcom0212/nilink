"""
Nilink API Server
=================
REST + WebSocket API for the Nilink forensic verification engine.

Endpoints:
- POST /verify       — Analyze a single image (upload or URL)
- POST /verify/base64 — Analyze a base64-encoded image
- POST /verify/batch — Analyze multiple images
- GET  /health       — Health check
- WS   /ws/stream    — Real-time video stream analysis

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings
from logging_config import setup_logging, RequestLoggingMiddleware
from Nilink_engine import NilinkVerifier, VerificationResult


# --- Logging ---

logger = setup_logging()


# --- Rate Limiting ---

limiter = Limiter(key_func=get_remote_address)


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}. Please retry later."},
    )


# --- Models ---

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error description")


class VerifyResponse(BaseModel):
    """Response schema for image verification."""
    global_trust_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Trust score: 0.0 (fake) to 1.0 (authentic)",
        json_schema_extra={"example": 0.8234},
    )
    anomalies_found: list[str] = Field(
        default_factory=list,
        description="List of detected anomalies",
        json_schema_extra={"example": ["ELA: Inconsistent compression levels detected"]},
    )
    suspicious_regions: list[dict] = Field(
        default_factory=list,
        description="Bounding boxes of suspicious areas [{x, y, w, h, reason}]",
        json_schema_extra={"example": [{"x": 120, "y": 80, "w": 200, "h": 150, "reason": "Inconsistent error level"}]},
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds",
        json_schema_extra={"example": 42.56},
    )
    heatmap_base64: Optional[str] = Field(
        None, description="Manipulation heatmap as base64-encoded PNG (only when include_heatmap=true)",
    )


class VerifyBase64Request(BaseModel):
    """Request body for base64 image verification."""
    image_base64: str = Field(
        ..., description="Base64-encoded image (JPEG or PNG)",
    )
    include_heatmap: bool = Field(
        default=False, description="Include heatmap visualization in response",
    )


class BatchResponse(BaseModel):
    """Response for batch verification."""
    results: list[VerifyResponse] = Field(description="Verification results for each image")
    total_processing_time_ms: float = Field(description="Total wall-clock processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Engine status: 'ok' or 'not_ready'")
    engine_version: str = "0.1.0"
    detectors: dict = Field(description="Enabled state of each detector")


# --- App ---

# Shared engine instance
verifier: Optional[NilinkVerifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine on startup."""
    global verifier
    verifier = NilinkVerifier()
    logger.info("Nilink engine initialized")
    yield
    verifier = None
    logger.info("Nilink engine shut down")


DESCRIPTION = """\
Forensic analysis API for detecting image and video manipulations in real-time.

## Capabilities

* **ELA** — Error Level Analysis for inpainting / AI-generation detection
* **FFT** — Spectral analysis for GAN / diffusion model artifacts
* **rPPG** — Remote photoplethysmography for deepfake / liveness detection
* **Upscale** — Noise pattern analysis for AI upscaling detection

## Rate Limits

| Endpoint | Limit |
|---|---|
| `POST /verify` | {rate_limit} |
| `POST /verify/base64` | {rate_limit} |
| `POST /verify/batch` | {rate_limit_batch} |
| `GET /health` | No limit |
| `WS /ws/stream` | No limit (frame dropping handles load) |

Exceeding the limit returns **429 Too Many Requests**.
""".format(rate_limit=settings.RATE_LIMIT, rate_limit_batch=settings.RATE_LIMIT_BATCH)

app = FastAPI(
    title="Nilink Verifier API",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "verification", "description": "Image and video forensic verification endpoints"},
        {"name": "monitoring", "description": "Health and status endpoints"},
    ],
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Engine not initialized"},
    },
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers ---

def _encode_heatmap(heatmap: Optional[np.ndarray]) -> Optional[str]:
    """Encode heatmap numpy array to base64 PNG string."""
    if heatmap is None:
        return None
    _, buffer = cv2.imencode(".png", heatmap)
    return base64.b64encode(buffer).decode("utf-8")


def _result_to_response(result: VerificationResult, include_heatmap: bool = False) -> VerifyResponse:
    """Convert engine result to API response."""
    return VerifyResponse(
        global_trust_score=round(result.global_trust_score, 4),
        anomalies_found=result.anomalies_found,
        suspicious_regions=result.suspicious_regions,
        processing_time_ms=round(result.processing_time_ms, 2),
        heatmap_base64=_encode_heatmap(result.manipulation_heatmap) if include_heatmap else None,
    )


async def _read_upload_image(file: UploadFile) -> np.ndarray:
    """Read uploaded file and decode to numpy array."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file. Supported formats: JPEG, PNG, BMP, TIFF.")
    return image


def _decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    try:
        image_bytes = base64.b64decode(b64_string)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding.")
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image from base64. Supported formats: JPEG, PNG.")
    return image


# --- REST Endpoints ---

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["monitoring"],
    summary="Health check",
    responses={200: {"description": "Engine status and enabled detectors"}},
)
async def health():
    """Health check — confirms the engine is ready and lists enabled detectors."""
    return HealthResponse(
        status="ok" if verifier else "not_ready",
        detectors={
            "ela": verifier.enable_ela if verifier else False,
            "fft": verifier.enable_fft if verifier else False,
            "rppg": verifier.enable_rppg if verifier else False,
            "upscale": verifier.enable_upscale_detection if verifier else False,
        },
    )


@app.post(
    "/verify",
    response_model=VerifyResponse,
    tags=["verification"],
    summary="Analyze a single image (file upload)",
    responses={
        200: {"description": "Forensic analysis result with trust score and anomalies"},
        400: {"model": ErrorResponse, "description": "Invalid image file"},
    },
)
@limiter.limit(settings.RATE_LIMIT)
async def verify_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG)"),
    include_heatmap: bool = Query(False, description="Include heatmap in response"),
):
    """
    Analyze a single image for manipulations.

    Upload an image file (JPEG, PNG) and receive a forensic analysis
    with trust score, detected anomalies, and suspicious regions.

    **Rate limit:** {rate_limit}
    """.format(rate_limit=settings.RATE_LIMIT)
    if verifier is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    image = await _read_upload_image(file)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, verifier.verify_image, image)

    logger.info(
        "verify trust_score=%.4f anomalies=%d",
        result.global_trust_score,
        len(result.anomalies_found),
        extra={"trust_score": round(result.global_trust_score, 4)},
    )

    return _result_to_response(result, include_heatmap)


@app.post(
    "/verify/base64",
    response_model=VerifyResponse,
    tags=["verification"],
    summary="Analyze a base64-encoded image",
    responses={
        200: {"description": "Forensic analysis result"},
        400: {"model": ErrorResponse, "description": "Invalid base64 or image data"},
    },
)
@limiter.limit(settings.RATE_LIMIT)
async def verify_image_base64(
    request: Request,
    body: VerifyBase64Request,
):
    """
    Analyze a base64-encoded image for manipulations.

    Send a base64-encoded image and receive forensic analysis.
    Useful for browser integrations and API-to-API calls.

    **Rate limit:** {rate_limit}
    """.format(rate_limit=settings.RATE_LIMIT)
    if verifier is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    image = _decode_base64_image(body.image_base64)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, verifier.verify_image, image)

    logger.info(
        "verify/base64 trust_score=%.4f anomalies=%d",
        result.global_trust_score,
        len(result.anomalies_found),
        extra={"trust_score": round(result.global_trust_score, 4)},
    )

    return _result_to_response(result, body.include_heatmap)


@app.post(
    "/verify/batch",
    response_model=BatchResponse,
    tags=["verification"],
    summary="Analyze multiple images in one request",
    responses={
        200: {"description": "Array of forensic analysis results"},
        400: {"model": ErrorResponse, "description": "Invalid image or batch size exceeded"},
    },
)
@limiter.limit(settings.RATE_LIMIT_BATCH)
async def verify_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Multiple image files to analyze"),
    include_heatmap: bool = Query(False, description="Include heatmaps in responses"),
):
    """
    Analyze multiple images in a single request.

    Upload up to {max_batch} images. Each is analyzed independently.

    **Rate limit:** {rate_limit_batch}
    """.format(max_batch=settings.MAX_BATCH_SIZE, rate_limit_batch=settings.RATE_LIMIT_BATCH)
    if verifier is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.MAX_BATCH_SIZE} images per batch request.",
        )

    start = time.perf_counter()

    # Decode all images first (fail fast on bad input)
    images = []
    for f in files:
        images.append(await _read_upload_image(f))

    # Process all images concurrently
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, verifier.verify_image, img) for img in images]
    results = await asyncio.gather(*tasks)

    responses = [_result_to_response(r, include_heatmap) for r in results]
    total_time = (time.perf_counter() - start) * 1000

    return BatchResponse(results=responses, total_processing_time_ms=round(total_time, 2))


# --- WebSocket ---

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Real-time video stream analysis via WebSocket.

    Protocol:
    - Client connects to /ws/stream
    - Client sends binary frames (JPEG-encoded) or JSON with base64 image
    - Server responds with JSON verification result for each frame
    - Server drops frames if processing can't keep up

    Binary message format:
        Raw JPEG bytes

    JSON message format:
        {"image_base64": "...", "timestamp_ms": 12345.0}

    Response format:
        {
            "global_trust_score": 0.85,
            "anomalies_found": [...],
            "suspicious_regions": [...],
            "processing_time_ms": 42.5,
            "frame_dropped": false
        }
    """
    if verifier is None:
        await ws.close(code=1013, reason="Engine not initialized")
        return

    await ws.accept()

    # Create a dedicated verifier for this stream (separate rPPG buffer)
    stream_verifier = NilinkVerifier()

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message and message["bytes"]:
                # Binary frame (raw JPEG)
                frame_bytes = message["bytes"]
                timestamp_ms = time.perf_counter() * 1000

                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await ws.send_json({"error": "Invalid frame data"})
                    continue

            elif "text" in message and message["text"]:
                # JSON message with base64
                import json
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"error": "Invalid JSON"})
                    continue

                if "image_base64" not in data:
                    await ws.send_json({"error": "Missing image_base64 field"})
                    continue

                timestamp_ms = data.get("timestamp_ms", time.perf_counter() * 1000)

                try:
                    frame_bytes = base64.b64decode(data["image_base64"])
                except Exception:
                    await ws.send_json({"error": "Invalid base64"})
                    continue

                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await ws.send_json({"error": "Could not decode image"})
                    continue
            else:
                continue

            # Process frame with latency management
            result = await stream_verifier.process_stream_frame(frame, timestamp_ms)

            # Send result (no heatmap over WebSocket to save bandwidth)
            await ws.send_json({
                "global_trust_score": round(result.global_trust_score, 4),
                "anomalies_found": result.anomalies_found,
                "suspicious_regions": result.suspicious_regions,
                "processing_time_ms": round(result.processing_time_ms, 2),
                "frame_dropped": result.frame_dropped,
            })

    except WebSocketDisconnect:
        pass
    except Exception:
        await ws.close(code=1011, reason="Internal error")


# --- Entry point ---

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  NILINK VERIFIER API")
    print(f"  http://localhost:{settings.PORT}")
    print(f"  Docs: http://localhost:{settings.PORT}/docs")
    print("=" * 50)
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
