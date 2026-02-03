"""
Nilink API Server
=================
REST + WebSocket API for the Nilink forensic verification engine.

Endpoints:
- POST /verify       — Analyze a single image (upload or URL)
- POST /verify/batch — Analyze multiple images
- GET  /health       — Health check
- WS   /ws/stream    — Real-time video stream analysis

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from Nilink_engine import NilinkVerifier, VerificationResult


# --- Models ---

class VerifyResponse(BaseModel):
    """Response schema for image verification."""
    global_trust_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Trust score: 0.0 (fake) to 1.0 (authentic)",
    )
    anomalies_found: list[str] = Field(
        default_factory=list,
        description="List of detected anomalies",
    )
    suspicious_regions: list[dict] = Field(
        default_factory=list,
        description="Bounding boxes of suspicious areas [{x, y, w, h, reason}]",
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds",
    )
    heatmap_base64: Optional[str] = Field(
        None, description="Manipulation heatmap as base64-encoded PNG",
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
    results: list[VerifyResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    engine_version: str = "0.1.0"
    detectors: dict


# --- App ---

# Shared engine instance
verifier: Optional[NilinkVerifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine on startup."""
    global verifier
    verifier = NilinkVerifier()
    yield
    verifier = None


app = FastAPI(
    title="Nilink Verifier API",
    description="Forensic analysis API for detecting image/video manipulations in real-time.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms the engine is ready."""
    return HealthResponse(
        status="ok" if verifier else "not_ready",
        detectors={
            "ela": verifier.enable_ela if verifier else False,
            "fft": verifier.enable_fft if verifier else False,
            "rppg": verifier.enable_rppg if verifier else False,
            "upscale": verifier.enable_upscale_detection if verifier else False,
        },
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify_image(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG)"),
    include_heatmap: bool = Query(False, description="Include heatmap in response"),
):
    """
    Analyze a single image for manipulations.

    Upload an image file (JPEG, PNG) and receive a forensic analysis
    with trust score, detected anomalies, and suspicious regions.
    """
    if verifier is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    image = await _read_upload_image(file)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, verifier.verify_image, image)

    return _result_to_response(result, include_heatmap)


@app.post("/verify/base64", response_model=VerifyResponse)
async def verify_image_base64(request: VerifyBase64Request):
    """
    Analyze a base64-encoded image for manipulations.

    Send a base64-encoded image and receive forensic analysis.
    Useful for browser integrations and API-to-API calls.
    """
    if verifier is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    image = _decode_base64_image(request.image_base64)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, verifier.verify_image, image)

    return _result_to_response(result, request.include_heatmap)


@app.post("/verify/batch", response_model=BatchResponse)
async def verify_batch(
    files: list[UploadFile] = File(..., description="Multiple image files to analyze"),
    include_heatmap: bool = Query(False, description="Include heatmaps in responses"),
):
    """
    Analyze multiple images in a single request.

    Upload up to 10 images. Each is analyzed independently.
    """
    if verifier is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch request.")

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
    print("  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
