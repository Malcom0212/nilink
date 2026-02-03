"""
Tests for the Nilink API server.

Run with:
    pytest tests/test_api.py -v
"""

import base64
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture
def client():
    """FastAPI test client with lifespan (initializes engine)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_jpeg_bytes():
    """Generate a sample JPEG image as bytes."""
    rng = np.random.default_rng(42)
    img = rng.integers(80, 180, (200, 300, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


@pytest.fixture
def sample_jpeg_base64(sample_jpeg_bytes):
    """Sample image as base64 string."""
    return base64.b64encode(sample_jpeg_bytes).decode("utf-8")


# --- Health ---

class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["engine_version"] == "0.1.0"
        assert "ela" in data["detectors"]

    def test_health_detectors_all_enabled(self, client):
        data = client.get("/health").json()
        assert data["detectors"]["ela"] is True
        assert data["detectors"]["fft"] is True
        assert data["detectors"]["rppg"] is True
        assert data["detectors"]["upscale"] is True


# --- POST /verify ---

class TestVerifyUpload:
    def test_verify_jpeg(self, client, sample_jpeg_bytes):
        response = client.post(
            "/verify",
            files={"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["global_trust_score"] <= 1.0
        assert isinstance(data["anomalies_found"], list)
        assert data["processing_time_ms"] > 0
        assert data["heatmap_base64"] is None  # Not requested

    def test_verify_with_heatmap(self, client, sample_jpeg_bytes):
        response = client.post(
            "/verify?include_heatmap=true",
            files={"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["heatmap_base64"] is not None
        # Verify it's valid base64
        decoded = base64.b64decode(data["heatmap_base64"])
        assert len(decoded) > 0

    def test_verify_invalid_file(self, client):
        response = client.post(
            "/verify",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400


# --- POST /verify/base64 ---

class TestVerifyBase64:
    def test_verify_base64(self, client, sample_jpeg_base64):
        response = client.post(
            "/verify/base64",
            json={"image_base64": sample_jpeg_base64},
        )
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["global_trust_score"] <= 1.0

    def test_verify_base64_invalid(self, client):
        response = client.post(
            "/verify/base64",
            json={"image_base64": "not_valid_base64!!!"},
        )
        assert response.status_code == 400

    def test_verify_base64_with_heatmap(self, client, sample_jpeg_base64):
        response = client.post(
            "/verify/base64",
            json={"image_base64": sample_jpeg_base64, "include_heatmap": True},
        )
        assert response.status_code == 200
        assert response.json()["heatmap_base64"] is not None


# --- POST /verify/batch ---

class TestVerifyBatch:
    def test_batch_two_images(self, client, sample_jpeg_bytes):
        response = client.post(
            "/verify/batch",
            files=[
                ("files", ("img1.jpg", sample_jpeg_bytes, "image/jpeg")),
                ("files", ("img2.jpg", sample_jpeg_bytes, "image/jpeg")),
            ],
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["total_processing_time_ms"] > 0

    def test_batch_limit_exceeded(self, client, sample_jpeg_bytes):
        files = [
            ("files", (f"img{i}.jpg", sample_jpeg_bytes, "image/jpeg"))
            for i in range(11)
        ]
        response = client.post("/verify/batch", files=files)
        assert response.status_code == 400


# --- WebSocket ---

class TestWebSocketStream:
    def test_binary_frame(self, client, sample_jpeg_bytes):
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_bytes(sample_jpeg_bytes)
            data = ws.receive_json()
            assert 0.0 <= data["global_trust_score"] <= 1.0
            assert isinstance(data["anomalies_found"], list)
            assert data["frame_dropped"] is False

    def test_json_frame(self, client, sample_jpeg_base64):
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"image_base64": sample_jpeg_base64})
            data = ws.receive_json()
            assert 0.0 <= data["global_trust_score"] <= 1.0

    def test_invalid_json(self, client):
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"no_image": True})
            data = ws.receive_json()
            assert "error" in data

    def test_multiple_frames(self, client, sample_jpeg_bytes):
        with client.websocket_connect("/ws/stream") as ws:
            for _ in range(3):
                ws.send_bytes(sample_jpeg_bytes)
                data = ws.receive_json()
                assert "global_trust_score" in data
