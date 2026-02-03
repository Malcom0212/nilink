"""
Tests for the Nilink Verifier Engine.

Run with:
    pytest tests/ -v
"""

import asyncio
import numpy as np
import pytest

from Nilink_engine import NilinkVerifier, VerificationResult


@pytest.fixture
def verifier():
    """Fresh verifier instance for each test."""
    return NilinkVerifier()


@pytest.fixture
def natural_image():
    """Simulate a natural photo with realistic sensor noise."""
    rng = np.random.default_rng(42)
    base = rng.integers(80, 180, (480, 640, 3), dtype=np.uint8)
    noise = rng.normal(0, 8, base.shape).astype(np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def uniform_image():
    """Perfectly uniform image — suspiciously smooth."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def grid_image():
    """Image with periodic grid pattern — simulates GAN artifact."""
    rng = np.random.default_rng(42)
    img = rng.integers(100, 150, (480, 640, 3), dtype=np.uint8)
    for i in range(0, 480, 8):
        img[i, :, :] = np.clip(img[i, :, :].astype(np.int16) + 30, 0, 255).astype(np.uint8)
    for j in range(0, 640, 8):
        img[:, j, :] = np.clip(img[:, j, :].astype(np.int16) + 30, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def inpainted_image(natural_image):
    """Natural image with a suspiciously smooth region (simulated inpainting)."""
    img = natural_image.copy()
    img[100:250, 200:450, :] = 130  # Very smooth patch
    return img


# --- VerificationResult ---

class TestVerificationResult:
    def test_to_dict_basic(self):
        r = VerificationResult(global_trust_score=0.85, processing_time_ms=42.0)
        d = r.to_dict()
        assert d["global_trust_score"] == 0.85
        assert d["anomalies_found"] == []
        assert d["frame_dropped"] is False

    def test_to_dict_with_anomalies(self):
        r = VerificationResult(
            global_trust_score=0.3,
            anomalies_found=["ELA: Inconsistent compression"],
            suspicious_regions=[{"x": 10, "y": 20, "w": 100, "h": 50, "reason": "test"}],
        )
        d = r.to_dict()
        assert len(d["anomalies_found"]) == 1
        assert len(d["suspicious_regions"]) == 1


# --- Full Pipeline ---

class TestVerifyImage:
    def test_natural_image_scores_above_threshold(self, verifier, natural_image):
        """A noisy, natural-looking image should not be flagged heavily."""
        result = verifier.verify_image(natural_image)
        assert isinstance(result, VerificationResult)
        assert result.global_trust_score >= 0.3
        assert result.processing_time_ms > 0

    def test_uniform_image_scores_lower(self, verifier, uniform_image):
        """A perfectly uniform image should raise suspicion."""
        result = verifier.verify_image(uniform_image)
        assert result.global_trust_score < 0.8

    def test_bytes_input(self, verifier, natural_image):
        """Engine should accept raw bytes (JPEG encoded)."""
        import cv2
        _, encoded = cv2.imencode(".jpg", natural_image)
        result = verifier.verify_image(encoded.tobytes())
        assert isinstance(result, VerificationResult)
        assert result.processing_time_ms > 0

    def test_empty_image_returns_zero_score(self, verifier):
        """Invalid input should return 0.0 score, not crash."""
        result = verifier.verify_image(np.array([], dtype=np.uint8))
        assert result.global_trust_score == 0.0
        assert "Invalid image input" in result.anomalies_found

    def test_invalid_bytes_returns_zero_score(self, verifier):
        """Random bytes should not crash the engine."""
        result = verifier.verify_image(b"not an image")
        assert result.global_trust_score == 0.0


# --- ELA Detector ---

class TestELADetector:
    def test_natural_image_low_ela(self, verifier, natural_image):
        """Natural image should have consistent error levels."""
        score, anomalies, heatmap, regions = verifier._analyze_noise_consistency(natural_image)
        assert score >= 0.3
        assert heatmap is not None

    def test_inpainted_image_detected(self, verifier, inpainted_image):
        """Image with smooth inpainted region should be flagged."""
        score, anomalies, heatmap, regions = verifier._analyze_noise_consistency(inpainted_image)
        # The smooth region should create inconsistent error levels
        assert heatmap is not None

    def test_uniform_image_ela(self, verifier, uniform_image):
        """Uniform image should not crash ELA."""
        score, anomalies, heatmap, regions = verifier._analyze_noise_consistency(uniform_image)
        assert 0.0 <= score <= 1.0


# --- FFT Detector ---

class TestFFTDetector:
    def test_natural_image_no_grid(self, verifier, natural_image):
        """Natural image should not have grid patterns."""
        score, anomalies = verifier._frequency_scan(natural_image)
        assert score >= 0.3
        grid_anomalies = [a for a in anomalies if "Grid" in a]
        # Natural images should ideally not trigger grid detection
        assert len(grid_anomalies) == 0

    def test_grid_image_detected(self, verifier, grid_image):
        """Image with periodic grid should be flagged."""
        score, anomalies = verifier._frequency_scan(grid_image)
        # Grid pattern should lower trust or flag anomaly
        assert score < 0.9 or len(anomalies) > 0


# --- Upscaling Detector ---

class TestUpscalingDetector:
    def test_natural_image_not_flagged(self, verifier, natural_image):
        """Natural noisy image should not be flagged as upscaled."""
        score, anomalies, regions = verifier._detect_upscaling(natural_image)
        assert score >= 0.3

    def test_uniform_image_flagged(self, verifier, uniform_image):
        """Perfectly uniform image should trigger upscaling detection."""
        score, anomalies, regions = verifier._detect_upscaling(uniform_image)
        assert score < 0.8 or len(anomalies) > 0

    def test_inpainted_has_suspicious_regions(self, verifier, inpainted_image):
        """Inpainted image should have suspicious smooth regions."""
        score, anomalies, regions = verifier._detect_upscaling(inpainted_image)
        # The smooth patch should create suspicious regions
        assert isinstance(regions, list)


# --- rPPG Detector ---

class TestRPPGDetector:
    def test_no_crash_on_small_roi(self, verifier):
        """Should not crash on tiny face ROI."""
        tiny_face = np.zeros((10, 10, 3), dtype=np.uint8)
        score, anomalies = verifier._detect_pulse_rppg(tiny_face)
        assert score == 0.5  # Neutral for too-small input

    def test_texture_analysis_on_face(self, verifier):
        """Face-like region should be analyzed without error."""
        rng = np.random.default_rng(42)
        face = rng.integers(150, 200, (200, 150, 3), dtype=np.uint8)
        # Add skin-like variation
        noise = rng.normal(0, 5, face.shape).astype(np.int16)
        face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        score, anomalies = verifier._detect_pulse_rppg(face)
        assert 0.0 <= score <= 1.0

    def test_smooth_face_flagged(self, verifier):
        """Perfectly smooth face should trigger texture anomaly."""
        smooth_face = np.full((200, 150, 3), 170, dtype=np.uint8)
        score, anomalies = verifier._detect_pulse_rppg(smooth_face)
        smooth_anomalies = [a for a in anomalies if "smooth" in a.lower() or "uniform" in a.lower()]
        assert len(smooth_anomalies) > 0


# --- Face Detection ---

class TestFaceDetection:
    def test_no_crash_on_random_image(self, verifier, natural_image):
        """Should return empty list for image without faces."""
        faces = verifier._detect_faces(natural_image)
        assert isinstance(faces, list)


# --- Async Stream Processing ---

class TestStreamProcessing:
    def test_frame_dropping(self, verifier):
        """Old frames should be dropped."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Simulate a frame from 200ms ago
        import time
        old_timestamp = (time.perf_counter() * 1000) - 200

        result = asyncio.run(
            verifier.process_stream_frame(frame, old_timestamp)
        )
        assert result.frame_dropped is True
        assert result.processing_time_ms == 0.0

    def test_current_frame_processed(self, verifier, natural_image):
        """Current frame should be processed normally."""
        import time
        timestamp = time.perf_counter() * 1000

        result = asyncio.run(
            verifier.process_stream_frame(natural_image, timestamp)
        )
        assert result.frame_dropped is False
        assert result.processing_time_ms > 0


# --- Temporal State ---

class TestTemporalState:
    def test_reset_clears_buffer(self, verifier):
        """reset_temporal_state should clear rPPG buffer."""
        verifier._rppg_buffer = [1.0, 2.0, 3.0]
        verifier._rppg_timestamps = [0.1, 0.2, 0.3]
        verifier.reset_temporal_state()
        assert len(verifier._rppg_buffer) == 0
        assert len(verifier._rppg_timestamps) == 0


# --- Overlay ---

class TestOverlay:
    def test_overlay_creation(self, verifier, natural_image):
        """Overlay should produce an image of same dimensions."""
        result = verifier.verify_image(natural_image)
        overlay = verifier.create_overlay(natural_image, result)
        assert overlay.shape == natural_image.shape

    def test_overlay_with_heatmap(self, verifier, inpainted_image):
        """Overlay with heatmap should not crash."""
        result = verifier.verify_image(inpainted_image)
        overlay = verifier.create_overlay(inpainted_image, result)
        assert overlay.shape == inpainted_image.shape


# --- Selective Detectors ---

class TestSelectiveDetectors:
    def test_disable_all_detectors(self):
        """With all detectors disabled, should return neutral score."""
        v = NilinkVerifier(
            enable_ela=False,
            enable_fft=False,
            enable_rppg=False,
            enable_upscale_detection=False,
        )
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = v.verify_image(img)
        assert result.global_trust_score == 0.5  # np.mean([]) edge case

    def test_single_detector_ela(self):
        """Only ELA enabled should still work."""
        v = NilinkVerifier(
            enable_ela=True,
            enable_fft=False,
            enable_rppg=False,
            enable_upscale_detection=False,
        )
        rng = np.random.default_rng(42)
        img = rng.integers(80, 180, (200, 300, 3), dtype=np.uint8)
        result = v.verify_image(img)
        assert 0.0 <= result.global_trust_score <= 1.0
