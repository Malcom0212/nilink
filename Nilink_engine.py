"""
Nilink Verifier Engine
======================
Forensic analysis engine for detecting image/video manipulations in real-time.

Detects:
- Deepfakes (face swap, puppeteering) via rPPG analysis
- AI-generated images (GANs, diffusion models) via FFT spectral analysis
- Inpainting/erasure via Error Level Analysis (ELA)
- AI upscaling/restoration via noise consistency analysis

Author: Nilink Team
License: Proprietary
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Union
from io import BytesIO

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import fftpack


@dataclass
class VerificationResult:
    """Result of forensic analysis on an image/frame."""

    global_trust_score: float  # 0.0 (fake) to 1.0 (authentic)
    anomalies_found: list[str] = field(default_factory=list)
    manipulation_heatmap: Optional[NDArray[np.uint8]] = None
    suspicious_regions: list[dict] = field(default_factory=list)  # [{"x", "y", "w", "h", "reason"}]
    processing_time_ms: float = 0.0
    frame_dropped: bool = False

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        result = {
            "global_trust_score": round(self.global_trust_score, 4),
            "anomalies_found": self.anomalies_found,
            "suspicious_regions": self.suspicious_regions,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "frame_dropped": self.frame_dropped,
        }
        if self.manipulation_heatmap is not None:
            # Encode heatmap as base64 for JSON transport if needed
            result["heatmap_shape"] = list(self.manipulation_heatmap.shape)
        return result


class NilinkVerifier:
    """
    Real-time forensic verification engine for detecting image/video manipulations.

    Uses a hybrid approach combining:
    - Error Level Analysis (ELA) for inpainting/generation detection
    - FFT spectral analysis for GAN/diffusion model artifacts
    - Remote Photoplethysmography (rPPG) for deepfake/liveness detection
    - Noise pattern analysis for upscaling detection

    Designed for real-time processing (15+ FPS) without heavy ML dependencies.
    """

    # Configuration constants
    DEFAULT_MAX_LATENCY_MS = 66.0  # ~15 FPS threshold
    ELA_QUALITY = 92  # JPEG recompression quality for ELA
    FFT_GRID_THRESHOLD = 0.15  # Threshold for detecting periodic artifacts
    RPPG_MIN_FRAMES = 30  # Minimum frames needed for pulse detection
    NOISE_BLOCK_SIZE = 16  # Block size for noise analysis

    def __init__(
        self,
        max_latency_ms: float = DEFAULT_MAX_LATENCY_MS,
        enable_ela: bool = True,
        enable_fft: bool = True,
        enable_rppg: bool = True,
        enable_upscale_detection: bool = True,
    ):
        """
        Initialize the Nilink Verifier Engine.

        Args:
            max_latency_ms: Maximum allowed processing latency before frame dropping.
            enable_ela: Enable Error Level Analysis detector.
            enable_fft: Enable FFT spectral analysis detector.
            enable_rppg: Enable rPPG pulse detection (requires face ROI).
            enable_upscale_detection: Enable AI upscaling detection.
        """
        self.max_latency_ms = max_latency_ms
        self.enable_ela = enable_ela
        self.enable_fft = enable_fft
        self.enable_rppg = enable_rppg
        self.enable_upscale_detection = enable_upscale_detection

        # Face detection for rPPG (using OpenCV's DNN face detector)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # rPPG temporal buffer for pulse analysis
        self._rppg_buffer: list[float] = []
        self._rppg_timestamps: list[float] = []

        # Frame timing for latency management
        self._last_frame_time: float = 0.0

    def verify_image(self, image: Union[NDArray[np.uint8], bytes]) -> VerificationResult:
        """
        Synchronous verification of a single image.

        Args:
            image: Input image as numpy array (BGR) or raw bytes.

        Returns:
            VerificationResult with trust score, anomalies, and heatmap.
        """
        start_time = time.perf_counter()

        # Convert bytes to numpy array if needed
        if isinstance(image, bytes):
            image = self._decode_image(image)

        if image is None or image.size == 0:
            return VerificationResult(
                global_trust_score=0.0,
                anomalies_found=["Invalid image input"],
            )

        anomalies: list[str] = []
        scores: list[float] = []
        heatmaps: list[NDArray[np.uint8]] = []
        regions: list[dict] = []

        # Run enabled detectors
        if self.enable_ela:
            ela_score, ela_anomalies, ela_heatmap, ela_regions = self._analyze_noise_consistency(image)
            scores.append(ela_score)
            anomalies.extend(ela_anomalies)
            if ela_heatmap is not None:
                heatmaps.append(ela_heatmap)
            regions.extend(ela_regions)

        if self.enable_fft:
            fft_score, fft_anomalies = self._frequency_scan(image)
            scores.append(fft_score)
            anomalies.extend(fft_anomalies)

        if self.enable_rppg:
            faces = self._detect_faces(image)
            if faces:
                # Analyze the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                face_roi = image[y:y+h, x:x+w]
                rppg_score, rppg_anomalies = self._detect_pulse_rppg(face_roi)
                scores.append(rppg_score)
                anomalies.extend(rppg_anomalies)

        if self.enable_upscale_detection:
            upscale_score, upscale_anomalies, upscale_regions = self._detect_upscaling(image)
            scores.append(upscale_score)
            anomalies.extend(upscale_anomalies)
            regions.extend(upscale_regions)

        # Calculate global trust score (weighted average)
        global_score = np.mean(scores) if scores else 0.5

        # Merge heatmaps
        combined_heatmap = self._merge_heatmaps(heatmaps, image.shape[:2]) if heatmaps else None

        processing_time = (time.perf_counter() - start_time) * 1000

        return VerificationResult(
            global_trust_score=float(global_score),
            anomalies_found=anomalies,
            manipulation_heatmap=combined_heatmap,
            suspicious_regions=regions,
            processing_time_ms=processing_time,
        )

    async def process_stream_frame(
        self,
        frame: NDArray[np.uint8],
        timestamp_ms: Optional[float] = None,
    ) -> VerificationResult:
        """
        Asynchronous processing of a video stream frame with latency management.

        Implements frame dropping logic: if the frame is older than max_latency_ms,
        it is skipped to prevent latency accumulation.

        Args:
            frame: Video frame as numpy array (BGR).
            timestamp_ms: Frame timestamp in milliseconds. If None, uses current time.

        Returns:
            VerificationResult (may have frame_dropped=True if skipped).
        """
        current_time = time.perf_counter() * 1000

        if timestamp_ms is None:
            timestamp_ms = current_time

        # Frame dropping logic: skip if frame is too old
        latency = current_time - timestamp_ms
        if latency > self.max_latency_ms:
            return VerificationResult(
                global_trust_score=0.5,  # Neutral score for dropped frames
                anomalies_found=[],
                frame_dropped=True,
                processing_time_ms=0.0,
            )

        # Process frame asynchronously
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.verify_image, frame)

        # Update rPPG buffer with timestamp for temporal analysis
        self._last_frame_time = current_time

        return result

    def _decode_image(self, data: bytes) -> Optional[NDArray[np.uint8]]:
        """Decode image from bytes (JPEG, PNG, etc.)."""
        try:
            nparr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _analyze_noise_consistency(
        self,
        image: NDArray[np.uint8],
    ) -> tuple[float, list[str], Optional[NDArray[np.uint8]], list[dict]]:
        """
        Error Level Analysis (ELA) for detecting inpainting and AI-generated regions.

        Forensic principle: When an image is recompressed at a specific JPEG quality,
        the compression error is relatively uniform across authentic regions. AI-generated
        or inpainted areas typically show different error levels because:

        1. They lack natural sensor noise (CCD/CMOS noise patterns)
        2. They may have been compressed at different quality levels
        3. Generative models produce "too smooth" textures

        The method compares the original image with a recompressed version.
        Regions with significantly different error levels are flagged as suspicious.

        Args:
            image: Input BGR image.

        Returns:
            Tuple of (trust_score, anomalies, heatmap, suspicious_regions).
        """
        try:
            # Recompress image in memory
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.ELA_QUALITY]
            _, encoded = cv2.imencode(".jpg", image, encode_params)
            recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

            if recompressed is None:
                return 1.0, [], None, []

            # Calculate absolute difference (error level)
            diff = cv2.absdiff(image, recompressed)

            # Convert to grayscale and amplify for visibility
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Amplify the difference (scale to 0-255 range)
            ela_image = cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX)
            ela_image = ela_image.astype(np.uint8)

            # Statistical analysis of error distribution
            mean_error = np.mean(ela_image)
            std_error = np.std(ela_image)

            # Detect anomalous regions (error > mean + 2*std)
            threshold = mean_error + 2 * std_error
            anomaly_mask = ela_image > threshold

            anomalies: list[str] = []
            regions: list[dict] = []

            # Find contours of suspicious regions
            contours, _ = cv2.findContours(
                anomaly_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            # Filter significant contours
            min_area = image.shape[0] * image.shape[1] * 0.01  # 1% of image
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]

            if significant_contours:
                anomalies.append("ELA: Inconsistent compression levels detected")
                for contour in significant_contours[:5]:  # Limit to 5 regions
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "reason": "Inconsistent error level",
                    })

            # Calculate trust score based on error uniformity
            # High std relative to mean indicates potential manipulation
            if mean_error < 1.0:
                # Nearly zero error means image has no compression artifacts at all
                # This is suspicious: real photos always have some JPEG noise
                trust_score = 0.3
            elif mean_error > 0:
                coefficient_of_variation = std_error / mean_error
                # CV > 1.5 is suspicious, < 0.5 is likely authentic
                trust_score = max(0.0, min(1.0, 1.0 - (coefficient_of_variation - 0.5) / 1.5))
            else:
                trust_score = 0.3

            # Create colored heatmap
            heatmap = cv2.applyColorMap(ela_image, cv2.COLORMAP_JET)

            return trust_score, anomalies, heatmap, regions

        except Exception:
            # Fail silently to not crash the video stream
            return 1.0, [], None, []

    def _frequency_scan(
        self,
        image: NDArray[np.uint8],
    ) -> tuple[float, list[str]]:
        """
        FFT spectral analysis for detecting GAN and diffusion model artifacts.

        Forensic principle: Generative AI models (GANs, diffusion models) often
        produce images with characteristic periodic artifacts that are invisible
        to the human eye but detectable in the frequency domain:

        1. GANs using transposed convolutions create "checkerboard" patterns
        2. Diffusion models may have regular grid-like artifacts from the denoising process
        3. These artifacts appear as peaks at specific frequencies in the FFT spectrum

        The method converts the image to frequency domain and analyzes the
        magnitude spectrum for periodic patterns that indicate synthetic origin.

        Args:
            image: Input BGR image.

        Returns:
            Tuple of (trust_score, anomalies).
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Apply 2D FFT
            f_transform = fftpack.fft2(gray)
            f_shift = fftpack.fftshift(f_transform)

            # Calculate magnitude spectrum (log scale for better visualization)
            magnitude = np.abs(f_shift)
            magnitude_log = np.log1p(magnitude)

            # Analyze high-frequency region for periodic artifacts
            h, w = magnitude_log.shape
            center_y, center_x = h // 2, w // 2

            # Create mask for high-frequency analysis (exclude DC and low frequencies)
            y_coords, x_coords = np.ogrid[:h, :w]
            distance_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)

            # High frequency band (outer 30% of spectrum)
            min_radius = min(h, w) * 0.35
            max_radius = min(h, w) * 0.5
            hf_mask = (distance_from_center >= min_radius) & (distance_from_center <= max_radius)

            hf_magnitude = magnitude_log[hf_mask]

            if len(hf_magnitude) == 0:
                return 1.0, []

            # Statistical analysis of high-frequency components
            hf_mean = np.mean(hf_magnitude)
            hf_std = np.std(hf_magnitude)
            hf_max = np.max(hf_magnitude)

            anomalies: list[str] = []

            # Detect periodic peaks (potential GAN artifacts)
            # Peaks significantly above mean indicate regular patterns
            peak_threshold = hf_mean + 3 * hf_std
            num_peaks = np.sum(hf_magnitude > peak_threshold)
            peak_ratio = num_peaks / len(hf_magnitude)

            if peak_ratio > self.FFT_GRID_THRESHOLD:
                anomalies.append("FFT: Grid pattern detected (possible GAN artifact)")

            # Check for suspiciously uniform high-frequency distribution
            # Real images have varied high-frequency content; AI images may be too uniform
            if hf_std < hf_mean * 0.3:
                anomalies.append("FFT: Unusually uniform frequency distribution")

            # Calculate trust score
            # High peak ratio = low trust, high uniformity = low trust
            peak_penalty = min(1.0, peak_ratio / self.FFT_GRID_THRESHOLD)
            uniformity_penalty = max(0.0, 1.0 - (hf_std / (hf_mean * 0.5))) if hf_mean > 0 else 0

            trust_score = max(0.0, 1.0 - (peak_penalty * 0.5 + uniformity_penalty * 0.5))

            return trust_score, anomalies

        except Exception:
            return 1.0, []

    def _detect_faces(self, image: NDArray[np.uint8]) -> list[tuple[int, int, int, int]]:
        """
        Detect faces in the image using OpenCV Haar cascades.

        Args:
            image: Input BGR image.

        Returns:
            List of face bounding boxes as (x, y, width, height).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            return [tuple(face) for face in faces]
        except Exception:
            return []

    def _detect_pulse_rppg(
        self,
        face_roi: NDArray[np.uint8],
    ) -> tuple[float, list[str]]:
        """
        Remote Photoplethysmography (rPPG) analysis for deepfake/liveness detection.

        Forensic principle: Real human faces exhibit subtle color variations
        caused by blood flow (pulse). These variations are most visible in the
        green channel and follow a periodic pattern (60-100 BPM typically).

        Deepfakes and AI-generated faces typically fail to reproduce these
        micro-variations because:

        1. The source video may not capture the original pulse
        2. Face-swapping disrupts the temporal coherence of skin color
        3. AI generation doesn't model cardiovascular physiology

        For single-frame analysis, we analyze spatial color consistency
        in the forehead region where pulse is most visible.

        Args:
            face_roi: Cropped face region (BGR).

        Returns:
            Tuple of (trust_score, anomalies).
        """
        try:
            if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                return 0.5, []  # Neutral score if face too small

            # Extract forehead region (top 30% of face)
            h, w = face_roi.shape[:2]
            forehead = face_roi[0:int(h * 0.3), int(w * 0.2):int(w * 0.8)]

            if forehead.size == 0:
                return 0.5, []

            # Extract green channel (most sensitive to blood oxygen)
            green_channel = forehead[:, :, 1].astype(np.float32)

            # Analyze spatial uniformity of green channel
            # Real skin has natural variation; AI-generated may be too uniform
            mean_green = np.mean(green_channel)
            std_green = np.std(green_channel)

            # Calculate local variance using block-based analysis
            block_size = max(4, min(forehead.shape[:2]) // 8)
            local_vars = []

            for i in range(0, forehead.shape[0] - block_size, block_size):
                for j in range(0, forehead.shape[1] - block_size, block_size):
                    block = green_channel[i:i+block_size, j:j+block_size]
                    local_vars.append(np.var(block))

            if not local_vars:
                return 0.5, []

            # Variance of local variances - real skin has varied texture
            var_of_vars = np.var(local_vars)
            mean_local_var = np.mean(local_vars)

            anomalies: list[str] = []

            # Suspiciously uniform skin texture
            if mean_local_var < 10:  # Very low local variance
                anomalies.append("rPPG: Unnaturally smooth skin texture")

            # Suspiciously consistent variance across regions
            if var_of_vars < mean_local_var * 0.1 and mean_local_var > 0:
                anomalies.append("rPPG: Uniform texture pattern (possible AI generation)")

            # Store green channel mean for temporal analysis (video mode)
            self._rppg_buffer.append(mean_green)
            self._rppg_timestamps.append(time.perf_counter())

            # Keep buffer limited
            max_buffer = self.RPPG_MIN_FRAMES * 2
            if len(self._rppg_buffer) > max_buffer:
                self._rppg_buffer = self._rppg_buffer[-max_buffer:]
                self._rppg_timestamps = self._rppg_timestamps[-max_buffer:]

            # If we have enough frames, do temporal pulse analysis
            if len(self._rppg_buffer) >= self.RPPG_MIN_FRAMES:
                pulse_score, pulse_anomalies = self._analyze_pulse_signal()
                anomalies.extend(pulse_anomalies)

                # Weight: 70% pulse analysis, 30% texture analysis
                texture_score = 1.0 if not anomalies else 0.5
                trust_score = pulse_score * 0.7 + texture_score * 0.3
            else:
                # Single frame: texture analysis only
                if anomalies:
                    trust_score = 0.5
                else:
                    trust_score = 0.7  # Neutral-positive without temporal data

            return trust_score, anomalies

        except Exception:
            return 0.5, []

    def _analyze_pulse_signal(self) -> tuple[float, list[str]]:
        """
        Analyze accumulated rPPG buffer for pulse detection.

        A valid pulse signal should show periodic variation in the
        60-100 BPM range (1-1.67 Hz).
        """
        try:
            if len(self._rppg_buffer) < self.RPPG_MIN_FRAMES:
                return 0.5, []

            signal = np.array(self._rppg_buffer)
            timestamps = np.array(self._rppg_timestamps)

            # Detrend signal
            signal = signal - np.mean(signal)

            # Estimate sampling rate
            if len(timestamps) > 1:
                avg_interval = np.mean(np.diff(timestamps))
                if avg_interval > 0:
                    sample_rate = 1.0 / avg_interval
                else:
                    sample_rate = 30.0  # Default assumption
            else:
                sample_rate = 30.0

            # FFT of the signal
            n = len(signal)
            freqs = fftpack.fftfreq(n, 1.0 / sample_rate)
            fft_signal = fftpack.fft(signal)
            magnitude = np.abs(fft_signal)

            # Focus on pulse frequency range (0.8-2.5 Hz = 48-150 BPM)
            pulse_band = (freqs >= 0.8) & (freqs <= 2.5)

            if not np.any(pulse_band):
                return 0.5, []

            pulse_magnitude = magnitude[pulse_band]
            pulse_freqs = freqs[pulse_band]

            anomalies: list[str] = []

            # Check for dominant frequency in pulse range
            if len(pulse_magnitude) > 0:
                max_idx = np.argmax(pulse_magnitude)
                peak_freq = pulse_freqs[max_idx]
                peak_power = pulse_magnitude[max_idx]
                total_power = np.sum(magnitude[freqs > 0])

                if total_power > 0:
                    pulse_ratio = peak_power / total_power

                    # Strong pulse signal indicates real human
                    if pulse_ratio > 0.1:
                        bpm = peak_freq * 60
                        if 50 < bpm < 120:
                            # Valid pulse detected
                            return 0.9, []
                        else:
                            anomalies.append(f"rPPG: Unusual pulse rate ({bpm:.0f} BPM)")
                            return 0.6, anomalies
                    else:
                        anomalies.append("rPPG: No clear pulse signal detected")
                        return 0.4, anomalies

            return 0.5, anomalies

        except Exception:
            return 0.5, []

    def _detect_upscaling(
        self,
        image: NDArray[np.uint8],
    ) -> tuple[float, list[str], list[dict]]:
        """
        Detect AI upscaling/restoration artifacts.

        Forensic principle: AI upscaling (ESRGAN, Real-ESRGAN, etc.) "hallucinates"
        high-frequency details that weren't in the original image. These invented
        details have characteristics that differ from genuine high-resolution captures:

        1. Noise patterns don't match typical camera sensor noise (Gaussian, Poisson)
        2. Texture repetition - AI may repeat similar patterns across the image
        3. Edge artifacts - unnaturally sharp or smooth edges
        4. Block artifacts from tile-based processing

        This method analyzes local noise statistics and texture patterns.

        Args:
            image: Input BGR image.

        Returns:
            Tuple of (trust_score, anomalies, suspicious_regions).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # High-pass filter to isolate noise/texture
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            high_pass = gray - blurred

            # Analyze noise in blocks
            h, w = gray.shape
            block_size = self.NOISE_BLOCK_SIZE

            block_stats = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = high_pass[i:i+block_size, j:j+block_size]
                    block_stats.append({
                        "x": j,
                        "y": i,
                        "std": float(np.std(block)),
                        "mean": float(np.mean(np.abs(block))),
                    })

            if not block_stats:
                return 1.0, [], []

            # Analyze distribution of noise levels
            stds = np.array([b["std"] for b in block_stats])
            global_std = np.std(stds)
            global_mean_std = np.mean(stds)

            anomalies: list[str] = []
            regions: list[dict] = []

            # Image with virtually no noise is suspicious (AI-generated or uniform)
            if global_mean_std < 1.0:
                anomalies.append("Upscale: Image is suspiciously smooth (no sensor noise)")
                trust_score = 0.2
                return trust_score, anomalies, regions

            # AI upscaling often produces very uniform noise
            if global_std < global_mean_std * 0.3:
                anomalies.append("Upscale: Unnaturally uniform noise pattern")

            # Detect blocks with significantly different noise (processing boundaries)
            threshold_high = global_mean_std + 2.5 * global_std
            threshold_low = max(0, global_mean_std - 2.5 * global_std)

            for block in block_stats:
                if block["std"] > threshold_high or block["std"] < threshold_low:
                    if block["std"] < 0.5:  # Very smooth - suspicious
                        regions.append({
                            "x": block["x"],
                            "y": block["y"],
                            "w": block_size,
                            "h": block_size,
                            "reason": "Suspiciously smooth region",
                        })

            # Limit reported regions
            if len(regions) > 10:
                anomalies.append(f"Upscale: {len(regions)} suspicious smooth regions detected")
                regions = regions[:5]
            elif regions:
                anomalies.append("Upscale: Inconsistent noise levels detected")

            # Calculate trust score
            # Natural sensor noise: moderate mean_std with some variation (global_std)
            # AI upscaling: low mean_std OR very uniform noise (low global_std relative to mean)
            noise_level_score = min(1.0, global_mean_std / 3.0)  # Penalize very low noise
            variation_score = min(1.0, global_std / (global_mean_std * 0.3)) if global_mean_std > 0 else 0.0
            uniformity_score = noise_level_score * 0.6 + variation_score * 0.4
            region_penalty = min(0.3, len(regions) * 0.03)

            trust_score = max(0.0, uniformity_score - region_penalty)

            return trust_score, anomalies, regions

        except Exception:
            return 1.0, [], []

    def _merge_heatmaps(
        self,
        heatmaps: list[NDArray[np.uint8]],
        target_shape: tuple[int, int],
    ) -> NDArray[np.uint8]:
        """Merge multiple heatmaps into a single visualization."""
        if not heatmaps:
            return np.zeros((*target_shape, 3), dtype=np.uint8)

        # Resize all heatmaps to target shape and average
        resized = []
        for hm in heatmaps:
            if hm.shape[:2] != target_shape:
                hm = cv2.resize(hm, (target_shape[1], target_shape[0]))
            resized.append(hm.astype(np.float32))

        merged = np.mean(resized, axis=0).astype(np.uint8)
        return merged

    def reset_temporal_state(self) -> None:
        """Reset temporal buffers (call when switching video sources)."""
        self._rppg_buffer.clear()
        self._rppg_timestamps.clear()
        self._last_frame_time = 0.0

    def create_overlay(
        self,
        image: NDArray[np.uint8],
        result: VerificationResult,
        alpha: float = 0.4,
    ) -> NDArray[np.uint8]:
        """
        Create visualization overlay with heatmap and annotations.

        Args:
            image: Original image.
            result: Verification result.
            alpha: Heatmap transparency (0-1).

        Returns:
            Image with overlay visualization.
        """
        output = image.copy()

        # Overlay heatmap if available
        if result.manipulation_heatmap is not None:
            heatmap = result.manipulation_heatmap
            if heatmap.shape[:2] != image.shape[:2]:
                heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            output = cv2.addWeighted(output, 1 - alpha, heatmap, alpha, 0)

        # Draw suspicious regions
        for region in result.suspicious_regions:
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Add trust score indicator
        score = result.global_trust_score
        color = (
            int(255 * (1 - score)),  # Red component
            int(255 * score),         # Green component
            0,
        )
        cv2.putText(
            output,
            f"Trust: {score:.1%}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        # Add anomaly count
        if result.anomalies_found:
            cv2.putText(
                output,
                f"Anomalies: {len(result.anomalies_found)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return output


async def demo_webcam(verifier: NilinkVerifier) -> None:
    """Demo: Real-time webcam analysis."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Nilink Verifier - Webcam Demo")
    print("Press 'q' to quit, 'r' to reset temporal state")
    print("-" * 40)

    frame_count = 0
    fps_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            timestamp_ms = time.perf_counter() * 1000
            result = await verifier.process_stream_frame(frame, timestamp_ms)

            # Create visualization
            display = verifier.create_overlay(frame, result)

            # Calculate FPS
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.perf_counter()
                cv2.putText(
                    display,
                    f"FPS: {fps:.1f}",
                    (display.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # Show anomalies
            y_offset = 90
            for anomaly in result.anomalies_found[:3]:
                cv2.putText(
                    display,
                    anomaly[:50],
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                )
                y_offset += 20

            cv2.imshow("Nilink Verifier", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                verifier.reset_temporal_state()
                print("Temporal state reset")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def demo_image(verifier: NilinkVerifier, image_path: str) -> None:
    """Demo: Single image analysis."""
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return

    print(f"Analyzing: {image_path}")
    print("-" * 40)

    result = verifier.verify_image(image)

    print(f"Trust Score: {result.global_trust_score:.1%}")
    print(f"Processing Time: {result.processing_time_ms:.2f}ms")

    if result.anomalies_found:
        print("\nAnomalies detected:")
        for anomaly in result.anomalies_found:
            print(f"  - {anomaly}")
    else:
        print("\nNo anomalies detected")

    if result.suspicious_regions:
        print(f"\nSuspicious regions: {len(result.suspicious_regions)}")

    # Show visualization
    display = verifier.create_overlay(image, result)
    cv2.imshow("Nilink Analysis", display)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_synthetic() -> None:
    """Demo: Generate synthetic test images and analyze them."""
    verifier = NilinkVerifier()

    print("Nilink Verifier - Synthetic Test Demo")
    print("=" * 40)

    # Test 1: Natural-looking gradient (should score high)
    print("\n[Test 1] Natural gradient image...")
    natural = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    # Add some natural variation
    noise = np.random.normal(0, 10, natural.shape).astype(np.int16)
    natural = np.clip(natural.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    result = verifier.verify_image(natural)
    print(f"  Trust Score: {result.global_trust_score:.1%}")
    print(f"  Anomalies: {result.anomalies_found}")

    # Test 2: Perfectly uniform image (suspicious - too smooth)
    print("\n[Test 2] Perfectly uniform image (suspicious)...")
    uniform = np.full((480, 640, 3), 128, dtype=np.uint8)

    result = verifier.verify_image(uniform)
    print(f"  Trust Score: {result.global_trust_score:.1%}")
    print(f"  Anomalies: {result.anomalies_found}")

    # Test 3: Image with grid pattern (simulating GAN artifact)
    print("\n[Test 3] Grid pattern image (GAN-like artifact)...")
    grid = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    # Add periodic pattern
    for i in range(0, 480, 8):
        grid[i, :, :] = grid[i, :, :] + 20
    for j in range(0, 640, 8):
        grid[:, j, :] = grid[:, j, :] + 20
    grid = np.clip(grid, 0, 255).astype(np.uint8)

    result = verifier.verify_image(grid)
    print(f"  Trust Score: {result.global_trust_score:.1%}")
    print(f"  Anomalies: {result.anomalies_found}")

    # Test 4: Image with one manipulated region
    print("\n[Test 4] Image with manipulated region...")
    manipulated = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    noise = np.random.normal(0, 10, manipulated.shape).astype(np.int16)
    manipulated = np.clip(manipulated.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Add a very smooth region (like AI inpainting)
    manipulated[100:200, 200:400, :] = 130

    result = verifier.verify_image(manipulated)
    print(f"  Trust Score: {result.global_trust_score:.1%}")
    print(f"  Anomalies: {result.anomalies_found}")
    print(f"  Suspicious regions: {len(result.suspicious_regions)}")

    print("\n" + "=" * 40)
    print("Synthetic tests complete.")


if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("  NILINK VERIFIER ENGINE")
    print("  Forensic Analysis for Image/Video Authenticity")
    print("=" * 50)

    verifier = NilinkVerifier()

    if len(sys.argv) > 1:
        # Analyze provided image
        demo_image(verifier, sys.argv[1])
    else:
        # Run synthetic demo first
        demo_synthetic()

        # Then offer webcam demo
        print("\n" + "-" * 50)
        response = input("Run webcam demo? (y/n): ").strip().lower()
        if response == "y":
            asyncio.run(demo_webcam(verifier))
