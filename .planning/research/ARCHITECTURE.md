# Architecture Patterns: Forensic Image/Video Manipulation Detection Systems

**Domain:** Real-time Forensic Manipulation Detection
**Researched:** 2026-02-02
**Confidence:** MEDIUM (based on verified 2025-2026 research with cross-referenced sources)

## Executive Summary

Modern forensic manipulation detection systems (2025-2026) employ **dual-stream architectures** that combine spatial and frequency domain analysis, orchestrated by an async pipeline manager. The architecture prioritizes:

1. **Multi-modal detection** - Multiple independent detectors (ELA, FFT, rPPG, upscaling) running in parallel
2. **Async processing** - Non-blocking I/O for handling concurrent video streams
3. **Frame dropping strategies** - Intelligent buffer management to maintain real-time latency (<67ms for 15 FPS)
4. **Result fusion** - Aggregating detector outputs for final classification

For NilinkVerifierEngine, the recommended architecture is a **detector registry pattern** with async orchestration.

---

## Recommended Architecture for NilinkVerifierEngine

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│  ┌──────────────────┐              ┌──────────────────────────┐ │
│  │   REST Endpoint  │              │  WebSocket Endpoint      │ │
│  │   (Images)       │              │  (Video Streams)         │ │
│  └────────┬─────────┘              └──────────┬───────────────┘ │
└───────────┼────────────────────────────────────┼─────────────────┘
            │                                    │
            ▼                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                   Pipeline Orchestrator                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  - Frame buffer manager (drop policy)                       │ │
│  │  - Detector scheduler (parallel execution)                  │ │
│  │  - Result aggregator (fusion logic)                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────┬───────────────────────────────────────────────────────┘
            │
            ├───────────┬──────────┬──────────┬──────────────┐
            ▼           ▼          ▼          ▼              ▼
    ┌───────────┐ ┌─────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐
    │    ELA    │ │   FFT   │ │  rPPG  │ │ Upscaling│ │  Future  │
    │ Detector  │ │Detector │ │Detector│ │ Detector │ │ Detectors│
    └─────┬─────┘ └────┬────┘ └───┬────┘ └────┬─────┘ └────┬─────┘
          │            │          │           │            │
          └────────────┴──────────┴───────────┴────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Result Fusion   │
                    │   (Confidence    │
                    │   Aggregation)   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Response Model  │
                    │  (JSON/WebSocket)│
                    └──────────────────┘
```

### Component Boundaries

| Component | Responsibility | Inputs | Outputs | Communicates With |
|-----------|---------------|---------|---------|-------------------|
| **API Layer** | Accept REST/WebSocket requests, validate input, return responses | HTTP request / WebSocket frames | JSON responses / WebSocket messages | Pipeline Orchestrator |
| **Pipeline Orchestrator** | Manage async detector execution, frame buffer, result aggregation | Image/video frames | Detection results (dict) | All Detectors, Result Fusion |
| **Frame Buffer Manager** | Maintain bounded queue, drop frames when overloaded | Raw frames from stream | Frames for processing | Pipeline Orchestrator |
| **Detector Scheduler** | Spawn async tasks for each detector, collect results | Frame + detector registry | Detector outputs (list) | All Detectors |
| **ELA Detector** | Compute JPEG error level analysis | Image (PIL/numpy) | Confidence score (0-1) | Orchestrator |
| **FFT Detector** | Analyze frequency domain artifacts | Image (numpy array) | Confidence score (0-1) | Orchestrator |
| **rPPG Detector** | Extract heartbeat signal inconsistencies (video only) | Frame sequence (list) | Confidence score (0-1) | Orchestrator |
| **Upscaling Detector** | Detect resampling artifacts | Image (numpy array) | Confidence score (0-1) | Orchestrator |
| **Result Fusion** | Aggregate detector scores into final verdict | List of detector results | Final confidence + verdict | API Layer |

---

## Data Flow

### 1. Image Processing Flow (REST API)

```
Client Upload (POST /detect/image)
    │
    ├─> [1] API validates image format/size
    │
    ├─> [2] Pipeline Orchestrator receives image
    │
    ├─> [3] Spawn async tasks for detectors:
    │       ├─> ELA Detector (process concurrently)
    │       ├─> FFT Detector (process concurrently)
    │       └─> Upscaling Detector (process concurrently)
    │
    ├─> [4] Await all detector results (asyncio.gather)
    │
    ├─> [5] Result Fusion aggregates scores
    │       └─> Weighted average or voting logic
    │
    └─> [6] Return JSON response:
            {
              "is_manipulated": true/false,
              "confidence": 0.87,
              "detectors": {
                "ela": {"score": 0.92, "status": "detected"},
                "fft": {"score": 0.85, "status": "detected"},
                "upscaling": {"score": 0.84, "status": "detected"}
              },
              "processing_time_ms": 156
            }
```

### 2. Video Stream Processing Flow (WebSocket)

```
Client WebSocket Connection (ws://host/detect/stream)
    │
    ├─> [1] Establish persistent connection
    │
    ├─> [2] Client streams video frames (binary)
    │
    ├─> [3] Frame Buffer Manager:
    │       ├─> Add frame to bounded queue (max size: 10)
    │       ├─> If queue full: DROP oldest frame (maintain latency)
    │       └─> Emit frame for processing
    │
    ├─> [4] Pipeline Orchestrator (per frame):
    │       ├─> Spawn async detector tasks:
    │       │   ├─> ELA Detector
    │       │   ├─> FFT Detector
    │       │   ├─> rPPG Detector (requires frame history: last 30 frames)
    │       │   └─> Upscaling Detector
    │       │
    │       └─> Detectors execute concurrently
    │
    ├─> [5] Result Fusion per frame
    │
    ├─> [6] Send result back via WebSocket:
    │       {
    │         "frame_id": 142,
    │         "is_manipulated": false,
    │         "confidence": 0.23,
    │         "latency_ms": 58,
    │         "dropped_frames": 2
    │       }
    │
    └─> [7] Repeat for each incoming frame
```

### 3. Frame History Management (for rPPG)

```
rPPG Detector requires temporal context:
    │
    ├─> Maintain sliding window buffer (30 frames ~ 2 seconds at 15 FPS)
    │
    ├─> On new frame arrival:
    │   ├─> Add to buffer
    │   └─> Remove oldest if buffer exceeds 30
    │
    └─> Extract rPPG signal from face regions across buffer
        └─> Analyze frequency consistency (FFT of color changes)
```

---

## Patterns to Follow

### Pattern 1: Detector Registry Pattern

**What:** Dynamically register detectors at runtime without hardcoding dependencies.

**When:** You need to add/remove detectors without modifying orchestrator code.

**Benefits:**
- Easy to add new detectors
- Detectors can be enabled/disabled via config
- Supports A/B testing of detector versions

**Example:**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import asyncio

class BaseDetector(ABC):
    """Base class for all manipulation detectors."""

    @abstractmethod
    async def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Returns: {
            "score": float (0-1),
            "status": str ("clean" | "detected" | "uncertain"),
            "metadata": dict (optional detector-specific data)
        }
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class DetectorRegistry:
    """Registry for all available detectors."""

    def __init__(self):
        self._detectors: Dict[str, BaseDetector] = {}

    def register(self, detector: BaseDetector):
        self._detectors[detector.name] = detector

    async def run_all(self, image: np.ndarray) -> Dict[str, Dict]:
        """Run all registered detectors concurrently."""
        tasks = {
            name: detector.detect(image)
            for name, detector in self._detectors.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return dict(zip(tasks.keys(), results))


# Usage
registry = DetectorRegistry()
registry.register(ELADetector())
registry.register(FFTDetector())
registry.register(UpscalingDetector())

results = await registry.run_all(image)
```

### Pattern 2: Bounded Frame Buffer with Drop Policy

**What:** Fixed-size queue that drops frames when full to prevent latency buildup.

**When:** Processing video streams where maintaining real-time latency is critical.

**Benefits:**
- Prevents unbounded memory growth
- Guarantees latency upper bound
- Client aware of dropped frames

**Example:**

```python
from collections import deque
from dataclasses import dataclass
import asyncio

@dataclass
class Frame:
    id: int
    data: bytes
    timestamp: float

class BoundedFrameBuffer:
    """Frame buffer with automatic dropping."""

    def __init__(self, max_size: int = 10):
        self.buffer = deque(maxlen=max_size)
        self.dropped_count = 0

    def add_frame(self, frame: Frame) -> bool:
        """
        Add frame to buffer. Returns True if frame was added,
        False if it was dropped.
        """
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()  # Drop oldest
            self.dropped_count += 1
            return False

        self.buffer.append(frame)
        return True

    async def get_frame(self) -> Frame:
        """Get next frame for processing."""
        while len(self.buffer) == 0:
            await asyncio.sleep(0.01)  # Wait for frames
        return self.buffer.popleft()
```

### Pattern 3: Result Fusion with Weighted Voting

**What:** Combine multiple detector outputs into single confidence score.

**When:** You have multiple detectors with varying reliability/sensitivity.

**Benefits:**
- Reduces false positives from single detectors
- Balances precision vs recall
- Configurable via weights

**Example:**

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class DetectorConfig:
    name: str
    weight: float  # 0.0 - 1.0
    threshold: float  # Score above which detection is positive

class ResultFusion:
    """Fuse multiple detector results into final verdict."""

    def __init__(self, configs: List[DetectorConfig]):
        self.configs = {cfg.name: cfg for cfg in configs}
        self._validate_weights()

    def _validate_weights(self):
        total = sum(cfg.weight for cfg in self.configs.values())
        assert abs(total - 1.0) < 0.01, "Weights must sum to 1.0"

    def fuse(self, results: Dict[str, Dict]) -> Dict:
        """
        Aggregate detector results into final verdict.

        Args:
            results: {detector_name: {score, status, ...}}

        Returns:
            {
                "is_manipulated": bool,
                "confidence": float,
                "detectors": original results
            }
        """
        weighted_sum = 0.0

        for name, result in results.items():
            if name not in self.configs:
                continue  # Skip unknown detectors

            cfg = self.configs[name]
            score = result.get("score", 0.0)
            weighted_sum += score * cfg.weight

        # Final verdict: confidence > 0.5 = manipulated
        return {
            "is_manipulated": weighted_sum > 0.5,
            "confidence": weighted_sum,
            "detectors": results
        }


# Usage
fusion = ResultFusion([
    DetectorConfig("ela", weight=0.35, threshold=0.7),
    DetectorConfig("fft", weight=0.35, threshold=0.6),
    DetectorConfig("upscaling", weight=0.30, threshold=0.65),
])

final_result = fusion.fuse({
    "ela": {"score": 0.92, "status": "detected"},
    "fft": {"score": 0.85, "status": "detected"},
    "upscaling": {"score": 0.42, "status": "clean"}
})
# Result: is_manipulated=True, confidence=0.77
```

### Pattern 4: Async Pipeline Orchestrator

**What:** Coordinate async detector execution with timeout handling.

**When:** You need concurrent detector execution with fault tolerance.

**Benefits:**
- Detectors run in parallel (reduce latency)
- Timeout prevents slow detectors from blocking
- Graceful degradation on detector failures

**Example:**

```python
import asyncio
from typing import Dict, Any

class PipelineOrchestrator:
    """Orchestrate async detector execution."""

    def __init__(self, registry: DetectorRegistry, timeout: float = 5.0):
        self.registry = registry
        self.timeout = timeout

    async def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Run all detectors with timeout."""
        try:
            results = await asyncio.wait_for(
                self.registry.run_all(image),
                timeout=self.timeout
            )
            return self._handle_results(results)
        except asyncio.TimeoutError:
            return {
                "error": "Processing timeout",
                "status": "timeout"
            }

    def _handle_results(self, results: Dict) -> Dict:
        """Handle exceptions from individual detectors."""
        cleaned = {}
        for name, result in results.items():
            if isinstance(result, Exception):
                cleaned[name] = {
                    "score": 0.0,
                    "status": "error",
                    "error": str(result)
                }
            else:
                cleaned[name] = result
        return cleaned
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Synchronous Sequential Processing

**What:** Processing detectors one-by-one in sequence.

**Why bad:**
- Total latency = sum of all detector latencies (e.g., 4 detectors × 40ms = 160ms)
- Misses 15 FPS target (67ms per frame)
- Wastes CPU during I/O waits

**Consequences:**
- Unacceptable latency for real-time video
- Poor resource utilization

**Instead:** Use async/concurrent execution to parallelize detectors. Latency = max(detector_latencies) instead of sum.

```python
# BAD - Sequential
result_ela = ela_detector.detect(image)  # 40ms
result_fft = fft_detector.detect(image)  # 35ms
result_rppg = rppg_detector.detect(image)  # 45ms
# Total: 120ms

# GOOD - Parallel
results = await asyncio.gather(
    ela_detector.detect(image),
    fft_detector.detect(image),
    rppg_detector.detect(image)
)
# Total: 45ms (slowest detector)
```

### Anti-Pattern 2: Unbounded Frame Buffers

**What:** Accumulating frames in queue without size limit.

**Why bad:**
- Memory grows unboundedly during processing spikes
- Latency increases (processing old frames)
- System becomes unresponsive

**Consequences:**
- Out of memory crashes
- Stale results (reporting on frames from 10 seconds ago)

**Instead:** Use bounded buffer with drop policy (Pattern 2).

### Anti-Pattern 3: Tight Coupling Between Detectors and Orchestrator

**What:** Orchestrator directly imports and instantiates specific detector classes.

**Why bad:**
- Adding new detector requires modifying orchestrator code
- Cannot disable detectors without code changes
- Difficult to test detectors in isolation

**Consequences:**
- Rigid architecture
- Higher maintenance burden
- Deployment risks (must redeploy orchestrator for detector changes)

**Instead:** Use detector registry pattern (Pattern 1) with dependency injection.

```python
# BAD - Tight coupling
class Orchestrator:
    def __init__(self):
        self.ela = ELADetector()
        self.fft = FFTDetector()
        # Adding new detector requires code change here

    async def process(self, image):
        ela_result = await self.ela.detect(image)
        fft_result = await self.fft.detect(image)
        # And here...

# GOOD - Loose coupling via registry
class Orchestrator:
    def __init__(self, registry: DetectorRegistry):
        self.registry = registry  # Detectors injected

    async def process(self, image):
        return await self.registry.run_all(image)
```

### Anti-Pattern 4: Ignoring Frame History for Temporal Detectors

**What:** Running rPPG detector on individual frames without temporal context.

**Why bad:**
- rPPG requires analyzing color changes across 30+ frames (2 seconds)
- Single-frame analysis produces meaningless noise
- False positives/negatives

**Consequences:**
- Temporal detectors produce garbage output
- Misleading confidence scores

**Instead:** Maintain sliding window buffer for temporal detectors, skip if insufficient history.

```python
# BAD - Single frame
rppg_result = await rppg_detector.detect(single_frame)  # Meaningless

# GOOD - Frame sequence
class VideoOrchestrator:
    def __init__(self):
        self.frame_history = deque(maxlen=30)

    async def process_frame(self, frame):
        self.frame_history.append(frame)

        # Only run rPPG if we have enough history
        if len(self.frame_history) >= 30:
            rppg_result = await rppg_detector.detect(
                list(self.frame_history)
            )
```

### Anti-Pattern 5: Not Logging Dropped Frames

**What:** Silently dropping frames without tracking or reporting.

**Why bad:**
- Client doesn't know if results are incomplete
- No visibility into system load
- Cannot diagnose performance issues

**Consequences:**
- Misleading results (gaps in video analysis)
- Production incidents hard to debug

**Instead:** Track and report dropped frame count in responses.

```python
{
  "frame_id": 142,
  "is_manipulated": false,
  "confidence": 0.23,
  "dropped_frames": 5,  # Client knows 5 frames were skipped
  "warning": "High system load - consider reducing frame rate"
}
```

---

## Scalability Considerations

| Concern | At 1 User | At 10 Users | At 100+ Users |
|---------|-----------|-------------|---------------|
| **API** | Single FastAPI instance | Add load balancer (nginx), multiple FastAPI workers | Kubernetes horizontal pod autoscaling, Redis pub/sub for WebSocket fan-out |
| **Detector Execution** | Async within process | Thread pool for CPU-bound detectors (ProcessPoolExecutor) | Celery/RQ task queue, separate worker processes per detector type |
| **Frame Buffers** | In-memory deque | In-memory per-connection | Redis Streams for distributed buffering |
| **rPPG History** | In-memory per WebSocket | LRU cache with TTL | Redis with TTL for frame history |
| **Result Storage** | Not needed (real-time only) | SQLite for audit logs | PostgreSQL cluster for forensic audit trail |
| **GPU Acceleration** | Not needed (classical detectors) | Single GPU for all detectors | Multiple GPUs with load balancing (for future DL detectors) |

---

## Build Order (Dependency Graph)

Suggested implementation sequence based on dependencies:

```
Phase 1: Foundation
├─> [1.1] Base detector interface (ABC)
├─> [1.2] Detector registry
└─> [1.3] Result fusion logic

Phase 2: Detectors (can parallelize)
├─> [2.1] ELA detector (simplest - JPEG recompression)
├─> [2.2] FFT detector (frequency domain analysis)
├─> [2.3] Upscaling detector (spectral peak detection)
└─> [2.4] rPPG detector (most complex - requires face detection + temporal analysis)

Phase 3: Pipeline Orchestrator
├─> [3.1] Async orchestrator (run detectors concurrently)
├─> [3.2] Image processing flow (single frame)
└─> [3.3] Frame buffer manager (bounded queue with drop policy)

Phase 4: API Layer
├─> [4.1] REST endpoint (POST /detect/image)
└─> [4.2] WebSocket endpoint (ws://detect/stream)

Phase 5: Production Readiness
├─> [5.1] Logging and monitoring (dropped frames, latencies)
├─> [5.2] Configuration management (detector weights, thresholds)
└─> [5.3] Health checks and graceful shutdown
```

**Critical path:** Phase 1 → Phase 3 → Phase 4
**Parallelizable:** All Phase 2 detectors can be built concurrently

**Recommendation:** Start with ELA detector (Phase 2.1) as proof-of-concept while building orchestrator (Phase 3).

---

## Technology Stack Integration

### Core Framework: FastAPI + asyncio

**Why:**
- Native async/await support (critical for concurrent detector execution)
- WebSocket support built-in
- Auto-generated OpenAPI docs
- ASGI (async) vs Flask (sync/WSGI)

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI()

@app.post("/detect/image")
async def detect_image(file: UploadFile):
    image = await load_image(file)
    result = await orchestrator.process_image(image)
    return JSONResponse(result)

@app.websocket("/detect/stream")
async def detect_stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        frame_bytes = await websocket.receive_bytes()
        frame = decode_frame(frame_bytes)
        result = await orchestrator.process_frame(frame)
        await websocket.send_json(result)
```

### Async Execution: asyncio + ProcessPoolExecutor

**Why:**
- `asyncio` for I/O-bound tasks (API, WebSocket)
- `ProcessPoolExecutor` for CPU-bound detectors (image processing bypasses GIL)

```python
from concurrent.futures import ProcessPoolExecutor
import asyncio

# CPU-bound detector execution in separate process
executor = ProcessPoolExecutor(max_workers=4)

async def run_cpu_detector(detector, image):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, detector.detect, image)
```

### WebSocket: FastAPI native

**Why:**
- Built-in WebSocket support
- Async message handling
- Connection state management

### Image Processing: Pillow + NumPy + OpenCV

**Why:**
- Pillow: Load/save images, JPEG operations (for ELA)
- NumPy: Fast array operations (for FFT, spectral analysis)
- OpenCV: Face detection (for rPPG), video frame decoding

---

## Sources

### Architecture Research (HIGH confidence)

- [VAAS: Vision-Attention Anomaly Scoring for Image Manipulation Detection](https://arxiv.org/html/2512.15512) - Dual-module forensic architecture (2025)
- [Unravelling Digital Forgeries: Systematic Survey on Image Manipulation Detection](https://dl.acm.org/doi/10.1145/3731243) - Comprehensive architecture survey
- [MMFusion: Combining Image Forensic Filters](https://arxiv.org/html/2312.01790v2) - Multi-modal fusion architecture
- [DF-Net: Digital Forensics Network for Image Forgery Detection](https://arxiv.org/html/2503.22398v1) - Lightweight detection network

### Video Pipeline Architecture (MEDIUM confidence)

- [Deepfake Video Detection Using Visual Attention](https://www.nature.com/articles/s41598-025-23920-0) - Pipeline components (2025)
- [A Robust Hybrid Deepfake Detection Pipeline](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5901142) - Spatial-temporal-explainability architecture
- [Networking Systems for Video Anomaly Detection](https://dl.acm.org/doi/10.1145/3729222) - Real-time video system architecture

### Detector-Specific Architecture (MEDIUM confidence)

- [Detecting Image Manipulation with ELA-CNN Integration](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323046/) - ELA architecture (2024)
- [TSFF-Net: Time-Frequency Domain Fusion for Deepfake Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC11642989/) - FFT-based architecture
- [Local Attention and Long-Distance Interaction of rPPG for Deepfake Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC10052279/) - rPPG detector architecture
- [AVENUE: Novel Deepfake Detection Method Based on TCN and rPPG](https://dl.acm.org/doi/10.1145/3702232) - Temporal convolutional network for rPPG

### Python Async Pipeline (MEDIUM confidence)

- [VidGear: High-Performance Video Processing Framework](https://github.com/abhiTronix/vidgear) - Async video processing
- [Real-Time Video Processing Pipeline](https://fenilsonani.com/articles/real-time-video-processing-pipeline) - Scalable architecture patterns
- [PyNvVideoCodec 2.0 for GPU-Accelerated Processing](https://developer.nvidia.com/blog/whats-new-in-pynvvideocodec-2-0-for-python-gpu-accelerated-video-processing/) - GPU pipeline architecture

### WebSocket & Frame Management (LOW confidence - older research)

- [WebSocket Video Processing GitHub](https://github.com/waqarmumtaz369/WebSocket-Video-Processing) - Frame streaming patterns
- [Python WebSocket Video Streaming](https://medium.com/python-other/pythons-simple-websocket-based-video-streaming-to-web-pages-f6b5789525cc) - Implementation examples
- [WebSocket Streaming 2025 Guide](https://www.videosdk.live/developer-hub/websocket/websocket-streaming) - Modern WebSocket patterns

### API Design (MEDIUM confidence)

- [REST API Security Best Practices 2026](https://www.levo.ai/resources/blogs/rest-api-security-best-practices) - Security patterns
- [Modern API Design Best Practices 2026](https://www.xano.com/blog/modern-api-design-best-practices/) - API architecture

### Upscaling Detection (LOW confidence - limited recent research)

- [Robust Estimation of Upscaling Factor on Double JPEG](https://ieeexplore.ieee.org/document/9409696/) - Upscaling detection methods
- [Upscaling Factor Estimation on Double JPEG Compressed Images](https://link.springer.com/article/10.1007/s11042-019-08519-8) - Spectral analysis approach

### Frame Buffer Management (LOW confidence - older academic papers)

- [Frame Rate Control Buffer Management for Video Conferencing](https://link.springer.com/chapter/10.1007/978-981-10-7605-3_124) - Buffer management techniques
- [Understanding Latency in Video Compression Systems](https://www.design-reuse.com/articles/33005/understanding-latency-in-video-compression-systems.html) - Latency reduction strategies

---

## Confidence Assessment by Area

| Area | Confidence | Rationale |
|------|-----------|-----------|
| Overall Architecture | HIGH | Multiple 2025-2026 papers confirm dual-stream + fusion pattern |
| Detector Patterns | MEDIUM | Specific implementations vary, but core principles well-established |
| Python Async Pipeline | MEDIUM | VidGear and modern frameworks confirm feasibility, but NilinkVerifierEngine-specific implementation untested |
| WebSocket Streaming | MEDIUM | Pattern established, but limited 2026-specific research |
| Frame Dropping | LOW | Mostly older academic research, modern practices less documented |
| rPPG Implementation | MEDIUM | Recent papers (2025) confirm architecture, but complexity high |
| Scalability | LOW | Based on general web service patterns, not forensics-specific |

---

## Open Questions for Phase-Specific Research

1. **rPPG Frame Requirements**: Exactly how many frames needed for reliable signal? Papers suggest 30-60, but optimal for real-time unclear.
2. **Detector Weight Optimization**: How to tune fusion weights? A/B testing required or meta-learning approach?
3. **GPU Acceleration**: When does GPU become cost-effective? At 10 users? 100?
4. **False Positive Rates**: What confidence threshold minimizes false positives while maintaining recall?
5. **Compression Robustness**: How does JPEG compression (WhatsApp, social media) affect detector accuracy?

These questions should be addressed during implementation phases via experimentation and benchmarking.
