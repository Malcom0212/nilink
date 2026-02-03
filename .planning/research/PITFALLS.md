# Pitfalls Research

**Domain:** Forensic Image/Video Manipulation Detection
**Researched:** 2026-02-02
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Overconfidence in Mathematical Methods Against Adversarial Attacks

**What goes wrong:**
Teams building forensic detection systems assume that mathematical analysis (ELA, FFT, rPPG) is inherently robust because it doesn't rely on trained models. Adversaries then craft counter-forensic attacks that specifically target these mathematical signatures, causing the system to misclassify manipulated media as authentic. GAN-based adversarial training can reduce evasion success by 15-25%, but many mathematical methods lack even basic adversarial defenses.

**Why it happens:**
The mindset "we're not using ML, so we're safe from adversarial examples" is fundamentally flawed. Mathematical forensic methods leave predictable traces (ELA patterns, frequency anomalies, rPPG signals) that attackers can learn to suppress or forge. Investigators assume their detection approach is novel enough that adversaries won't reverse-engineer it, but in 2026, automation and AI assistance make counter-forensic techniques cheaper, faster, and more repeatable.

**How to avoid:**
1. **Assume adversarial awareness:** Design each detection method with the assumption that attackers know exactly how it works
2. **Multi-layered defense:** Combine multiple forensic techniques so defeating one doesn't compromise the entire system
3. **Randomization:** Where possible, introduce randomness in detection parameters (e.g., which frequency bands to analyze, ELA compression levels)
4. **Continuous updating:** Monitor for new counter-forensic techniques and adapt detection methods accordingly
5. **Testing with adversarial samples:** Explicitly test against known CF attacks (violation attacks, evasion attacks) during development

**Warning signs:**
- Detection accuracy suddenly drops on new samples from the same source
- Manipulated media from a specific generator consistently passes detection
- Forensic traces appear "too clean" or suspiciously uniform
- Detection confidence scores cluster at threshold boundaries

**Phase to address:**
Phase 1 (Core Detection Engine) must include adversarial robustness from day one. Phase 3+ should include red-teaming and adversarial testing protocols.

---

### Pitfall 2: False Positives from Legitimate Compression and Processing

**What goes wrong:**
The system flags authentic, unmanipulated media as fake due to compression artifacts, codec-specific signatures, or legitimate post-processing (brightness adjustment, cropping, format conversion). ELA-based methods are particularly susceptible, showing false positives in very homogeneous areas of JPEG images. Users lose trust in the system when it incorrectly flags genuine content, especially in high-stakes scenarios (legal proceedings, content moderation).

**Why it happens:**
Forensic algorithms detect "traces of processing" rather than "intent to deceive." A photo saved in Photoshop shows ELA artifacts from auto-sharpening even if nothing was maliciously altered. Videos compressed with H.264/H.265 create prediction errors and motion artifacts that resemble manipulation traces. Social media platforms re-encode uploads, destroying original forensic signatures and introducing platform-specific artifacts. Teams optimize for recall (catch all manipulations) at the expense of precision (avoid false flags), leading to production systems with unacceptable false positive rates.

**How to avoid:**
1. **Threshold tuning with production data:** Lab performance doesn't predict real-world FPR. Test on diverse authentic media from actual use cases
2. **Codec-aware baselines:** Establish expected artifact patterns for common codecs (H.264, H.265, VP9, AV1) and filter out codec-specific signatures
3. **Platform-specific models:** Train or calibrate detection for media from specific sources (YouTube, TikTok, Instagram) if known
4. **Multi-evidence requirement:** Require multiple independent forensic signals before flagging as manipulated
5. **Confidence scoring:** Provide probabilistic scores rather than binary authentic/fake classifications
6. **Human-in-the-loop for borderline cases:** Flag low-confidence detections for expert review rather than auto-classifying

**Warning signs:**
- False positive rate exceeds 10% in pilot testing
- Users report "obvious real photos" being flagged
- Detection performance degrades after media passes through social media platforms
- High variance in detection scores for media from the same authentic source
- Analysts spending >30% of time investigating false alarms

**Phase to address:**
Phase 2 (Testing & Validation) must include comprehensive false positive evaluation. Phase 4+ should implement threshold optimization and confidence calibration for production deployment.

---

### Pitfall 3: Real-Time Performance Degradation at Scale

**What goes wrong:**
The system achieves 15+ FPS in development on single video streams but collapses to <5 FPS in production when handling concurrent requests, high-resolution media, or long videos. Performance bottlenecks emerge from unoptimized FFT computations, rPPG signal extraction across all frames, or synchronous processing architectures. API response times spike during peak load, causing timeouts and user frustration.

**Why it happens:**
Teams optimize for single-stream throughput ("it runs at 20 FPS on my laptop") without considering multi-tenancy, concurrent processing, or edge cases (4K video, 60-minute livestreams). Memory usage scales linearly with video length, causing OOM errors on long clips. Database I/O for logging detection results becomes a bottleneck. The 15 FPS target doesn't account for frame pre-processing (decoding, color space conversion) or post-processing (result aggregation, confidence scoring). Developers test with synthetic or short clips rather than production-representative media.

**How to avoid:**
1. **Async processing architecture:** Use worker queues (Celery, BullMQ) to decouple API requests from processing
2. **Frame sampling strategies:** Don't analyze every frame—sample keyframes or use temporal sliding windows
3. **Early stopping:** Exit detection pipeline early when confidence threshold is reached
4. **Hardware acceleration:** Leverage GPU/NPU for FFT and frequency analysis if available
5. **Streaming processing:** Process video in chunks rather than loading entire files into memory
6. **Resource pooling:** Pre-allocate processing resources and reuse across requests
7. **Performance testing with realistic load:** Simulate production conditions (concurrent users, diverse media types, network latency)

**Warning signs:**
- Memory usage grows unbounded with video length
- CPU pegged at 100% during multi-stream processing
- Response times increase super-linearly with video resolution
- System fails on videos >10 minutes or >1080p
- Throughput drops when handling >5 concurrent requests

**Phase to address:**
Phase 1 (Core Detection Engine) must architect for streaming/chunked processing. Phase 3 (API Service) requires comprehensive load testing and performance optimization before production.

---

### Pitfall 4: Ignoring Manipulation Chain Complexity

**What goes wrong:**
The system successfully detects single-operation manipulations (face swap, deepfake generation) but fails on media subjected to multiple editing operations (deepfake → compress → brightness adjust → re-encode → share on social media). Each processing step erodes earlier forensic traces, with later operations overwriting signatures left by previous ones. Detection accuracy declines dramatically on real-world media that has undergone multiple transformations.

**Why it happens:**
Forensic research focuses on isolated manipulation types with clean datasets (FaceForensics++, DFDC). Real adversaries don't generate a deepfake and immediately publish—they post-process to evade detection. Social media platforms automatically transcode uploads, adding another layer to the manipulation chain. Teams test detection on "manipulation A" and "manipulation B" separately but never "manipulation A → post-processing B → platform encoding C." The assumption that "if we detect the original manipulation, we're good" breaks down when that signature is destroyed by subsequent processing.

**How to avoid:**
1. **Chain-aware testing:** Explicitly test on manipulation chains, not isolated operations
2. **Robust feature selection:** Prioritize forensic features that survive re-encoding and compression
3. **Multi-stage detection:** Attempt to identify multiple operations in the processing history
4. **End-to-end evaluation:** Test on media that has passed through real publishing pipelines (social media upload/download cycles)
5. **Temporal consistency checks:** For video, analyze inter-frame relationships which are harder to destroy through post-processing

**Warning signs:**
- Lab accuracy: 95%, production accuracy: 60%
- Detection works on downloaded deepfakes but fails on social media screenshots
- Performance degrades significantly on lossy compressed media
- System flags freshly generated manipulations but misses "in the wild" examples

**Phase to address:**
Phase 2 (Testing & Validation) must include manipulation chain scenarios. Phase 4+ should implement chain-robust detection strategies if initial results show degradation.

---

### Pitfall 5: rPPG Signal Reliability Assumptions

**What goes wrong:**
Teams implement rPPG-based deepfake detection assuming that manipulated videos lack physiological signals, only to discover that high-quality deepfakes actually contain heart-rate signals, compressed videos destroy rPPG signatures in authentic footage, and environmental factors (lighting, facial movement) cause detection failures. The system labels genuine videos as fake when lighting conditions interfere with pulse detection.

**Why it happens:**
Earlier research showed that simple deepfakes lacked rPPG signals, leading to the assumption this is a reliable detection method. However, contemporary deepfake generation techniques are sophisticated enough that physiological signals appear even without explicit modeling. Compression (especially social media platforms) removes the subtle pixel variations needed for rPPG extraction. Teams test in controlled environments (studio lighting, minimal movement) rather than real-world scenarios (variable lighting, motion, camera shake). The method works beautifully in labs but fails in production.

**How to avoid:**
1. **Don't rely solely on rPPG:** Use it as one signal among many, not the primary detection method
2. **Quality gating:** Detect when rPPG extraction conditions are poor (low resolution, high compression, poor lighting) and fall back to other methods
3. **Negative evidence handling:** Absence of rPPG signal means "inconclusive," not "fake"
4. **Environmental robustness testing:** Test across lighting conditions, compression levels, and motion scenarios
5. **Cross-validation with other methods:** Require corroboration from ELA or FFT before flagging based on rPPG alone

**Warning signs:**
- High false positive rate on authentic videos from smartphones or security cameras
- Detection accuracy varies dramatically based on video source quality
- System performs well on uncompressed lab videos but poorly on social media content
- rPPG signal extraction fails silently, defaulting to "suspicious" classification

**Phase to address:**
Phase 2 (Testing & Validation) must evaluate rPPG reliability across realistic conditions. Phase 3+ should implement quality detection and fallback mechanisms.

---

### Pitfall 6: Frequency Domain Analysis Limitations

**What goes wrong:**
Teams implement FFT-based manipulation detection but encounter limitations: inability to localize tampering spatially (global frequency analysis doesn't pinpoint where edits occurred), phase information loss when focusing only on magnitude spectrum, computational expense on high-resolution images, and Gibbs phenomenon artifacts from frequency domain filtering that are mistaken for manipulation traces.

**Why it happens:**
Frequency domain analysis is powerful for detecting periodic patterns and compression artifacts but has inherent tradeoffs. The Fourier transform spreads local features across all frequency bins, destroying spatial localization. Teams analyze magnitude spectrum (easy to visualize) while ignoring phase (harder to interpret but contains critical positional information). Filter design mistakes (sharp cutoffs) introduce ringing artifacts. For very large images or real-time video, 2D FFT becomes computationally prohibitive without optimization.

**How to avoid:**
1. **Wavelet transforms for localization:** Use discrete wavelet transform (DWT) instead of FFT when spatial localization is needed
2. **Phase analysis:** Include phase spectrum analysis, not just magnitude
3. **Windowing for video:** Apply FFT to local patches rather than entire frames
4. **Filter design discipline:** Use gradual filter transitions to avoid Gibbs phenomenon
5. **Hybrid spatial-frequency:** Combine frequency analysis with spatial methods (ELA, noise analysis) for complete picture
6. **GPU acceleration:** Offload FFT computations to GPU for real-time performance

**Warning signs:**
- Can detect "something is wrong" but can't localize the manipulation
- High computational cost limits real-time processing
- Filter artifacts mistaken for tampering evidence
- Detection performance varies wildly across image types (smooth vs textured)

**Phase to address:**
Phase 1 (Core Detection Engine) should implement FFT carefully with awareness of limitations. Phase 3+ may add wavelet analysis if spatial localization proves critical.

---

### Pitfall 7: ELA Method Over-Reliance and Misinterpretation

**What goes wrong:**
Teams implement Error Level Analysis as a primary detection method but suffer from systematic false positives on homogeneous image regions (sky, walls, solid colors), inability to detect manipulations when images are saved with the same quantization tables, and misleading results on images processed with Adobe products (auto-sharpening creates high error levels that don't indicate manipulation). Investigators mistake tool artifacts for evidence of forgery.

**Why it happens:**
ELA is conceptually simple and produces visually intuitive outputs (heatmaps), making it popular despite significant limitations. JPEG compression behaves unpredictably in homogeneous regions, creating high error levels that look suspicious but are compression artifacts. When forged images are re-saved with the same software and settings as the original, error levels normalize. Simple image adjustments (save in Photoshop, adjust brightness) create ELA signatures indistinguishable from intentional manipulation. Teams see ELA as "the forgery detection tool" rather than "one technique among many."

**How to avoid:**
1. **Never use ELA alone:** Require corroboration from multiple forensic methods
2. **Codec-aware interpretation:** Understand expected ELA patterns for different JPEG quality levels and encoders
3. **Homogeneous region filtering:** Ignore or de-weight ELA results in uniform areas
4. **Software signature database:** Catalog ELA patterns from common editing tools to distinguish tool artifacts from manipulation
5. **Explicit uncertainty:** Flag "inconclusive" results rather than forcing binary classification
6. **User education:** If exposing ELA results, provide context on interpretation pitfalls

**Warning signs:**
- High error levels in sky regions, solid backgrounds, or gradients
- Every image processed in Photoshop flags as suspicious
- Detection rate is binary rather than nuanced
- Analysts misinterpreting ELA heatmaps without supporting evidence

**Phase to address:**
Phase 1 (Core Detection Engine) must implement ELA with awareness of limitations and never as sole detection method. Phase 4+ should include analyst training if human review is part of workflow.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Processing every video frame | Higher detection confidence | 10-20x slower performance, can't achieve 15 FPS target | Never in production—use frame sampling or keyframe analysis |
| Binary authentic/fake classification | Simple API, easy to understand | High false positive impact, no nuance for borderline cases | Only in MVP—must add confidence scoring for production |
| Single-threaded synchronous processing | Easier to debug, simpler code | Can't handle concurrent requests, poor scalability | Only during Phase 1 prototyping |
| Loading entire video into memory | Simpler code, faster random access | OOM errors on long videos, can't handle large files | Never—must stream/chunk from day one |
| Hardcoded detection thresholds | Fast to implement, no tuning needed | Brittle across different media sources, high FP rate | Only in early development—must be configurable by Phase 2 |
| Skipping adversarial testing | Faster initial development | Vulnerable to counter-forensic attacks, poor production robustness | Never—adversarial awareness must be built-in from Phase 1 |
| Using only one forensic method | Simpler implementation, faster processing | Easily defeated, high error rates | Never—multi-method approach is critical |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Social media APIs | Assuming video metadata is reliable for forensics | Ignore metadata—it's easily forged. Only trust pixel-level analysis |
| Video decoding (FFmpeg) | Not specifying exact pixel format | Explicitly set pixel format (RGB24, YUV420p) to ensure consistent color space for analysis |
| Cloud storage (S3, GCS) | Downloading entire video before processing | Use byte-range requests to stream chunks, process as data arrives |
| Database logging | Synchronous writes blocking response | Async background logging—don't make users wait for forensic logs to persist |
| External ML APIs | Assuming high accuracy translates to forensics | Verify explainability—black-box ML isn't admissible evidence in many contexts |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading full video into RAM | Works fine in testing, OOM errors in production | Streaming/chunked processing from day one | Videos >500MB or >10 concurrent users |
| Synchronous processing per request | Dev: 1 request at a time. Prod: timeouts under load | Queue-based async processing (Celery, Bull) | >5 concurrent requests or videos >1 minute |
| Per-frame FFT without batching | Single frame: 50ms. Full video: 20 minutes | Batch FFT operations, GPU acceleration | Videos >30 seconds at 30 FPS |
| N+1 database queries for results | Fast with 1 video. Crawls with 100. | Batch/bulk operations, caching | >50 detection results to aggregate |
| Global locking for resource access | Single thread: no problem. Multi-thread: serialized | Lock-free data structures, per-resource locks | >3 concurrent processing workers |
| Disk I/O for intermediate results | Local testing: SSD is fast. Cloud: network latency kills | In-memory processing, minimize I/O | Cloud deployment or processing >10 videos/minute |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting client-provided media metadata | Adversary forges timestamps, camera models to mislead forensics | Ignore all metadata—only analyze pixel/frequency data |
| Exposing detection method details in API responses | Adversary learns which techniques you use and crafts counter-forensics | Return opaque confidence scores, not method-specific breakdowns |
| No rate limiting on detection API | Adversary uses your API to test counter-forensic techniques at scale | Aggressive rate limiting, require authentication, monitor for abuse patterns |
| Storing sensitive detection logs indefinitely | Privacy violations, legal liability if breached | Retention policies—delete forensic analysis logs after 30-90 days unless flagged |
| Not validating media file format | Malicious files could exploit decoder vulnerabilities | Strict file format validation, run decoders in sandboxed environment |
| Leaking timing information | Processing time reveals detection method internals | Constant-time responses or add random jitter to prevent timing attacks |

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Binary "FAKE" / "AUTHENTIC" labels | Users treat low-confidence results as definitive, leading to wrong decisions | Confidence scores + "inconclusive" category for borderline cases |
| No explanation of detection basis | Users don't trust black-box results, especially in high-stakes scenarios | Provide forensic evidence summary: "Multiple compression layers detected, face region shows inconsistent noise patterns" |
| Hiding false positive rate from users | Users over-rely on system, don't verify results | Display system accuracy metrics, encourage human verification |
| Slow processing without progress indication | Users assume system is broken, abandon requests | Real-time progress updates: "Analyzing frame 45/300... 15% complete" |
| Treating all media types identically | Video detection is much harder than images—same UX creates false confidence | Separate workflows, different confidence thresholds for image vs video |
| No recourse for disputed results | Users can't challenge false positives, lose trust in system | Appeal/review mechanism for high-stakes classifications |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **ELA implementation:** Often missing homogeneous region handling—verify false positive rate on sky/solid-color images
- [ ] **rPPG detection:** Often missing quality checks—verify behavior on compressed/low-light videos where signal extraction fails
- [ ] **FFT analysis:** Often missing phase information—verify you're using both magnitude AND phase, not just magnitude spectrum
- [ ] **API endpoints:** Often missing streaming support—verify can handle >1GB video files without loading into memory
- [ ] **Confidence scoring:** Often missing calibration—verify scores are meaningfully distributed across range, not clustered at extremes
- [ ] **Adversarial robustness:** Often missing counter-forensic testing—verify performance against known CF attack techniques
- [ ] **Performance testing:** Often missing concurrent load—verify throughput with 20+ simultaneous requests, not just sequential
- [ ] **Codec compatibility:** Often missing H.265/VP9/AV1 support—verify works on modern codecs, not just H.264
- [ ] **Error handling:** Often missing graceful degradation—verify behavior when one detection method fails
- [ ] **Documentation:** Often missing interpretation guidelines—verify users understand what confidence scores mean

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| High false positive rate in production | MEDIUM | 1. Implement confidence threshold tuning UI. 2. Collect production FP samples. 3. Re-calibrate thresholds offline. 4. Deploy updated thresholds. 5. Monitor FP rate over 7 days. |
| Performance < 15 FPS under load | HIGH | 1. Identify bottleneck via profiling (CPU/memory/I/O). 2. Implement async processing queue. 3. Add frame sampling (every Nth frame). 4. Deploy incremental fixes. 5. Load test iteratively. (2-4 week effort) |
| Adversarial attacks bypassing detection | HIGH | 1. Collect adversarial samples. 2. Analyze which forensic methods failed. 3. Add new detection methods or strengthen existing. 4. Adversarial training if using ML components. 5. Re-deploy and test. (3-6 week effort) |
| ELA misinterpreting compression artifacts | LOW | 1. Add codec-aware baseline filtering. 2. Require multi-method corroboration. 3. Update documentation with interpretation guide. 4. Deploy in 1-2 days. |
| rPPG failing on compressed videos | LOW | 1. Add video quality detection. 2. Disable rPPG below quality threshold. 3. Fall back to other methods. 4. Deploy in 2-3 days. |
| Manipulation chain eroding detection | MEDIUM | 1. Build test dataset with manipulation chains. 2. Identify which chains cause failures. 3. Add chain-robust features or multi-stage detection. 4. Re-test and deploy. (2-3 week effort) |
| Memory exhaustion on long videos | LOW | 1. Implement streaming processing. 2. Process in fixed-size chunks (e.g., 10-second segments). 3. Deploy with testing on long videos. (3-5 day effort) |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Adversarial attacks bypassing detection | Phase 1 (Core Engine) | Test against known CF attack techniques, measure evasion success rate |
| False positives from compression | Phase 2 (Testing & Validation) | FP rate <5% on authentic media from 10+ diverse sources |
| Real-time performance degradation | Phase 3 (API Service) | Load testing: 15+ FPS sustained with 20 concurrent requests |
| Manipulation chain complexity | Phase 2 (Testing & Validation) | Test on 5+ manipulation chain scenarios, accuracy >80% |
| rPPG signal reliability | Phase 2 (Testing & Validation) | Quality detection implemented, graceful fallback when rPPG fails |
| FFT limitations | Phase 1 (Core Engine) | Combine with spatial methods, measure localization accuracy |
| ELA over-reliance | Phase 1 (Core Engine) | Multi-method requirement enforced, no single-method classifications |
| Binary classification issues | Phase 3 (API Service) | Confidence scoring + "inconclusive" category implemented |
| Performance traps | Phase 3 (API Service) | Streaming processing, async queue, demonstrated on production-scale load |
| Security vulnerabilities | Phase 4 (Production Hardening) | Security audit, rate limiting, metadata validation, sandboxing |

## Sources

### Forensic Detection Challenges & Limitations
- [Unravelling Digital Forgeries: A Systematic Survey](https://dl.acm.org/doi/10.1145/3731243)
- [Recent advances in digital image manipulation detection techniques](https://www.sciencedirect.com/science/article/abs/pii/S0379073820301730)
- [Understanding image forensics techniques and their impact](https://www.cameraforensics.com/blog/2024/03/25/understanding-image-forensics-techniques-and-their-impact/)

### Deepfake Detection & False Positives
- [Unmasking digital deceptions: An integrative review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508882/)
- [Deepfake Detection 2026: The Future of Trust in the Digital](https://learnaitools.in/deepfake-detection-2026-the-future-of-trust/)
- [Deepfake video detection methods, approaches, and challenges](https://www.sciencedirect.com/science/article/pii/S111001682500465X)
- [Deepfake Media Forensics: Status and Future Challenges](https://pmc.ncbi.nlm.nih.gov/articles/PMC11943306/)

### ELA Limitations
- [Error Level Analysis (Wikipedia)](https://en.wikipedia.org/wiki/Error_level_analysis)
- [An evaluation of Error Level Analysis in image forensics](https://ieeexplore.ieee.org/document/7412439/)
- [Tutorial: Error Level Analysis (FotoForensics)](https://www.fotoforensics.com/tutorial-ela.php)

### Performance & Scalability
- [Why Manual Forensic Video Analysis Is Becoming Obsolete](https://www.ambient.ai/blog/forensic-video-analysis)
- [Unlocking workflow efficiencies in video forensics 2026](https://www.magnetforensics.com/resources/unlocking-workflow-efficiencies-in-video-forensics-whats-new-in-2026/)
- [Real-Time Video Processing with AI: Best Practices for 2025](https://www.forasoft.com/blog/article/real-time-video-processing-with-ai-best-practices)

### Adversarial Attacks
- [Forensics And Futures: Navigating Digital Evidence, AI, And Risk In 2026](https://lcgdiscovery.com/forensics-and-futures-navigating-digital-evidence-ai-and-risk-in-2026-part-1/)
- [A Survey of Machine Learning Techniques in Adversarial Image Forensics](https://www.researchgate.net/publication/344778716_A_Survey_of_Machine_Learning_Techniques_in_Adversarial_Image_Forensics)
- [Evasion Attacks on Image Classification Models](https://link.springer.com/chapter/10.1007/978-981-96-4933-4_18)

### FFT & Frequency Domain Analysis
- [Fast Fourier Transform in Image Processing](https://www.geeksforgeeks.org/computer-vision/fast-fourier-transform-in-image-processing/)
- [Frequency-Aware Deepfake Detection: Improving Generalizability](https://ojs.aaai.org/index.php/AAAI/article/view/28310/28609)
- [Fourier Image Analysis](https://www.dspguide.com/ch24/5.htm)

### rPPG Detection Challenges
- [High-quality deepfakes have a heart!](https://www.frontiersin.org/journals/imaging/articles/10.3389/fimag.2025.1504551/full)
- [On Using rPPG Signals for DeepFake Detection: A Cautionary Note](https://dl.acm.org/doi/10.1007/978-3-031-43153-1_20)
- [DeepFakes Detection Based on Heart Rate Estimation](https://link.springer.com/chapter/10.1007/978-3-030-87664-7_12)

### Production Deployment Challenges
- [Image Forgery Detection Techniques: Latest Trends And Key Challenges](https://www.researchgate.net/publication/385833419_Image_Forgery_Detection_Techniques_Latest_Trends_And_Key_Challenges)
- [Hybrid framework for image forgery detection and robustness against adversarial attacks](https://www.nature.com/articles/s41598-025-25436-z)

### Video Codec Complexity
- [Video Compression Artifacts in Surveillance Footage](https://blog.ampedsoftware.com/2021/04/13/video-compression-artifacts)
- [CoFFEE: codec-based forensic feature extraction for H.264 videos](https://link.springer.com/article/10.1186/s13635-024-00181-4)
- [Behind the Screen: Video Codecs and Formats Unveiled](https://blog.ampedsoftware.com/2024/11/08/video-codecs-and-formats)

### Mathematical vs ML Tradeoffs
- [Deepfake Forensics Is Much More Than Deepfake Detection!](https://blog.ampedsoftware.com/2025/08/05/deepfake-forensics)
- [Comparison of Deepfake Detection Techniques through Deep Learning](https://www.mdpi.com/2624-800X/2/1/7)
- [Deepfake video detection: challenges and opportunities](https://link.springer.com/article/10.1007/s10462-024-10810-6)

### Ensemble Methods & Performance Tradeoffs
- [Speed/accuracy trade-offs for modern convolutional object detectors](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_SpeedAccuracy_Trade-Offs_for_CVPR_2017_paper.pdf)
- [Forensic Technology: Algorithms Strengthen Forensic Analysis](https://www.gao.gov/products/gao-21-435sp)

---
*Pitfalls research for: Forensic Image/Video Manipulation Detection*
*Researched: 2026-02-02*
