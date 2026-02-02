# Nilink - Project State

**Last updated:** 2026-02-02
**Status:** Core engine implemented, ready for next phase

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Detecter les manipulations visuelles en temps reel sans ML lourd
**Current focus:** API layer (REST + WebSocket)

## What's Done

### Phase 0: Project Setup ✓
- [x] PROJECT.md created with full context
- [x] config.json with workflow preferences (YOLO, standard depth, parallel)
- [x] Git repo initialized

### Phase 1: Core Engine ✓
- [x] `Nilink_engine.py` implemented (~980 lines)
- [x] 4 detecteurs forensiques:
  - ELA (Error Level Analysis) — inpainting/generation
  - FFT (analyse spectrale) — artefacts GAN/diffusion
  - rPPG (photoplethysmographie) — deepfake/liveness
  - Noise consistency — upscaling IA
- [x] Async stream processing avec frame dropping
- [x] Detection visage OpenCV Haar
- [x] Output: score 0-1, anomalies, heatmap, regions
- [x] Demo modes: synthetic, image, webcam

## What's Next

### Phase 2: API Layer (a faire)
- [ ] FastAPI REST endpoint pour images
- [ ] WebSocket endpoint pour flux video
- [ ] Serialisation JSON des resultats
- [ ] Gestion erreurs et validation input

### Phase 3: Testing & Validation (a faire)
- [ ] Tests unitaires pour chaque detecteur
- [ ] Benchmark images vraies vs fakes
- [ ] Tuning seuils de detection
- [ ] Tests de performance (FPS, latence)

### Phase 4: Production Hardening (a faire)
- [ ] Docker containerization
- [ ] Rate limiting
- [ ] Logging et monitoring
- [ ] Documentation API

## Quick Resume Commands

```bash
# Voir l'etat du projet
cd C:\Users\matys\Nilink
cat .planning/STATE.md

# Tester le moteur
pip install opencv-python numpy scipy
python Nilink_engine.py

# Continuer le developpement
/gsd:progress
```

## Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| Pas de ML lourd (PyTorch/TF) | Performance temps reel 15+ FPS prioritaire |
| 4 detecteurs hybrides | Couvrir differents types de manipulation |
| OpenCV Haar pour visages | Dependance legere, suffisant pour ROI |
| Frame dropping actif | Maintenir latence constante |

## Config

```json
{
  "mode": "yolo",
  "depth": "standard",
  "parallelization": true,
  "model_profile": "balanced"
}
```

---
*Pour reprendre: lance `/gsd:progress` ou lis ce fichier*
