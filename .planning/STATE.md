# Nilink - Project State

**Last updated:** 2026-02-03
**Status:** Engine + API + Production Hardening done — Tests passing (40/40)

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Detecter les manipulations visuelles en temps reel sans ML lourd
**Current focus:** Phase 4 complete, prochaine etape a definir

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

### Phase 2: API Layer ✓
- [x] FastAPI REST endpoint pour images (`POST /verify`, `/verify/base64`, `/verify/batch`)
- [x] WebSocket endpoint pour flux video (`/ws/stream`)
- [x] Serialisation JSON des resultats
- [x] Gestion erreurs et validation input
- [x] Health check endpoint (`GET /health`)
- [x] CORS middleware

### Phase 3: Testing & Validation ✓
- [x] Tests unitaires pour chaque detecteur (40/40 passent)
- [x] Tests API REST + WebSocket
- [x] Fix seuils de detection (ELA/upscaling pour images uniformes)
- [x] Compatibilite Python 3.14 (asyncio)
- [ ] Benchmark images vraies vs fakes (a completer)
- [ ] Tests de performance (FPS, latence)

## What's Next

### Phase 4: Production Hardening ✓
- [x] Configuration centralisee (`config.py` + `pydantic-settings` + `.env.example`)
- [x] Logging structure JSON (`logging_config.py` + middleware request_id)
- [x] Rate limiting par IP (`slowapi` — 60/min verify, 10/min batch)
- [x] Docker containerization (`Dockerfile` + `.dockerignore`)
- [x] Documentation API enrichie (tags, exemples, schemas erreurs, rate limits)

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
