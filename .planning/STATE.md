# Nilink - Project State

**Last updated:** 2026-02-03
**Status:** Phases 0-4 completes — 40/40 tests passing

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Detecter les manipulations visuelles en temps reel sans ML lourd
**Current focus:** Phases 0-4 completes, prochaine etape a definir

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
- [x] FastAPI REST endpoints (`POST /verify`, `/verify/base64`, `/verify/batch`)
- [x] WebSocket endpoint pour flux video (`/ws/stream`)
- [x] Serialisation JSON des resultats
- [x] Gestion erreurs et validation input
- [x] Health check endpoint (`GET /health`)
- [x] CORS middleware

### Phase 3: Testing & Validation ✓
- [x] Tests unitaires pour chaque detecteur (26 tests engine)
- [x] Tests API REST + WebSocket (14 tests API)
- [x] Fix seuils de detection (ELA/upscaling pour images uniformes)
- [x] Compatibilite Python 3.14 (asyncio)
- [ ] Benchmark images vraies vs fakes (a completer)
- [ ] Tests de performance (FPS, latence)

### Phase 4: Production Hardening ✓
- [x] Configuration centralisee (`config.py` + `pydantic-settings` + `.env.example`)
- [x] Logging structure JSON (`logging_config.py` + middleware request_id)
- [x] Rate limiting par IP (`slowapi` — 60/min verify, 10/min batch)
- [x] Docker containerization (`Dockerfile` + `.dockerignore`)
- [x] Documentation API enrichie (tags, exemples, schemas erreurs, rate limits)
- [x] `.gitignore` ajoute

## What's Next

A definir — pistes possibles :
- Phase 5 : Frontend / dashboard
- Phase 5 : Deploiement cloud (fly.io, Railway, etc.)
- Phase 5 : Benchmark & optimisation performance
- Phase 5 : Extension navigateur

## Architecture actuelle

```
nilink/
├── Nilink_engine.py    # Moteur forensique (4 detecteurs)
├── api.py              # FastAPI REST + WebSocket
├── config.py           # Configuration centralisee (pydantic-settings)
├── logging_config.py   # Logging JSON + middleware request_id
├── requirements.txt    # Dependencies Python
├── Dockerfile          # Container Docker
├── .dockerignore       # Exclusions Docker
├── .env.example        # Template variables d'environnement
├── .gitignore          # Exclusions Git
├── CLAUDE.md           # Instructions Claude Code
├── tests/
│   ├── test_engine.py  # 26 tests moteur
│   └── test_api.py     # 14 tests API
└── .planning/
    ├── STATE.md        # Ce fichier
    └── PROJECT.md      # Vision et specs projet
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.10+ |
| Moteur forensique | OpenCV, NumPy, SciPy |
| API | FastAPI + Uvicorn |
| Config | pydantic-settings + .env |
| Rate limiting | slowapi |
| Logging | stdlib logging + JSON formatter |
| Container | Docker (python:3.12-slim) |
| Tests | pytest + httpx (40 tests) |

## Quick Resume Commands

```bash
# Tester
pytest tests/ -v

# Lancer l'API
uvicorn api:app --host 0.0.0.0 --port 8000

# Build Docker
docker build -t nilink .

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
| pydantic-settings | Config type-safe avec .env support natif |
| slowapi | Rate limiting simple, integre a FastAPI |
| JSON logging | Parsable par outils monitoring (ELK, Datadog, etc.) |

---
*Pour reprendre: lance `/gsd:progress` ou lis ce fichier*
