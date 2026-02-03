# Nilink - Project State

**Last updated:** 2026-02-03
**Status:** Milestone v1.0 complete — API deployee sur Railway

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Detecter les manipulations visuelles en temps reel sans ML lourd
**Current focus:** Milestone v1.0 complete
**Production URL:** https://nilink-production.up.railway.app

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

### Phase 4: Production Hardening ✓
- [x] Configuration centralisee (`config.py` + `pydantic-settings` + `.env.example`)
- [x] Logging structure JSON (`logging_config.py` + middleware request_id)
- [x] Rate limiting par IP (`slowapi` — 60/min verify, 10/min batch)
- [x] Docker containerization (`Dockerfile` + `.dockerignore`)
- [x] Documentation API enrichie (tags, exemples, schemas erreurs, rate limits)
- [x] `.gitignore` ajoute

### Phase 5: Deploiement Cloud Railway ✓
- [x] Dockerfile CMD shell-form pour expansion $PORT Railway
- [x] railway.toml (builder DOCKERFILE, healthcheck /health, restart ON_FAILURE)
- [x] Deploiement Railway avec HTTPS automatique
- [x] CI/CD GitHub push-to-deploy
- [x] Health monitoring actif
- [x] Documentation Swagger deployee sur /docs

## Architecture actuelle

```
nilink/
├── Nilink_engine.py    # Moteur forensique (4 detecteurs)
├── api.py              # FastAPI REST + WebSocket
├── config.py           # Configuration centralisee (pydantic-settings)
├── logging_config.py   # Logging JSON + middleware request_id
├── requirements.txt    # Dependencies Python
├── Dockerfile          # Container Docker (shell-form CMD pour Railway)
├── railway.toml        # Configuration Railway-as-Code
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
| Hosting | Railway (Hobby plan) |
| CI/CD | GitHub push-to-deploy |

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
| Shell-form CMD Docker | Permet expansion de $PORT pour Railway |
| railway.toml | Infrastructure-as-Code pour Railway (healthcheck, restart) |
| Railway Hobby plan | $5/mois, pas de cold starts, simplicite |

---
*Pour reprendre: lance `/gsd:progress` ou lis ce fichier*
