# Nilink - Project State

**Last updated:** 2026-02-03
**Status:** v1.0 Foundation shipped — planning next milestone

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-03)

**Core value:** Detecter les manipulations visuelles en temps reel sans ML lourd
**Current focus:** Planning next milestone
**Production URL:** https://nilink-production.up.railway.app

## Current Position

Phase: Next milestone not started
Plan: Not started
Status: Ready to plan
Last activity: 2026-02-03 — v1.0 milestone complete

## Shipped Milestones

### v1.0 Foundation (2026-02-03)
- 6 phases (0-5), 27 fichiers, 4,953 lignes
- Moteur forensique 4 detecteurs + API REST/WebSocket + deploiement Railway
- See: .planning/MILESTONES.md

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
    ├── PROJECT.md      # Vision et specs projet
    └── MILESTONES.md   # Historique milestones
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

## Open Items

- Performance production ~1.9s/frame (Railway Hobby CPU shared)
- CORS_ORIGINS permissif (a configurer pour production)

---
*Pour reprendre: lance `/gsd:new-milestone` pour planifier le prochain milestone*
