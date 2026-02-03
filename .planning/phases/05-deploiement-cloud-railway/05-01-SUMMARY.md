# Phase 05 Plan 01: Configuration Railway Deployment Summary

**Phase:** 05-deploiement-cloud-railway
**Plan:** 01
**Completed:** 2026-02-03
**Duration:** ~2 minutes

## One-liner

Railway deployment configuration avec shell-form CMD pour $PORT expansion et railway.toml healthcheck/restart policy.

## What Was Built

### Configuration Modifications

1. **Dockerfile CMD Shell Form** (`Dockerfile`)
   - Modifié CMD de forme exec `["uvicorn", ...]` vers forme shell `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - Permet l'expansion de la variable `$PORT` injectée par Railway au runtime
   - Fix du bug "Application Failed to Respond" causé par l'absence d'expansion de variable en forme exec

2. **Railway-as-Code Configuration** (`railway.toml`)
   - Créé fichier de configuration Railway à la racine du projet
   - **Builder:** `DOCKERFILE` (force Railway à utiliser notre Dockerfile, pas Nixpacks)
   - **Healthcheck:** `/health` endpoint avec 300s timeout (laisse le temps à OpenCV de s'initialiser)
   - **Restart Policy:** `ON_FAILURE` avec 3 retries max (redémarrage automatique en cas de crash)

### Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Shell form CMD vs exec form | La forme shell (`CMD command arg1 arg2`) permet l'expansion des variables d'environnement comme `$PORT` |
| 300s healthcheck timeout | OpenCV et les dépendances système prennent du temps à charger au premier démarrage |
| ON_FAILURE restart policy | Redémarrage automatique uniquement en cas de crash (exit code != 0), pas sur stop manuel |
| Max 3 retries | Évite les boucles infinies de crash/restart, force l'investigation après 3 échecs |

## Dependency Graph

### Requires
- Phase 04 (production-hardening): `api.py` avec endpoint `/health` existant
- Phase 04: `Dockerfile` fonctionnel avec base python:3.12-slim

### Provides
- `railway.toml`: Configuration Railway pour healthcheck et restart policy
- `Dockerfile`: Compatible Railway avec $PORT dynamique

### Affects
- Plan 05-02: Déploiement Railway CLI/dashboard utilisera ces configs
- Futur monitoring: Healthcheck pointe vers `/health` qui retourne le status de l'engine

## Files Modified

### Created
- `railway.toml` (8 lignes) — Configuration Railway-as-Code

### Modified
- `Dockerfile` (1 ligne modifiée) — CMD ligne 17: forme shell avec $PORT

## Commits

| Commit | Type | Description | Files |
|--------|------|-------------|-------|
| 762fe89 | chore | Configure Railway deployment with shell-form CMD and railway.toml | Dockerfile, railway.toml |

## Deviations from Plan

None - plan executed exactly as written.

## Testing Notes

### Verification Performed
1. ✓ Dockerfile CMD vérifié: forme shell sans crochets JSON
2. ✓ railway.toml créé avec healthcheckPath, restartPolicyType, builder
3. ⚠ Docker build local non testé (Docker Desktop non démarré) — build sera validé par Railway lors du déploiement

### Expected Behavior
- Railway injectera `$PORT` au runtime (généralement 3000-8000)
- Healthcheck appellera `/health` toutes les N secondes après le timeout initial
- En cas de crash (exit code != 0), Railway redémarrera automatiquement (max 3 fois)

## Next Phase Readiness

### Ready for 05-02
- ✓ Dockerfile compatible Railway
- ✓ railway.toml configure
- ✓ Healthcheck pointe vers endpoint existant

### Potential Issues
- **Docker build non vérifié localement** — assumé fonctionnel basé sur les modifications minimales (1 ligne CMD changée)
- **Timeout 300s peut être trop long** — à ajuster après premiers déploiements si nécessaire

### Recommendations for Next Plan
1. Déployer sur Railway et observer les logs de build
2. Vérifier que $PORT est bien expansé au runtime
3. Tester le healthcheck après déploiement: `curl https://<app>.railway.app/health`
4. Monitorer les redémarrages automatiques s'il y en a

## Tech Stack Impact

### Added
- `railway.toml` (configuration Railway)

### Patterns Established
- **Infrastructure-as-Code Railway**: Configuration versionnée dans le repo (pas de config manuelle dashboard)
- **Shell-form CMD**: Pattern pour variables d'environnement dynamiques en Docker

### Dependencies
- Aucune nouvelle dépendance Python
- Railway CLI requis pour le prochain plan (installation dans 05-02)

## Performance Impact

- **Startup:** 300s healthcheck timeout = Railway attendra jusqu'à 5min avant de marquer le déploiement comme échoué
- **Resilience:** Restart automatique améliore la disponibilité (recovery en cas de crash temporaire)
- **No runtime impact:** Les modifications sont purement configuratives

## Knowledge for Future Sessions

### Railway Deployment Pattern
```dockerfile
# Shell form CMD pour expansion de variables
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
```

```toml
# railway.toml minimal
[build]
builder = "DOCKERFILE"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

### Debugging Tips
- Si "Application Failed to Respond": vérifier que CMD est en forme shell (pas de crochets)
- Si healthcheck échoue: vérifier que `/health` retourne 200 et que le timeout est suffisant
- Si redémarrages en boucle: augmenter maxRetries ou investiguer la cause du crash

## Success Metrics

- ✓ Dockerfile prêt pour Railway (CMD shell form avec $PORT)
- ✓ railway.toml créé avec configuration déploiement
- ⚠ Build Docker local non testé (non bloquant)

## Lessons Learned

1. **Shell vs Exec Form CMD**: La différence subtile entre `CMD ["cmd", "arg"]` (exec) et `CMD cmd arg` (shell) a un impact majeur sur l'expansion des variables
2. **Railway Healthcheck Timeout**: Les apps avec dépendances lourdes (OpenCV, ML) nécessitent des timeouts généreux au démarrage
3. **Infrastructure-as-Code**: Versionner la config Railway dans le repo évite la dérive entre environnements
