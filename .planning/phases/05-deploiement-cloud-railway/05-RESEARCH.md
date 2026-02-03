# Phase 5: Deploiement Cloud Railway - Research

**Researched:** 2026-02-03
**Domain:** Platform-as-a-Service (PaaS) deployment, Docker containerization, CI/CD
**Confidence:** HIGH

## Summary

Railway est une plateforme PaaS moderne qui simplifie drastiquement le deploiement d'applications containerisees. Pour Nilink, l'application FastAPI existante avec son Dockerfile est prete a etre deployee sans modification majeure. Railway detecte automatiquement les Dockerfiles, injecte les variables d'environnement necessaires (notamment `PORT`), genere des certificats SSL automatiquement, et offre monitoring/logs integres.

Les points cles identifies:
- **Dockerfile pret:** L'actuel `python:3.12-slim` avec dependances OpenCV (`libgl1`, `libglib2.0-0`) fonctionne sur Railway
- **Port binding critique:** Le CMD du Dockerfile doit utiliser la forme shell (`CMD uvicorn api:app --host 0.0.0.0 --port $PORT`) pour que la variable `$PORT` injectee par Railway soit correctement expansee
- **Zero configuration:** Railway detecte automatiquement le Dockerfile sans fichier `railway.toml` requis
- **Monitoring inclus:** Logs JSON structurees, metriques CPU/RAM/disk, alertes configurables via l'Observability Dashboard

**Primary recommendation:** Deployer via GitHub integration pour CI/CD automatique. Modifier le CMD du Dockerfile pour utiliser `$PORT` en forme shell. Configurer healthcheck sur `/health` avec timeout de 300s. Activer monitoring avec alertes sur CPU/RAM.

## Standard Stack

### Core Platform Components

| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| Railway CLI | v4.27.5 (Jan 2026) | Deployment automation | Official tool, active development |
| Dockerfile Builder | DOCKERFILE (Railpack) | Container build system | Nixpacks deprecated 2026, Railpack is current |
| Let's Encrypt SSL | Automatic | HTTPS certificate provisioning | Industry standard, zero-config on Railway |
| Uvicorn | 0.27.0+ | ASGI server for FastAPI | Production-ready, recommended by FastAPI docs |

### Supporting Services

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| railway.toml/json | Config as Code | Optional - only needed for advanced config (healthcheck, multi-region, cron) |
| Railway Observability Dashboard | Logs/metrics/alerts | Built-in - use for production monitoring |
| GitHub Integration | CI/CD automation | Recommended for auto-deploy on git push |
| Railway CLI | Manual deployment | Dev/testing or custom CI/CD workflows |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Railway | Render, Fly.io, Heroku | Railway: no cold starts, simpler config. Render: free tier. Fly.io: edge deployment. |
| GitHub Integration | CLI + GitHub Actions | CLI gives more control but requires workflow config |
| Dockerfile | Nixpacks auto-detect | Dockerfile gives full control over dependencies (critical for OpenCV) |

**Installation:**
```bash
# Railway CLI (optional, for manual deploys)
npm install -g @railway/cli
# or
curl -fsSL https://railway.app/install.sh | sh
```

## Architecture Patterns

### Recommended Project Structure

```
nilink/
├── Dockerfile              # Railway auto-detects this
├── .dockerignore          # Exclude .git, tests, .env
├── requirements.txt       # Python dependencies
├── config.py             # pydantic-settings reads from env vars
├── .env.example          # Template (NOT deployed)
├── api.py               # FastAPI app
├── Nilink_engine.py     # Core logic
└── railway.toml         # OPTIONAL config as code
```

### Pattern 1: Port Binding with Railway's $PORT Variable

**What:** Railway injecte automatiquement une variable `PORT` que l'application doit ecouter.

**When to use:** Toujours. C'est obligatoire pour que Railway route le traffic correctement.

**Example:**
```dockerfile
# ❌ INCORRECT - Forme exec n'expanse pas $PORT
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "$PORT"]

# ✅ CORRECT - Forme shell expanse $PORT
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Source:** [Railway PORT variable documentation](https://docs.railway.com/guides/public-networking), [Community experiences](https://medium.com/@tomhag_17/debugging-a-railway-deployment-my-journey-through-port-variables-and-configuration-conflicts-eb49cfb19cb8)

**Alternative approach:** Lire `PORT` dans le code Python:
```python
import os
port = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```

### Pattern 2: Healthcheck Configuration

**What:** Railway appelle repetitivement un endpoint HTTP jusqu'a recevoir un `200 OK` avant d'activer le nouveau deploiement.

**When to use:** Toujours en production pour zero-downtime deployments.

**Example (via railway.toml):**
```toml
[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

**Source:** [Railway Healthchecks documentation](https://docs.railway.com/reference/healthchecks)

**Note:** Le healthcheck n'est appele qu'au demarrage du deploiement, pas en continu. Pour monitoring continu, utiliser Railway Observability Dashboard avec alertes.

### Pattern 3: Environment Variables Injection

**What:** Railway injecte les variables d'environnement au build-time ET runtime.

**When to use:** Pour configuration sensible (secrets, API keys) et configuration par environnement (dev/prod).

**Example:**
```dockerfile
# Build-time variables (optional, pour ARG)
ARG RAILWAY_SERVICE_NAME
RUN echo "Building $RAILWAY_SERVICE_NAME"

# Runtime variables sont automatiquement disponibles
# Pas besoin de les declarer dans Dockerfile
```

**Code Python (config.py actuel - deja correct):**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_config = {"env_file": ".env"}  # Local dev
    PORT: int = 8000  # Railway override automatiquement
    LOG_LEVEL: str = "info"
    # ...
```

**Source:** [Railway Variables Guide](https://docs.railway.com/guides/variables)

**Variables Railway auto-injectees:**
- `PORT` - Port TCP sur lequel ecouter
- `RAILWAY_PUBLIC_DOMAIN` - Domaine public genere (ex: `nilink-prod.up.railway.app`)
- `RAILWAY_ENVIRONMENT_NAME` - Nom de l'environnement (production, staging, etc.)
- `RAILWAY_SERVICE_NAME`, `RAILWAY_PROJECT_ID`, etc.

### Pattern 4: CI/CD avec GitHub Integration

**What:** Railway detecte les pushs sur la branche liee et deploie automatiquement.

**When to use:** Production. Plus simple que CLI + GitHub Actions pour workflows standards.

**Workflow:**
1. Connecter repo GitHub a Railway
2. Selectionner branche (ex: `main`)
3. Railway build + deploy automatiquement a chaque push
4. Rollback facile via UI Railway

**Source:** [Railway CI/CD Guide](https://blog.railway.com/p/cicd-for-modern-deployment-from-manual-deploys-to-pr-environments)

**Alternative - CLI + GitHub Actions:**
```yaml
# .github/workflows/deploy.yml
name: Deploy to Railway
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: bervProject/railway-deploy@v1
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}
          service: nilink-api
```

**Tradeoff:** GitHub Actions donne plus de controle (tests pre-deploy, notifications Slack), mais GitHub integration Railway est plus simple (zero config).

### Pattern 5: OpenCV Dependencies dans Docker

**What:** OpenCV-python requiert des bibliotheques systeme (`libGL.so.1`, `libglib-2.0.so.0`) meme pour operations "headless".

**When to use:** Toujours avec opencv-python standard. Alternative: `opencv-python-headless` n'a pas ces dependances.

**Current Dockerfile (deja correct):**
```dockerfile
FROM python:3.12-slim

# OpenCV runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Source:** [OpenCV Docker issues](https://station.railway.com/questions/open-cv-fast-api-issue-2b333433), [Docker OpenCV guide](https://medium.com/@rekharahul.agarwal/why-your-dockerized-opencv-fails-and-how-to-think-like-an-engineer-ea2aadb28c09)

**Note:** Le `EXPOSE 8000` est documentaire seulement - Railway utilise `$PORT` dynamiquement.

### Anti-Patterns to Avoid

- **Hardcoding PORT=8000 dans CMD:** Railway injecte un port aleatoire, ignorer `$PORT` cause "Application Failed to Respond"
- **Forme exec JSON dans CMD:** `CMD ["uvicorn", ..., "$PORT"]` n'expanse PAS la variable, utiliser forme shell
- **Oublier libgl1 avec opencv-python:** Cause `ImportError: libGL.so.1` au runtime meme sans GUI
- **Commit .env avec secrets:** Railway injecte les variables, `.env` local ne doit JAMAIS etre commite
- **Utiliser Nixpacks en 2026:** Deprecated, passer a Dockerfile ou Railpack
- **Cloudflare proxy + SSL Strict:** Avec custom domain, utiliser SSL "Full" pas "Full Strict" ([source](https://docs.railway.com/guides/public-networking))

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSL certificates | Custom Let's Encrypt setup | Railway auto-provisioning | Automatic renewal, wildcard support, zero config |
| Load balancing | Custom nginx/haproxy | Railway automatic routing | Multi-region, health-aware routing included |
| Log aggregation | Custom ELK stack | Railway Observability Dashboard | Structured logs, filtering, search built-in |
| Secrets management | .env files in repo | Railway environment variables | Encrypted at rest, per-environment, no repo leaks |
| Zero-downtime deploys | Custom blue-green scripts | Railway healthchecks | Automatic traffic switching after healthcheck passes |
| Monitoring & alerts | Custom Prometheus + Grafana | Railway Monitors | CPU/RAM/disk/network alerts via email/webhook, no setup |

**Key insight:** Railway abstracts toute l'infrastructure. Ne pas reinventer Docker orchestration, service discovery, TLS termination, ou log shipping - tout est inclus et optimise.

## Common Pitfalls

### Pitfall 1: Application Failed to Respond Error

**What goes wrong:** Apres deploy, Railway marque le service "failed" avec erreur "Application failed to respond".

**Why it happens:**
1. Application n'ecoute pas sur `0.0.0.0:$PORT` (ex: hardcode `localhost:8000`)
2. CMD Dockerfile utilise forme exec qui n'expanse pas `$PORT`
3. Application crash avant d'ouvrir le port (dependances manquantes)
4. Healthcheck timeout (default 300s) depasse avant que l'app soit ready

**How to avoid:**
```python
# ✅ Lire PORT depuis env vars
import os
port = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```

```dockerfile
# ✅ Forme shell pour expansion de variables
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Warning signs:**
- Logs montrent "Uvicorn running on http://127.0.0.1:8000"
- Railway dashboard status reste "Deploying" puis passe a "Failed"
- Healthcheck logs: "Connection refused" ou timeout

**Source:** [Railway Troubleshooting](https://docs.railway.com/reference/errors/application-failed-to-respond)

### Pitfall 2: OpenCV ImportError in Production

**What goes wrong:** `ImportError: libGL.so.1: cannot open shared object file` au deploy, mais fonctionne localement.

**Why it happens:** Docker image slim ne contient pas les bibliotheques systeme GUI que OpenCV tente de charger dynamiquement, meme pour operations headless.

**How to avoid:**
```dockerfile
# ✅ Installer dependances OpenCV runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
```

**Alternative:** Utiliser `opencv-python-headless` dans `requirements.txt` si aucune fonctionnalite GUI n'est necessaire (mais Nilink utilise des operations image standard qui fonctionnent avec les deux).

**Warning signs:**
- Build Docker reussit, mais crash au premier `import cv2`
- Logs montrent "cannot open shared object file: libGL.so.1"
- Tests locaux passent mais deploy Railway echoue

**Source:** [Railway OpenCV Help Station](https://station.railway.com/questions/open-cv-fast-api-issue-2b333433)

### Pitfall 3: Environment Variable Not Expanding in Docker CMD

**What goes wrong:** Variable `$PORT` apparait litteralement comme string "$PORT" dans les logs au lieu d'etre expansee.

**Why it happens:** Docker CMD forme exec (`["cmd", "arg"]`) ne passe pas par un shell, donc pas d'expansion de variables.

**How to avoid:**
```dockerfile
# ❌ Forme exec - PAS d'expansion
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "$PORT"]

# ✅ Forme shell - expansion correcte
CMD uvicorn api:app --host 0.0.0.0 --port $PORT

# ✅ Alternative - shell explicite
CMD ["/bin/sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]
```

**Warning signs:**
- Logs montrent "Uvicorn running on http://0.0.0.0:$PORT" (literal)
- Railway healthcheck fail car wrong port

**Source:** [Docker CMD documentation](https://docs.docker.com/reference/dockerfile/#cmd), [Community debugging](https://medium.com/@tomhag_17/debugging-a-railway-deployment-my-journey-through-port-variables-and-configuration-conflicts-eb49cfb19cb8)

### Pitfall 4: Custom Domain SSL Certificate Stuck "Issuing"

**What goes wrong:** Apres ajout d'un custom domain, le certificat SSL reste en "Issuing TLS certificate" indefiniment.

**Why it happens:**
1. CNAME `_acme-challenge` manquant ou incorrect (requis pour Let's Encrypt validation)
2. DNS provider proxy le CNAME (Cloudflare orange cloud sur `_acme-challenge`)
3. DNS propagation incomplete (peut prendre jusqu'a 48h)

**How to avoid:**
1. Ajouter DEUX CNAME records:
   - `yourdomain.com` → `xyz.up.railway.app` (Railway value)
   - `_acme-challenge.yourdomain.com` → `authorize.railwaydns.net`
2. Si Cloudflare: desactiver proxy (grey cloud) sur `_acme-challenge`
3. Si Cloudflare proxy actif: mettre SSL/TLS mode sur "Full" (PAS "Full Strict")

**Warning signs:**
- Status "Issuing TLS certificate" > 10 minutes
- Railway UI ne montre pas green checkmark apres verification

**Source:** [Railway Custom Domain Guide](https://docs.railway.com/guides/public-networking#custom-domains), [SSL troubleshooting](https://station.railway.com/questions/ssl-certificate-error-on-custom-domain-f541bae6)

### Pitfall 5: Railway Hobby Plan Resource Limits Exceeded

**What goes wrong:** Service reduit performance ou crash avec "Out of memory" ou throttling CPU.

**Why it happens:** Hobby plan limite: 8 vCPU, 8 GB RAM, 5 GB storage par service. FastAPI + OpenCV peut consommer ~500MB RAM baseline + pics lors de traitement batch.

**How to avoid:**
1. Monitorer metriques Railway Dashboard (CPU/RAM graphs)
2. Configurer alertes a 70% RAM (via Monitors - requires Pro plan)
3. Optimiser batch endpoint: limiter `MAX_BATCH_SIZE=10` (deja fait)
4. Si depasse: upgrade vers Pro plan ($20/mois, 32 vCPU / 32 GB RAM)

**Warning signs:**
- Logs montrent "MemoryError" ou "Killed"
- Response times augmentent graduellement
- Railway dashboard metrics montrent RAM sustained > 90%

**Calcul usage Hobby plan:**
- Subscription: $5/mois (includes $5 credits)
- Usage: ~$0.000231/vCPU-hour, ~$0.000231/GB-hour
- Estimations: service 24/7 avec 1 vCPU + 2 GB RAM ≈ $10/mois (delta $5 facture)

**Source:** [Railway Pricing](https://docs.railway.com/reference/pricing/plans), [Hobby plan limits](https://www.saasworthy.com/product/railway-app/pricing)

## Code Examples

Verified patterns from official sources:

### Dockerfile Production-Ready for Railway

```dockerfile
# Source: Railway best practices + FastAPI Docker guide
FROM python:3.12-slim

# OpenCV system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Railway injects PORT dynamically
EXPOSE 8000

# ✅ Shell form for variable expansion
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Source:** [Railway FastAPI Guide](https://docs.railway.com/guides/fastapi), [OpenCV Docker patterns](https://medium.com/@albertqueralto/installing-opencv-within-docker-containers-for-computer-vision-and-development-a93b46996520)

### Railway Config as Code (railway.toml) - OPTIONAL

```toml
# Source: https://docs.railway.com/reference/config-as-code
# Only needed for advanced configuration

[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn api:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

**Note:** Railway detecte automatiquement le Dockerfile sans ce fichier. Utiliser `railway.toml` seulement pour:
- Healthcheck custom (path, timeout)
- Restart policy specifique
- Multi-region deployments
- Cron jobs

### Healthcheck Endpoint (deja implemente dans api.py)

```python
# Source: Current api.py implementation
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Railway deployment validation."""
    return HealthResponse(
        status="ok" if verifier else "not_ready",
        detectors={
            "ela": verifier.enable_ela if verifier else False,
            "fft": verifier.enable_fft if verifier else False,
            "rppg": verifier.enable_rppg if verifier else False,
            "upscale": verifier.enable_upscale_detection if verifier else False,
        },
    )
```

**Railway configuration:** Pas de configuration necessaire si endpoint est `/health`. Railway detecte automatiquement.

**Pour custom path:**
```toml
[deploy]
healthcheckPath = "/api/v1/health"  # Si different de /health
```

### Environment Variables Setup (Railway Dashboard)

```bash
# Variables a configurer dans Railway UI (Settings > Variables):

# Production configuration
LOG_LEVEL=info
RATE_LIMIT=60/minute
RATE_LIMIT_BATCH=10/minute
MAX_BATCH_SIZE=10
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
MAX_LATENCY_MS=66.0

# PORT et HOST sont auto-geres par Railway (NE PAS setter)
# Railway injecte automatiquement PORT
# HOST doit toujours etre 0.0.0.0 (hardcode dans config.py ok)
```

**Note:** Railway supporte aussi "Shared Variables" (scope project-wide) et "Reference Variables" (referencer autres services).

**Source:** [Railway Variables Guide](https://docs.railway.com/guides/variables)

### CLI Deployment Workflow

```bash
# Source: Railway CLI documentation v4.27.5

# 1. Install CLI
curl -fsSL https://railway.app/install.sh | sh

# 2. Login
railway login

# 3. Initialize project (or link existing)
railway init
# Select: "Create new project" > "Empty project"

# 4. Link to service
railway link
# Select project and service

# 5. Deploy
railway up
# Shows build logs, deployment URL at end

# 6. View logs
railway logs --tail

# 7. Open deployed service
railway open
```

**Source:** [Railway CLI Reference](https://docs.railway.com/reference/cli-api)

**Alternative - GitHub Integration (recommended):**
1. Railway Dashboard > New Project > Deploy from GitHub repo
2. Select `nilink` repo
3. Railway auto-detecte Dockerfile et deploie
4. Chaque push vers `main` (ou branche configuree) redeploy automatiquement

### Monitoring & Alerts Configuration

```python
# Railway Observability Dashboard - UI configuration only
# No code changes needed

# 1. Dashboard > Observability
# 2. Add Widget > "Logs" > Filter: `@level:error`
# 3. Add Widget > "Metrics" > CPU Usage (nilink-api)
# 4. Add Widget > "Metrics" > Memory Usage (nilink-api)

# 5. Configure Monitors (Alerts - requires Pro plan):
# - CPU > 80% for 5 minutes → Email alert
# - Memory > 90% for 3 minutes → Webhook to Slack
# - Disk > 85% → Email alert
```

**Structured Logging (deja implemente dans logging_config.py):**
```python
# Railway automatically indexes JSON logs
logger.info(
    "verify trust_score=%.4f anomalies=%d",
    result.global_trust_score,
    len(result.anomalies_found),
    extra={"trust_score": round(result.global_trust_score, 4)},
)

# Railway Log Explorer can filter:
# @level:info @trust_score:<0.5
```

**Source:** [Railway Observability Guide](https://docs.railway.com/guides/observability), [Monitoring blog post](https://blog.railway.com/p/using-logs-metrics-traces-and-alerts-to-understand-system-failures)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Nixpacks auto-builder | Dockerfile + Railpack | Q1 2026 | Nixpacks deprecated, Dockerfile preferred for control |
| Manual SSL certificate setup | Automatic Let's Encrypt | Since Railway launch | Zero config SSL, automatic renewal |
| Single-region deployment | Multi-region via railway.toml | 2025 | `multiRegionConfig` enables horizontal scaling |
| Basic logs in UI | Observability Dashboard | Jan 2025 ("Launch Week 01") | Unified logs/metrics/traces, advanced filtering |
| Environment variables in UI only | Config as Code (railway.toml) | 2024 | GitOps-friendly, version controlled config |
| Gunicorn + Uvicorn workers | Single Uvicorn process | Current best practice | Railway handles horizontal scaling, simpler setup |

**Deprecated/outdated:**
- **Nixpacks:** Maintenance mode depuis 2026, utiliser Dockerfile ou Railpack
- **railway.json with Nixpacks config:** Remplace par Dockerfile ou railway.toml with Railpack
- **Procfile:** Non supporte par Railway (utiliser `railway.toml` `startCommand` si besoin override)
- **Gunicorn multi-workers dans container:** Railway scale horizontalement (replicas), un worker Uvicorn par container suffit

**Emerging patterns:**
- **Monitors with webhooks:** Alertes vers Slack/Discord/PagerDuty (requires Pro plan)
- **PR Environments:** Railway cree environnements ephemeres par Pull Request (GitHub integration)
- **Config as Code:** Trend vers `railway.toml` commite pour reproductibilite

## Open Questions

1. **Multi-region deployment cost**
   - What we know: `railway.toml` supporte `multiRegionConfig` avec replica counts par region
   - What's unclear: Cost calculation (each replica counted separately?) et impact sur Hobby plan limits (8vCPU total ou par replica?)
   - Recommendation: Commencer single-region (auto-selected nearest), evaluer multi-region si latency EMEA/APAC critique

2. **Railway-provided PORT range**
   - What we know: Railway injecte variable `PORT` automatiquement
   - What's unclear: Range de ports utilise (ephemeral? fixe?)
   - Recommendation: Non-critical. Toujours utiliser `$PORT` variable, jamais hardcoder

3. **Observability Dashboard alerting without Pro plan**
   - What we know: Monitors (alertes) requierent Pro plan ($20/mois)
   - What's unclear: Si Hobby plan peut utiliser external monitoring (ex: UptimeRobot webhook → service)
   - Recommendation: Commencer Hobby sans alertes, monitorer manuellement. Upgrade Pro si alertes critiques

4. **OpenCV performance sur Railway shared infrastructure**
   - What we know: Hobby plan shared vCPU, traitement image CPU-intensive
   - What's unclear: Impact throttling CPU sur latency target 66ms (15 FPS)
   - Recommendation: Load test apres deploy initial, monitorer p95 latency. Si degradation, considerer Pro plan dedicated resources

5. **WebSocket scaling avec Railway load balancer**
   - What we know: WebSocket endpoint `/ws/stream` fonctionne, Railway supporte WebSockets
   - What's unclear: Comportement avec multiple replicas (sticky sessions? connection draining?)
   - Recommendation: Single replica initially (Hobby plan). Si scale WebSocket necessaire, tester multi-replica behavior ou utiliser dedicated WebSocket service

## Sources

### Primary (HIGH confidence)

- [Railway Dockerfile Guide](https://docs.railway.com/guides/dockerfiles) - Official Docker deployment documentation
- [Railway Config as Code Reference](https://docs.railway.com/reference/config-as-code) - Complete railway.toml schema
- [Railway FastAPI Guide](https://docs.railway.com/guides/fastapi) - Official FastAPI deployment guide
- [Railway Public Networking Guide](https://docs.railway.com/guides/public-networking) - PORT variable, domains, SSL
- [Railway Variables Reference](https://docs.railway.com/reference/variables) - Environment variables documentation
- [Railway Healthchecks Reference](https://docs.railway.com/reference/healthchecks) - Health check configuration
- [Railway Observability Guide](https://docs.railway.com/guides/observability) - Monitoring and logging
- [Railway CLI Reference](https://docs.railway.com/reference/cli-api) - CLI commands (v4.27.5)
- [Railway Pricing Plans](https://docs.railway.com/reference/pricing/plans) - Hobby plan limits and costs

### Secondary (MEDIUM confidence)

- [Railway Observability Blog Post](https://blog.railway.com/p/using-logs-metrics-traces-and-alerts-to-understand-system-failures) - Jan 2026 deep dive
- [Railway CI/CD Blog Post](https://blog.railway.com/p/cicd-for-modern-deployment-from-manual-deploys-to-pr-environments) - Deployment patterns
- [Railway OpenCV Help Station](https://station.railway.com/questions/open-cv-fast-api-issue-2b333433) - Community troubleshooting
- [Debugging Railway PORT Variable](https://medium.com/@tomhag_17/debugging-a-railway-deployment-my-journey-through-port-variables-and-configuration-conflicts-eb49cfb19cb8) - Real-world issues
- [FastAPI Docker Official Guide](https://fastapi.tiangolo.com/deployment/docker/) - FastAPI best practices
- [OpenCV Docker Installation Guide](https://medium.com/@albertqueralto/installing-opencv-within-docker-containers-for-computer-vision-and-development-a93b46996520) - System dependencies

### Tertiary (LOW confidence)

- [Railway vs Render Comparison](https://northflank.com/blog/railway-vs-render) - Third-party platform comparison
- [Railway Hosting 2026 Overview](https://kuberns.com/blogs/post/railway-hosting-explained/) - Independent analysis
- [SaaS Price Pulse Railway Pricing](https://www.saaspricepulse.com/tools/railway) - Pricing aggregator

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** - Official Railway docs current (Jan 2026), FastAPI patterns well-documented
- Architecture: **HIGH** - Dockerfile working locally, Railway deployment patterns verified in official guides
- Pitfalls: **HIGH** - Common issues documented in official troubleshooting + community help station

**Research date:** 2026-02-03
**Valid until:** 2026-03-03 (30 days - Railway stable platform, monthly updates expected)

**Next steps for planner:**
1. Create Dockerfile modification task (CMD shell form for $PORT)
2. Create Railway project initialization task (GitHub integration)
3. Create environment variables configuration task
4. Create healthcheck verification task
5. Create monitoring setup task (Dashboard widgets)
6. Create custom domain + SSL task (si applicable)
7. Create load testing task (verify 15 FPS target met)
