# Nilink

## What This Is

Nilink est un moteur de verification forensique qui detecte les manipulations d'images et videos en temps reel : deepfakes, images generees par IA, inpainting, et upscaling artificiel. Expose via API REST + WebSocket et deploye en production sur Railway, il permet aux developpeurs, entreprises et particuliers d'evaluer l'authenticite d'un contenu visuel.

## Core Value

**Detecter les manipulations visuelles en temps reel avec une precision suffisante pour etre actionnable, sans dependre de modeles ML lourds.**

Si tout le reste echoue, le moteur doit pouvoir analyser une image et retourner un score de confiance fiable en moins de 66ms (15+ FPS).

## Requirements

### Validated

- ✓ Moteur d'analyse forensique (`NilinkVerifierEngine`) capable de traiter images et flux video — v1.0
- ✓ Detecteur ELA (Error Level Analysis) pour inpainting et zones generees — v1.0
- ✓ Detecteur FFT (analyse spectrale) pour artefacts GAN/diffusion — v1.0
- ✓ Detecteur rPPG (photoplethysmographie) pour liveness/deepfake sur visages — v1.0
- ✓ Detecteur d'upscaling/restauration IA — v1.0
- ✓ Detection de visage integree (OpenCV Haar/DNN) — v1.0
- ✓ Output structure : score de confiance (0-1), liste d'anomalies, heatmap (image + coordonnees) — v1.0
- ✓ Performance temps reel : 15+ FPS sur flux video — v1.0 (mecanisme correct, hardware-dependent)
- ✓ Gestion du frame dropping pour maintenir la latence — v1.0
- ✓ API REST pour analyse d'images individuelles — v1.0
- ✓ API WebSocket pour flux video continu — v1.0
- ✓ Mode demo webcam pour validation — v1.0
- ✓ Suite de tests avec images connues (vraies + fakes) — v1.0

### Active

(A definir pour le prochain milestone)

### Out of Scope

- Modeles Deep Learning lourds (PyTorch/TensorFlow) — complexite et latence incompatibles avec l'objectif temps reel v1
- Interface utilisateur graphique — focus sur l'API d'abord
- Analyse audio/voix — focus sur le visuel pour v1
- Stockage/historique des analyses — stateless pour v1
- Authentification/facturation API — infrastructure SaaS viendra apres le moteur

## Context

**Probleme :** La proliferation des deepfakes et contenus generes par IA rend difficile la distinction entre vrai et faux. Les outils existants sont soit trop lents (modeles ML lourds), soit peu fiables (heuristiques simples).

**Approche Nilink :** Combine plusieurs signaux forensiques mathematiques (bruit capteur, analyse spectrale, signaux biologiques) pour une detection rapide sans ML lourd. L'approche hybride permet de couvrir differents types de manipulation.

**Signaux forensiques utilises :**
- **ELA (Error Level Analysis)** : Les zones generees par IA ont une signature de compression differente du reste de l'image
- **FFT (Fast Fourier Transform)** : Les GANs et modeles de diffusion laissent des artefacts periodiques (grilles) visibles dans le spectre haute frequence
- **rPPG (Remote Photoplethysmography)** : Les deepfakes ne reproduisent pas les micro-variations de couleur dues au flux sanguin
- **Analyse de bruit** : L'upscaling IA "invente" des details qui n'ont pas le bruit de capteur coherent d'une vraie photo HD

**Cibles utilisateurs :**
- Developpeurs integrant la verification dans leurs apps
- Entreprises (plateformes sociales, medias, assurances)
- Particuliers (extension navigateur future, app mobile)

**Etat actuel (post-v1.0) :**
- v1.0 shipped le 2026-02-03 avec 4,953 lignes de code
- Stack: Python 3.10+, FastAPI, OpenCV, NumPy, SciPy
- 40 tests (26 engine + 14 API)
- API deployee sur Railway : https://nilink-production.up.railway.app
- Dette technique : performance ~1.9s/frame en production (CPU shared), CORS permissif

## Constraints

- **Stack technique** : Python 3.10+, OpenCV, NumPy, SciPy — pas de PyTorch/TensorFlow pour v1
- **Performance** : 15+ FPS minimum, latence < 66ms par frame
- **Architecture** : Asynchrone (asyncio), non-bloquant, frame dropping si necessaire
- **Deploiement** : Doit tourner sur hardware standard (pas de GPU obligatoire pour v1)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pas de ML lourd pour v1 | Performance temps reel prioritaire, complexite de deploiement | ✓ Good — moteur fonctionne sans GPU |
| 4 detecteurs hybrides | Couvrir differents types de manipulation avec des approches complementaires | ✓ Good — couvre ELA/FFT/rPPG/upscaling |
| OpenCV pour detection visage | Dependance legere, suffisant pour localiser ROI | ✓ Good — Haar cascade suffisant pour v1 |
| REST + WebSocket | REST pour images ponctuelles, WebSocket pour video temps reel | ✓ Good — les deux endpoints fonctionnels |
| Frame dropping actif | Maintenir latence constante plutot que traiter toutes les frames | ✓ Good — compense la latence en production |
| pydantic-settings + .env | Config type-safe avec support natif .env | ✓ Good — config centralisee et flexible |
| slowapi pour rate limiting | Rate limiting simple integre a FastAPI | ✓ Good — 60/min verify, 10/min batch |
| JSON logging | Parsable par outils monitoring (ELK, Datadog) | ✓ Good — structurated logging operationnel |
| Shell-form CMD Docker | Permet expansion de $PORT pour Railway | ✓ Good — fix du "Application Failed to Respond" |
| Railway Hobby plan | $5/mois, pas de cold starts, simplicite | ✓ Good — deploiement fonctionnel |

---
*Last updated: 2026-02-03 after v1.0 milestone*
