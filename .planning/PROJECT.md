# Nilink

## What This Is

Nilink est un moteur de verification forensique qui detecte les manipulations d'images et videos en temps reel : deepfakes, images generees par IA, inpainting, et upscaling artificiel. Expose via API (REST + WebSocket), il permet aux developpeurs, entreprises et particuliers d'evaluer l'authenticite d'un contenu visuel.

## Core Value

**Detecter les manipulations visuelles en temps reel avec une precision suffisante pour etre actionnable, sans dependre de modeles ML lourds.**

Si tout le reste echoue, le moteur doit pouvoir analyser une image et retourner un score de confiance fiable en moins de 66ms (15+ FPS).

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Moteur d'analyse forensique (`NilinkVerifierEngine`) capable de traiter images et flux video
- [ ] Detecteur ELA (Error Level Analysis) pour inpainting et zones generees
- [ ] Detecteur FFT (analyse spectrale) pour artefacts GAN/diffusion
- [ ] Detecteur rPPG (photoplethysmographie) pour liveness/deepfake sur visages
- [ ] Detecteur d'upscaling/restauration IA
- [ ] Detection de visage integree (OpenCV Haar/DNN)
- [ ] Output structure : score de confiance (0-1), liste d'anomalies, heatmap (image + coordonnees)
- [ ] Performance temps reel : 15+ FPS sur flux video
- [ ] Gestion du frame dropping pour maintenir la latence
- [ ] API REST pour analyse d'images individuelles
- [ ] API WebSocket pour flux video continu
- [ ] Mode demo webcam pour validation
- [ ] Suite de tests avec images connues (vraies + fakes)

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

## Constraints

- **Stack technique** : Python 3.10+, OpenCV, NumPy, SciPy — pas de PyTorch/TensorFlow pour v1
- **Performance** : 15+ FPS minimum, latence < 66ms par frame
- **Architecture** : Asynchrone (asyncio), non-bloquant, frame dropping si necessaire
- **Deploiement** : Doit tourner sur hardware standard (pas de GPU obligatoire pour v1)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pas de ML lourd pour v1 | Performance temps reel prioritaire, complexite de deploiement | — Pending |
| 4 detecteurs hybrides | Couvrir differents types de manipulation avec des approches complementaires | — Pending |
| OpenCV pour detection visage | Dependance legere, suffisant pour localiser ROI | — Pending |
| REST + WebSocket | REST pour images ponctuelles, WebSocket pour video temps reel | — Pending |
| Frame dropping actif | Maintenir latence constante plutot que traiter toutes les frames | — Pending |

---
*Last updated: 2026-02-02 after initialization*
