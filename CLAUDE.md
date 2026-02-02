# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About Me

- **Prenom** : Matys
- **Langues** : Francais (natif), Anglais
- **Communication** : Repondre en francais par defaut

## Profil technique

- **Niveau** : Non-developpeur, utilise des outils no-code/low-code
- **Stack actuelle** : Vapi (assistants vocaux) + n8n (automatisation)

## Projet Nilink

**Vision:** Couche de confiance universelle du web — dissocier le vrai du faux en temps reel.

**Produit:** Moteur de verification forensique pour detecter manipulations d'images/videos (deepfakes, images IA, inpainting, upscaling). API SaaS pour developpeurs, entreprises et particuliers.

**Stack:** Python 3.10+, OpenCV, NumPy, SciPy (pas de PyTorch/TF pour v1)

**Performance cible:** 15+ FPS temps reel

### Etat actuel

- **Core engine:** `Nilink_engine.py` implemente (4 detecteurs: ELA, FFT, rPPG, upscaling)
- **Prochaine etape:** API REST/WebSocket avec FastAPI
- **Docs planning:** `.planning/STATE.md` pour l'etat complet

### Commandes utiles

```bash
# Tester le moteur
python Nilink_engine.py

# Voir l'etat du projet
cat .planning/STATE.md

# Reprendre le workflow GSD
/gsd:progress
```
