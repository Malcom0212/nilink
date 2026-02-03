# Feature Landscape

**Domain:** Forensic Image/Video Manipulation Detection
**Recherche effectuee:** 2026-02-02
**Confiance globale:** MEDIUM (verifie via WebSearch, necessite validation avec sources officielles pour certaines metriques)

---

## Table Stakes

Fonctionnalites que les utilisateurs attendent. Leur absence = le produit semble incomplet.

| Fonctionnalite | Pourquoi Attendue | Complexite | Notes |
|----------------|-------------------|------------|-------|
| **Detection multi-modale** | Standard de l'industrie 2026 - tous les leaders du marche (Sensity, Reality Defender) analysent images, videos, audio | Haute | Nilink se concentre sur image/video - conforme aux attentes |
| **Score de confiance** | Requis pour decisions utilisateur - tous les outils fournissent des scores de probabilite (0-100% ou classifications) | Basse | CRITIQUE: Sans score, utilisateurs ne peuvent pas evaluer la fiabilite |
| **API RESTful** | Integration d'entreprise standard - Reality Defender, Sensity, tous offrent des APIs cloud | Moyenne | Essentiel pour cas d'usage developpeurs |
| **Detection en temps reel (<100ms/frame)** | Smartphones traitent a 40-90ms/frame pour 10-15 FPS; systemes professionnels a 25-30 FPS | Haute | Objectif Nilink de 15+ FPS s'aligne avec normes industrie |
| **Analyse de compression (ELA)** | Technique forensique fondamentale depuis 2008+, presente dans tous les outils professionnels (FotoForensics, Ghiro) | Moyenne | Nilink l'inclut deja - table stakes verifie |
| **Analyse frequentielle (FFT/DCT)** | Complementaire a ELA, recherche 2026 montre F1-score ~0.81 avec DCT/FFT sur cartes ELA | Moyenne | Nilink l'inclut deja - table stakes verifie |
| **Traitement batch** | Entreprises analysent des volumes massifs - outils comme Ghiro concu pour "massive amounts of images" | Moyenne | Critique pour adoption enterprise |
| **Documentation claire** | Utilisateurs developpeurs exigent des docs d'API claires avec exemples | Basse | Non-technique mais bloquant pour adoption |

## Differentiateurs

Fonctionnalites qui distinguent le produit. Non attendues, mais valorisees.

| Fonctionnalite | Proposition de Valeur | Complexite | Notes |
|----------------|----------------------|------------|-------|
| **Detection sans ML lourd** | La plupart des outils 2026 dependent de deep learning - approche forensique legere = latence reduite + deploiement plus simple | Moyenne | AVANTAGE CLE: Nilink se differencie ici |
| **Heatmaps de manipulation** | Leaders du marche (Deep Media, Reality Defender) produisent des heatmaps visualisant les zones manipulees | Moyenne | Visualisation critique pour comprehension utilisateur |
| **Detection rPPG (deepfake video)** | Technique emergente analysant le flux sanguin pour detecter visages synthetiques | Haute | UNIQUE: Peu d'outils non-ML utilisent rPPG |
| **Detection d'upscaling** | Identifie images ameliorees par IA - marche niche mais en croissance avec outils comme Topaz/ESRGAN | Moyenne | DIFFERENTIATION: Rarement offert par concurrents |
| **Rapports forensiques exportables** | Formats structures (JSON/PDF) avec trails d'audit pour admissibilite en justice | Moyenne | Requis pour cas d'usage legal/entreprise |
| **Mode on-premise** | Alternative au cloud - Resolution de contraintes de confidentialite | Basse (API) | Reality Defender offre cloud/on-prem - avantage competitif |
| **Detection multi-techniques** | Combiner ELA+FFT+rPPG+upscaling = precision accrue vs detection single-layer | Haute | Recherche montre approches multi-couches reduisent faux positifs |
| **Explicabilite (XAI)** | Recherche 2026: systemes opaques inadmissibles en justice - SHAP/LIME pour transparence | Haute | CRITIQUE pour adoption legal/gouvernement |
| **Analyse de metadata** | Extraction EXIF/timestamps pour detecter incohérences temporelles ou modifications logicielles | Basse | Complementaire aux techniques pixel-level |
| **Detection inpainting** | Identifie zones "remplies" par IA (object removal, retouching) | Moyenne | Cas d'usage e-commerce/assurance en croissance |

## Anti-Features

Fonctionnalites a NE PAS construire deliberement. Erreurs courantes dans ce domaine.

| Anti-Feature | Pourquoi Eviter | Faire A La Place |
|--------------|-----------------|------------------|
| **Modeles ML proprietaires lourds** | Latence elevee (>500ms), couts GPU, dependance aux datasets d'entrainement, vulnerabilite aux adversarial attacks | Techniques forensiques classiques (ELA, FFT, rPPG) - plus rapides, explicables, robustes |
| **Detection face-only** | Echec avec manipulations non-faciales (inpainting, upscaling, backgrounds synthetiques) | Analyse full-frame avec techniques multi-domaines |
| **Scoring binaire (deepfake/real)** | Utilisateurs ont besoin de nuances - "65% confiance" vs "FAKE" absolu | Scores de probabilite gradues (0-100%) avec seuils configurables |
| **Analyse cloud obligatoire** | Problemes de confidentialite, latence reseau (200-800ms), conformite RGPD | API deployable on-premise ou edge |
| **Ignorer les modifications legitimes** | Faux positifs sur compression standard, overlays graphiques, editing normal = perte de confiance utilisateur | Filtres contextuels distinguant editing standard vs manipulation malveillante |
| **Black-box sans explicabilite** | Inadmissible en justice (norme DOJ 2026), manque de confiance utilisateur | Rapports detailles montrant QUELLES techniques ont detecte QUOI |
| **Training data bias** | Modeles ML echouent sur deepfakes hors-distribution ou nouvelles techniques (Sora/Veo) | Techniques agnostiques au generateur (proprietes physiques vs signatures modeles) |
| **Ignorer l'audio (video)** | Audio manipulation = vecteur d'attaque #1 en 2026 selon recherche | Pour video: analyser coherence audio-visuel (lip-sync, ambient sound) |
| **Pas de rate limiting** | APIs exposees a abus, couts serveur non-controles | Quotas par API key, throttling, tiers tarifaires |
| **Stocker les uploads utilisateur** | Risques de confidentialite, conformite complexe, couts stockage | Traitement ephemere - analyser puis supprimer immediatement |

## Dependances Entre Fonctionnalites

```
Flux de traitement:

1. Ingestion
   ├─ API RESTful (table stakes)
   └─ Traitement batch (table stakes)
         ↓
2. Pre-traitement
   ├─ Extraction metadata (differentiateur)
   └─ Frame extraction (video @ 25-30 FPS)
         ↓
3. Detection core
   ├─ ELA (table stakes) → FFT analysis (table stakes)
   ├─ rPPG analysis (differentiateur - video only)
   ├─ Upscaling detection (differentiateur)
   └─ Inpainting detection (differentiateur)
         ↓
4. Fusion & scoring
   ├─ Score de confiance (table stakes) ← Combine tous detecteurs
   └─ Heatmap generation (differentiateur) ← Localise manipulations
         ↓
5. Explicabilite
   ├─ Rapport forensique (differentiateur)
   └─ XAI explanations (differentiateur)
         ↓
6. Output
   ├─ JSON response avec scores
   └─ Optional: PDF forensic report
```

**Dependances critiques:**
- **Heatmaps** requierent detection par region (pas seulement score global)
- **XAI** requiert que chaque detecteur expose ses features intermediaires
- **Traitement batch** requiert orchestration asynchrone + queue management
- **rPPG** requiert extraction de ROI faciale (detection de visage en pre-traitement)

## Recommandation MVP

Pour le MVP, prioriser:

### Phase 1: Core Detection (Table Stakes)
1. **API RESTful single-image** - Endpoint `/detect` acceptant image
2. **ELA + FFT detection** - Deja implemente, base solide
3. **Score de confiance** - 0-100% avec classification (authentic/suspicious/manipulated)
4. **Documentation API** - Swagger/OpenAPI spec

### Phase 2: Real-time & Differentiation
5. **Upscaling detection** - Differentiation immediate vs concurrents
6. **Heatmap generation** - Visualisation critique pour comprehension
7. **Traitement batch** - Debloquer adoption enterprise

### Phase 3: Advanced Differentiation
8. **rPPG detection** (video) - Extension pour deepfake video
9. **Rapports forensiques exportables** - Cas d'usage legal
10. **Explicabilite XAI** - Admissibilite juridique

### Deferer post-MVP:
- **Metadata analysis** - Complementaire mais non-bloquant (Complexite: Basse, mais priorite moindre)
- **Inpainting detection** - Marche niche, valider demande client d'abord (Complexite: Moyenne)
- **Audio analysis** - Hors scope initial (focus image/video visuel)
- **Mode on-premise** - Ajouter apres validation cloud-first

## Matrice Complexite vs Impact

```
Impact Eleve │
            │ [Heatmaps]      [XAI]
            │ [Batch]         [rPPG]
            │
            │ [Scores]        [Multi-technique]
            │ [API REST]      [Forensic reports]
            │
Impact Bas  │ [Docs]          [Metadata]
            │                 [On-premise]
            └─────────────────────────────────
              Complexite      Complexite
              Basse           Haute
```

**Strategie recommandee:** Commencer quadrant bas-gauche → haut-gauche → haut-droite

## Considerations Performances

| Fonctionnalite | Target Metric | Rationale |
|----------------|---------------|-----------|
| Detection temps reel | 15+ FPS (66ms/frame) | Objectif projet aligne avec industrie (10-15 FPS smartphones, 25-30 FPS pro) |
| Latence API (single image) | <200ms total | Competitive avec outils cloud (hors reseau) |
| Traitement batch | 100+ images/min | Doit battre traitement sequentiel pour valeur ajoutee |
| Taille image max | 4K (3840x2160) | Standard moderne, mais offrir downsampling auto pour >4K |
| Formats supportes | JPEG, PNG, WebP (image); MP4, AVI (video) | Couvre 95%+ cas d'usage reels |

## Notes de Validation

**Sources de confiance MEDIUM:**
- Metriques FPS: Verifiees via recherche academique 2026 (smartphones 40-90ms/frame)
- Standards API: Confirmes via analyse concurrents (Sensity, Reality Defender, Hive AI)
- Techniques forensiques: ELA/FFT valides depuis 2008+, recherche recente confirme efficacite
- XAI requirements: Multiples publications academiques 2026 + DOJ framework

**Gaps necessitant validation:**
- Metriques exactes de Nilink (15+ FPS) - a valider via benchmarking reel
- Adoption marche de detection upscaling - donnees qualitatives, quantifier la demande
- Seuils de score optimaux (authentic vs suspicious vs manipulated) - requiert testing utilisateur

---

## Sources

### Detection Technologies & Performance
- [Unmasking digital deceptions: deepfake detection and multimedia forensics](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508882/)
- [CloudSEK: Best AI Deepfake Detection Tools 2026](https://www.cloudsek.com/knowledge-base/best-ai-deepfake-detection-tools)
- [Sensity AI: Best Deepfake Detection Software 2026](https://sensity.ai/)
- [WebProNews: 2026 AI Video Detection Advances](https://www.webpronews.com/2026-ai-video-detection-advances-combat-misinformation/)
- [Can AI Detect Deepfake Videos In Real Time](https://www.alibaba.com/product-insights/can-ai-detect-deepfake-videos-in-real-time-using-only-a-smartphone-camera.html)

### ELA/FFT Forensic Techniques
- [Effect of Frequency Features of ELA Maps (Springer 2026)](https://link.springer.com/chapter/10.1007/978-3-032-02088-8_22)
- [Forensic Analysis of AI-Generated Image Alterations](https://journal-isi.org/index.php/isi/article/view/1362)
- [FotoForensics: Error Level Analysis Tutorial](https://www.fotoforensics.com/tutorial-ela.php)

### AI Image Detection Tools Comparison
- [AI Image Detector Benchmark 2026](https://research.aimultiple.com/ai-image-detector/)
- [AU10TIX: AI Image Detector Tools](https://www.au10tix.com/blog/ai-image-detector-best-10-free-tools/)
- [SightEngine: Detect AI-Generated Images](https://sightengine.com/detect-ai-generated-images)

### Enterprise API Features
- [Reality Defender: Deepfake Detection](https://www.realitydefender.com/)
- [TechTarget: 5 Deepfake Detection Tools for Enterprise](https://www.techtarget.com/searchsecurity/tip/5-deepfake-detection-tools-to-protect-enterprise-users)
- [Gartner Peer Insights: Deepfake Detection Tools Reviews](https://www.gartner.com/reviews/market/deepfake-detection-tools)

### Visualization & Reporting
- [SoCRadar: Top 10 AI Deepfake Detection Tools 2025](https://socradar.io/blog/top-10-ai-deepfake-detection-tools-2025/)

### Batch Processing & Workflow
- [Ghiro: Automated Digital Image Forensics Tool](https://getghiro.org/)
- [Amped Software: Image and Video Forensics](https://ampedsoftware.com/)
- [IMATAG: Forensic Watermarking API](https://www.imatag.com/api/forensic-watermarking-api)

### Explainable AI & Transparency
- [Explainable AI for Digital Forensics: Ensuring Transparency](https://www.forensicscijournal.com/journals/jfsr/jfsr-aid1089.php)
- [XAI for Digital Forensics (Wiley)](https://wires.onlinelibrary.wiley.com/doi/am-pdf/10.1002/wfs2.1434)
- [Council on Criminal Justice: DOJ Report on AI](https://counciloncj.org/doj-report-on-ai-in-criminal-justice-key-takeaways/)

### Common Pitfalls
- [Partnership on AI: Manipulated Media Detection Requirements](https://partnershiponai.org/manipulated-media-detection-requires-more-than-tools-community-insights-on-whats-needed/)
- [TechPolicy.Press: Five Things 2025 Taught Us About AI Deception](https://www.techpolicy.press/five-things-2025-taught-us-about-ai-deception-and-detection/)
- [MIT Media Lab: Human Detection of Machine-Manipulated Media](https://www.media.mit.edu/articles/human-detection-of-machine-manipulated-media/)

### Market Analysis
- [Future Market Insights: Forensic Imaging Market 2035](https://www.futuremarketinsights.com/reports/forensic-imaging-market)
- [Fortune Business Insights: Forensic Technology Market 2032](https://www.fortunebusinessinsights.com/forensic-technology-market-110927)
