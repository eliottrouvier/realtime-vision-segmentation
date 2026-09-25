# Real-Time Video Detection & Instance Segmentation

Programme de détection d'objets et de segmentation d'instances en temps réel optimisé pour Apple Silicon (MPS), propulsé par **YOLO11-seg** / **OpenCV**.

---

## ⚡ Démarrage Rapide

```bash
# 1. Lancer la détection & segmentation sur la vidéo de démo (avec affichage en direct)
./.venv/bin/python video_segmentation.py --source samples/pedestrians.avi

# 2. Filtrer uniquement un objet spécifique (ex: uniquement les personnes)
./.venv/bin/python video_segmentation.py --source samples/pedestrians.avi --target person

# 3. Traiter et exporter la vidéo annotée
./.venv/bin/python video_segmentation.py --source samples/pedestrians.avi --output output_annotated.mp4
```

---

## 🎮 Contrôles au Clavier (Fenêtre en direct)

| Touche | Action |
| :---: | :--- |
| **`Espace`** ou **`p`** | Mettre en pause / Reprendre la lecture |
| **`s`** | Enregistrer un instantané (capture d'écran) de la frame courante |
| **`q`** ou **`Échap`** | Quitter le programme proprement |

---

## 🛠️ Options de la Ligne de Commande

| Option | Type | Défaut | Description |
| :--- | :---: | :---: | :--- |
| `--source` | `str` | `samples/pedestrians.avi` | Chemin vers la vidéo d'entrée (MP4, AVI, MOV...) |
| `--target` | `str` | `all` | Classes à filtrer séparées par des virgules (ex: `person`, `car,bus,truck` ou `all`) |
| `--conf` | `float` | `0.35` | Seuil de confiance minimal (entre 0.0 et 1.0) |
| `--model` | `str` | `yolo11n-seg.pt` | Modèle de segmentation Ultralytics (téléchargé automatiquement) |
| `--device` | `str` | `auto` (`mps`) | Puce de calcul (`mps` pour Mac Apple Silicon, `cuda`, ou `cpu`) |
| `--output` | `str` | `None` | Chemin de sortie pour enregistrer la vidéo résultante |
| `--no-display` | flag | `False` | Exécution sans interface graphique (idéal pour traitement batch/serveur) |

---

## 📈 Métriques & Performances

- **Matériel testé** : Apple Silicon (MPS)
- **Débit moyen** : **~25 - 32 FPS** en temps réel continu
- **Précision** : Masques polygonaux semi-transparents par instance + boîtes englobantes + labels et indices de confiance.
