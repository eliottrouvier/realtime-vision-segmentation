# Real-Time Video & Webcam Instance Segmentation

Programme de détection d'objets et de segmentation d'instances en temps réel optimisé pour Mac Apple Silicon (**MPS**), propulsé par **YOLO11-seg** / **OpenCV**.

---

## ⚡ Utilisation Universelle (`vision`)

La commande globale `vision` est installée dans votre terminal et accessible depuis n'importe quel dossier :

```bash
# 1. Ouvrir l'interface graphique de bureau (GUI sombre avec boutons & filtres)
vision

# 2. Lancer directement le traitement d'une vidéo
vision samples/pedestrians.avi

# 3. Lancer directement la webcam en direct
vision --webcam

# 4. Filtrer des objets précis en ligne de commande
vision samples/pedestrians.avi --target person
vision --webcam --target person,"cell phone",cup
```

---

## 🖥️ Interface Graphique Native de Bureau (`app_gui.py`)

En tapant simplement `vision`, une interface native sombre s'ouvre :
- **🎥 Lancer la Webcam** en 1 clic.
- **📁 Ouvrir une Vidéo** via le sélecteur natif de fichiers macOS.
- **Cases à cocher thématiques** : Personnes, Véhicules, Électronique, Objets du quotidien ou 80 classes COCO.
- **Filtre personnalisé** : saisissez n'importe quel mot-clé (ex: `dog, backpack`).
- **Curseur de confiance interactif** (10% à 90%).
- **Mode miroir et option d'enregistrement vidéo**.

---

## 🎮 Contrôles au Clavier (Fenêtre en direct)

| Touche | Action |
| :---: | :--- |
| **`Espace`** ou **`p`** | Mettre en pause / Reprendre la lecture |
| **`s`** | Enregistrer un instantané (capture d'écran) de la frame courante |
| **`f`** | *(Webcam)* Activer / désactiver le mode miroir horizontal |
| **`q`** ou **`Échap`** | Quitter le programme proprement |

---

## 🛠️ Options de la Ligne de Commande

| Option | Type | Défaut | Description |
| :--- | :---: | :---: | :--- |
| `source` | `str` | `samples/pedestrians.avi` | Chemin vers la vidéo d'entrée (MP4, AVI, MOV...) |
| `--webcam` | flag | `False` | Ouvre directement le flux de la webcam |
| `--target` | `str` | `all` | Classes à filtrer séparées par des virgules (ex: `person`, `car,bus,truck` ou `all`) |
| `--conf` | `float` | `0.35` | Seuil de confiance minimal (entre 0.0 et 1.0) |
| `--model` | `str` | `yolo11n-seg.pt` | Modèle Ultralytics (téléchargé automatiquement) |
| `--device` | `str` | `auto` (`mps`) | Puce de calcul (`mps` pour Mac Apple Silicon, `cuda`, ou `cpu`) |
| `--output` | `str` | `None` | Chemin de sortie pour enregistrer la vidéo résultante |
| `--no-mirror` | flag | `False` | Désactive le miroir horizontal en mode webcam |

---

## 📈 Métriques & Performances

- **Matériel testé** : Apple Silicon (MPS)
- **Débit continu** : **~25 - 32 FPS** en temps réel
- **Segmentation** : Masques polygonaux semi-transparents par instance au pixel près.
