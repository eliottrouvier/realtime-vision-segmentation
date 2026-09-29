# Vision Studio — Suite Multi-Modes de Vision par Ordinateur

> **Tech Stack : Python • PyTorch (MPS) • YOLO11 • YOLO-World • MediaPipe • PySide6 • OpenCV**  
> All-in-one real-time computer vision suite accelerated on Apple Silicon MPS, integrating instance segmentation, open-vocabulary detection, and multi-granularity pose tracking.

---

## ⚡ Démarrage Rapide (`vision`)

La commande globale `vision` est installée dans votre terminal et accessible depuis n'importe quel dossier :

```bash
# Ouvrir Vision Studio (démarre par défaut sur la vidéo des piétons en pause)
vision

# Ouvrir directement une vidéo spécifique
vision chemin/vers/ma_video.mp4

# Ouvrir directement en mode webcam
vision --webcam
```

---

## 🎛️ 3 Modes de Vision Disponibles (Onglets Supérieurs)

Basculez instantanément en un clic entre 3 paradigmes de vision grâce aux onglets en haut de la fenêtre :

### 1. 🟦 Segmentation d'Instances (`YOLO11-seg`)
- Segmentation polygonale au pixel près sur 80 classes COCO.
- **Affichage modulaire** : `Masques + Boîtes`, `Masques seuls` ou `Boîtes seules`.
- **Filtres d'objets rapides** : Personnes, Véhicules, Électronique, Objets du quotidien.

### 2. 🌍 Détection Universelle & Vocabulaire Ouvert (`YOLO-World`)
- Détecte des centaines d'objets détaillés (montre, lunettes, veste, chaussures, clavier, tasse, etc.).
- **Saisie en langage naturel** : tapez n'importe quel mot en texte libre pour le détecter en temps réel sur le flux vidéo ou la webcam !
- **Boutons de presets rapides** : Vêtements, Bureau, Général.

### 3. 🦴 Squelette & Articulations Corporelles (Double Moteur)
Basculez dans le panneau latéral entre deux niveaux de granularité selon vos besoins :
- **⚡ Squelette Rapide (17 points - YOLO11-Pose)** : Détection temps réel ultra-rapide des 17 articulations majeures (épaules, coudes, poignets, hanches, genoux, chevilles, tête).
- **🔬 Haute Précision (75+ points - MediaPipe Holistic)** :
  - **Corps complet (33 points)** : intègre les pouces, index, talons, chevilles et orteils pour chaque membre.
  - **Mains & Doigts ultra-détaillés (42 points)** : chaque phalange, articulation et bout des doigts est tracé indépendamment (21 points par main).
  - **Maillage facial expressif (478 points)** : contours délicats du visage, des yeux et des lèvres activables à la demande.

---

## 🎬 Vidéos de Démonstration Incluses

Un sélecteur rapide directement accessible dans le panneau latéral permet de basculer instantanément entre plusieurs séquences d'analyse :

1. **🚶 Piétons (`pedestrians.avi`)** : Surveillance urbaine, idéal pour la segmentation multi-classes et YOLO-World.
2. **💃 Danse & Posture (`dance_movement.mp4`)** : Mouvements amples, déplacements corporels et gestuelle fluide pour l'analyse articulaire.
3. **🏋️ Fitness & Flexions (`squats_workout.mp4`)** : Squats en 1080p HD, idéal pour observer la cinématique des genoux, hanches et coudes.
4. **🤸 Gainage & Pompes (`pushups_workout.mp4`)** : Mouvements horizontaux sur ballon de gym, analyse de l'alignement du rachis et des bras.

---

## 🎮 Contrôles de Lecture & Panneau Latéral

- **Barre temporelle (Scrubber)** : naviguez instantanément à n'importe quel moment de la vidéo.
- **Contrôles de transport** : `▶️ Lecture / ⏸️ Pause`, `↺ Recommencer (0:00)`, sélecteur de vitesse (`0.25x`, `0.5x`, `1x`, `1.5x`, `2x`).
- **Capture HD** : bouton pour sauvegarder l'image courante en haute définition.
- **Bascule Source** : passez en un clic de `Fichier Vidéo` à `Webcam Direct` (caméra FaceTime HD native macOS via AVFoundation).
- **Sensibilité** : curseur de confiance de 10% à 90%.

---

## 📈 Performances

- **Accélération matérielle** : Apple Silicon MPS (`mps`)
- **Cadence** : **25 à 35+ FPS** en temps réel continu.
