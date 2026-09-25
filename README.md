# Vision Studio — Suite Multi-Modes de Vision par Ordinateur

Studio de vision par ordinateur tout-en-un optimisé pour Mac Apple Silicon (**MPS**), intégrant segmentation d'instances, détection universelle en vocabulaire ouvert et estimation de posture en temps réel.

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

### 3. 🦴 Squelette & Articulations Corporelles (`YOLO-Pose`)
- Reconnaissance et suivi en direct des **17 articulations majeures** du corps humain (épaules, coudes, poignets, hanches, genoux, chevilles, tête).
- Tracé du squelette (membres et posture) avec points d'articulation lumineux.
- Parfait pour voir en direct les mouvements des bras, des jambes et la posture via la webcam ou la vidéo.

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
