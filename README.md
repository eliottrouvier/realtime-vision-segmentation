# Real-Time Video & Webcam Instance Segmentation

Studio de détection d'objets et de segmentation d'instances en temps réel optimisé pour Mac Apple Silicon (**MPS**), propulsé par **YOLO11-seg** / **OpenCV**.

---

## ⚡ Démarrage Rapide (`vision`)

La commande globale `vision` est installée dans votre terminal et accessible depuis n'importe quel dossier :

```bash
# 1. Ouvrir Vision Studio (interface tout-en-un avec panneau de réglages sur le côté)
vision

# 2. Ouvrir directement une vidéo dans le studio
vision chemin/vers/ma_video.mp4

# 3. Ouvrir directement la webcam dans le studio
vision --webcam
```

---

## 🎛️ Vision Studio — Tout-en-un avec Panneau Latéral

L'interface se divise en deux zones parfaitement intégrées :
- **À gauche : L'écran vidéo interactif**
  - Affichage direct du flux avec masques d'instances et boîtes englobantes.
  - **Barre de défilement temporelle (Timeline scrubber)** : cliquez ou glissez pour naviguer instantanément à n'importe quel moment de la vidéo.
  - **Contrôles de transport** :
    - `▶️ Lecture / ⏸️ Pause`
    - `🔄 Recommencer (0:00)`
    - **Sélecteur de vitesse** : `0.25x`, `0.5x`, `1x`, `1.5x`, `2x` (pour ralentir ou accélérer la vidéo).
    - `📸 Capture` : enregistre un instantané haute définition.
- **À droite : Le panneau latéral de réglages en direct**
  - **Source Vidéo** : basculez en 1 clic entre **📁 Fichier Vidéo** et **🎥 Webcam en direct**.
  - **Mode de Rendu** : basculez à la volée entre `Masques + Boîtes`, `Masques seuls` ou `Boîtes seules`.
  - **Filtres d'objets en direct** : filtrez instantanément les détections (`Tous`, `Personnes`, `Véhicules`, `Électronique`, `Objets du quotidien`, ou filtre personnalisé).
  - **Sensibilité** : curseur de seuil de confiance (10% à 90%).
  - **Télémétrie en direct** : FPS réel (accélération MPS Apple Silicon) et compteur d'objets.
  - **Bouton Quitter** : fermeture propre de l'application.

---

## 💻 Mode Terminal Sans GUI (CLI pur)

Pour exécuter un traitement en ligne de commande pure (par exemple en tâche de fond ou sur serveur) :

```bash
vision --cli samples/pedestrians.avi --target person --output resultat.mp4
```

---

## 📈 Performances Mesurées

- **Accélération matérielle** : Apple Silicon MPS (`mps`)
- **Cadence** : **25 à 32 FPS** en continu avec segmentation d'instances polygonales au pixel près.
