"""
Vision Entrypoint (Smart CLI & Studio GUI Dispatcher)
-----------------------------------------------------
Usage:
    vision                         -> Opens Vision Studio (all-in-one with side controls)
    vision video.mp4               -> Opens Vision Studio loaded with video.mp4
    vision --webcam                -> Opens Vision Studio directly in live webcam mode
    vision --cli [args...]         -> Runs raw OpenCV terminal mode
"""

import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)


def main():
    args = sys.argv[1:]

    # 1. Raw CLI mode request
    if "--cli" in args:
        cli_args = [a for a in args if a != "--cli"]
        if "--webcam" in cli_args or "-w" in cli_args:
            from webcam_segmentation import main as webcam_main
            sys.argv = [sys.argv[0]] + [a for a in cli_args if a not in ["--webcam", "-w"]]
            webcam_main()
        else:
            from video_segmentation import main as video_main
            sys.argv = [sys.argv[0]] + cli_args
            video_main()
        return

    # 2. Help request
    if "--help" in args or "-h" in args:
        print("""
Vision Studio — Détection & Segmentation Temps Réel (YOLO11-seg)
----------------------------------------------------------------
Usage:
  vision                           Ouvre l'interface Vision Studio (avec panneau de réglages latéral)
  vision <video_path>              Ouvre Vision Studio avec la vidéo spécifiée
  vision --webcam (ou -w)          Ouvre Vision Studio en mode webcam en direct

Contrôles dans le panneau latéral :
  - Vitesse de lecture (0.25x à 2x)
  - Boutons Play/Pause, Recommencer au début (0:00) et Timeline scrubber
  - Mode d'affichage : Masques + Boîtes, Masques seuls, Boîtes seules
  - Filtre d'objets en direct (Personnes, Véhicules, Électronique, Objets ou personnalisé)
  - Curseur de seuil de confiance
  - Bascule instantanée Source Vidéo <-> Webcam
  - Bouton Quitter
""")
        return

    # 3. Launch Vision Studio GUI
    import app_gui
    app_gui.main()


if __name__ == "__main__":
    main()
