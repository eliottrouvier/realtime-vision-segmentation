"""
Vision Entrypoint (Smart CLI & GUI Dispatcher)
----------------------------------------------
Usage:
    vision                         -> Launches the desktop GUI interface
    vision video.mp4               -> Directly processes video.mp4
    vision --webcam                -> Directly launches live webcam stream
    vision video.mp4 --target car  -> Filters specific objects directly
"""

import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)


def main():
    args = sys.argv[1:]

    # 1. No arguments: launch the native Desktop GUI
    if len(args) == 0:
        import app_gui
        app_gui.main()
        return

    # 2. Help request
    if "--help" in args or "-h" in args:
        print("""
Vision — Détection & Segmentation Temps Réel (YOLO11-seg)
---------------------------------------------------------
Usage:
  vision                           Ouvre l'interface graphique de bureau (GUI)
  vision <video_path>              Lance directement le traitement de la vidéo
  vision --webcam                  Lance directement la webcam en direct

Options avancées :
  --target <classes>               Filtre d'objets (ex: person ou car,bus)
  --conf <float>                   Seuil de confiance (ex: 0.35)
  --output <path>                  Enregistre la vidéo traitée (ex: output.mp4)
  --no-mirror                      Désactive le miroir en mode webcam
""")
        return

    # 3. Direct webcam request
    if "--webcam" in args or "-w" in args:
        from webcam_segmentation import main as webcam_main
        sys.argv = [sys.argv[0]] + [a for a in args if a not in ["--webcam", "-w"]]
        webcam_main()
        return

    # 4. Direct video file passed as first positional arg
    first_arg = args[0]
    if not first_arg.startswith("-") and (os.path.exists(first_arg) or any(first_arg.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"])):
        from video_segmentation import main as video_main
        new_argv = [sys.argv[0], "--source", first_arg] + args[1:]
        sys.argv = new_argv
        video_main()
        return

    # 5. Default fallback to video_segmentation CLI parser
    from video_segmentation import main as video_main
    video_main()


if __name__ == "__main__":
    main()
