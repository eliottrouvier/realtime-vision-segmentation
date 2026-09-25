"""
Real-Time Vision Segmentation — Modern Desktop GUI
---------------------------------------------------
A clean macOS-native dark-themed desktop interface for launching
video and webcam instance segmentation with interactive controls.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Ensure working directory is the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# Import backend runners
from video_segmentation import process_video
from webcam_segmentation import run_webcam


class VisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Vision — Détection & Segmentation Temps Réel")
        self.geometry("640x740")
        self.minsize(600, 680)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Variables
        self.conf_val = tk.DoubleVar(value=0.35)
        self.all_classes_var = tk.BooleanVar(value=True)
        self.person_var = tk.BooleanVar(value=False)
        self.vehicles_var = tk.BooleanVar(value=False)
        self.electronics_var = tk.BooleanVar(value=False)
        self.everyday_var = tk.BooleanVar(value=False)
        self.mirror_var = tk.BooleanVar(value=True)
        self.save_output_var = tk.BooleanVar(value=False)
        self.custom_filter_var = tk.StringVar(value="")

        self.is_running = False

        self._build_ui()

    def _build_ui(self):
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=25, pady=(20, 10))

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="👁️ Vision Studio",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ctk.CTkLabel(
            header_frame,
            text="Segmentation d'instances & détection en temps réel (YOLO11-seg • MPS)",
            font=ctk.CTkFont(size=13),
            text_color="#9da5b4"
        )
        sub_lbl.pack(anchor="w")

        # Card 1: Action Buttons (Large Primary Triggers)
        action_card = ctk.CTkFrame(self, corner_radius=12)
        action_card.pack(fill="x", padx=25, pady=10)

        ctk.CTkLabel(
            action_card,
            text="Source d'Entrée",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 8))

        btn_row = ctk.CTkFrame(action_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 14))

        self.btn_webcam = ctk.CTkButton(
            btn_row,
            text="🎥 Lancer la Webcam",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            fg_color="#10b981",
            hover_color="#059669",
            command=self.start_webcam
        )
        self.btn_webcam.pack(side="left", expand=True, fill="x", padx=(0, 8))

        self.btn_video = ctk.CTkButton(
            btn_row,
            text="📁 Ouvrir une Vidéo...",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            fg_color="#3b82f6",
            hover_color="#2563eb",
            command=self.choose_and_run_video
        )
        self.btn_video.pack(side="right", expand=True, fill="x", padx=(8, 0))

        # Card 2: Target Objects (Filters)
        filter_card = ctk.CTkFrame(self, corner_radius=12)
        filter_card.pack(fill="x", padx=25, pady=10)

        ctk.CTkLabel(
            filter_card,
            text="Objets Cibles à Détecter",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 6))

        # Toggle all
        self.chk_all = ctk.CTkSwitch(
            filter_card,
            text="Détecter toutes les classes COCO (80 objets)",
            variable=self.all_classes_var,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_toggle_all
        )
        self.chk_all.pack(anchor="w", padx=16, pady=(2, 8))

        # Categories
        cat_grid = ctk.CTkFrame(filter_card, fg_color="transparent")
        cat_grid.pack(fill="x", padx=16, pady=(0, 6))

        self.chk_person = ctk.CTkCheckBox(cat_grid, text="🚶 Personnes (person)", variable=self.person_var, command=self._on_cat_selected)
        self.chk_person.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)

        self.chk_vehicles = ctk.CTkCheckBox(cat_grid, text="🚗 Véhicules (car, bike, bus...)", variable=self.vehicles_var, command=self._on_cat_selected)
        self.chk_vehicles.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=4)

        self.chk_electronics = ctk.CTkCheckBox(cat_grid, text="📱 Électronique (phone, laptop)", variable=self.electronics_var, command=self._on_cat_selected)
        self.chk_electronics.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)

        self.chk_everyday = ctk.CTkCheckBox(cat_grid, text="☕ Objets (cup, bottle, chair)", variable=self.everyday_var, command=self._on_cat_selected)
        self.chk_everyday.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=4)

        # Custom entry
        custom_row = ctk.CTkFrame(filter_card, fg_color="transparent")
        custom_row.pack(fill="x", padx=16, pady=(6, 12))

        ctk.CTkLabel(custom_row, text="Filtre personnalisé :", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 8))
        self.entry_custom = ctk.CTkEntry(
            custom_row,
            placeholder_text="ex: dog, cat, backpack, suitcase",
            textvariable=self.custom_filter_var,
            height=32
        )
        self.entry_custom.pack(side="left", fill="x", expand=True)

        # Card 3: Parameters & Tuning
        param_card = ctk.CTkFrame(self, corner_radius=12)
        param_card.pack(fill="x", padx=25, pady=10)

        ctk.CTkLabel(
            param_card,
            text="Paramètres & Réglages",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 6))

        # Confidence slider
        slider_row = ctk.CTkFrame(param_card, fg_color="transparent")
        slider_row.pack(fill="x", padx=16, pady=(0, 8))

        self.lbl_conf = ctk.CTkLabel(slider_row, text=f"Seuil de Confiance : {int(self.conf_val.get() * 100)}%")
        self.lbl_conf.pack(side="left")

        slider = ctk.CTkSlider(
            slider_row,
            from_=0.10,
            to=0.90,
            variable=self.conf_val,
            command=self._on_slider_change
        )
        slider.pack(side="right", fill="x", expand=True, padx=(16, 0))

        # Options toggles
        opt_row = ctk.CTkFrame(param_card, fg_color="transparent")
        opt_row.pack(fill="x", padx=16, pady=(4, 14))

        sw_mirror = ctk.CTkSwitch(opt_row, text="Mode miroir (Webcam)", variable=self.mirror_var)
        sw_mirror.pack(side="left", padx=(0, 16))

        sw_save = ctk.CTkSwitch(opt_row, text="Enregistrer la vidéo de sortie", variable=self.save_output_var)
        sw_save.pack(side="left")

        # Footer Status Bar
        self.status_lbl = ctk.CTkLabel(
            self,
            text="Prêt • Cliquez sur un bouton pour démarrer",
            font=ctk.CTkFont(size=12),
            text_color="#8b949e"
        )
        self.status_lbl.pack(side="bottom", pady=12)

    def _on_slider_change(self, value):
        self.lbl_conf.configure(text=f"Seuil de Confiance : {int(value * 100)}%")

    def _on_toggle_all(self):
        if self.all_classes_var.get():
            self.person_var.set(False)
            self.vehicles_var.set(False)
            self.electronics_var.set(False)
            self.everyday_var.set(False)

    def _on_cat_selected(self):
        # If any specific category is selected, uncheck "All"
        if any([self.person_var.get(), self.vehicles_var.get(), self.electronics_var.get(), self.everyday_var.get()]):
            self.all_classes_var.set(False)
        else:
            self.all_classes_var.set(True)

    def _get_target_list(self):
        if self.all_classes_var.get():
            return None

        targets = []
        if self.person_var.get():
            targets.append("person")
        if self.vehicles_var.get():
            targets.extend(["car", "bicycle", "motorcycle", "bus", "truck"])
        if self.electronics_var.get():
            targets.extend(["cell phone", "laptop", "tv"])
        if self.everyday_var.get():
            targets.extend(["bottle", "cup", "chair", "backpack", "handbag"])

        custom = self.custom_filter_var.get().strip()
        if custom:
            for item in custom.split(","):
                if item.strip():
                    targets.append(item.strip().lower())

        return targets if targets else None

    def _set_ui_state(self, running=False):
        self.is_running = running
        state = "disabled" if running else "normal"
        self.btn_webcam.configure(state=state)
        self.btn_video.configure(state=state)
        if running:
            self.status_lbl.configure(
                text="⚡ En cours d'exécution • Contrôles : 'q' pour quitter, 'p' pour pause, 's' capture",
                text_color="#10b981"
            )
        else:
            self.status_lbl.configure(
                text="Prêt • Session terminée",
                text_color="#8b949e"
            )

    def start_webcam(self):
        if self.is_running:
            return

        targets = self._get_target_list()
        conf = float(self.conf_val.get())
        mirror = self.mirror_var.get()

        self._set_ui_state(True)

        def worker():
            try:
                run_webcam(
                    cam_index=0,
                    model_name="yolo11n-seg.pt",
                    target_classes=targets,
                    conf_threshold=conf,
                    mirror=mirror
                )
            except Exception as e:
                print(f"[ERROR] {e}")
            finally:
                self.after(0, lambda: self._set_ui_state(False))

        threading.Thread(target=worker, daemon=True).start()

    def choose_and_run_video(self):
        if self.is_running:
            return

        video_path = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=[
                ("Vidéos compatibles", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("Tous les fichiers", "*.*")
            ],
            initialdir=os.path.join(PROJECT_DIR, "samples")
        )
        if not video_path:
            return

        targets = self._get_target_list()
        conf = float(self.conf_val.get())
        output_path = None
        if self.save_output_var.get():
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_segmented.mp4"

        self._set_ui_state(True)

        def worker():
            try:
                process_video(
                    source_path=video_path,
                    model_name="yolo11n-seg.pt",
                    target_classes=targets,
                    conf_threshold=conf,
                    output_path=output_path,
                    display=True
                )
            except Exception as e:
                print(f"[ERROR] {e}")
            finally:
                self.after(0, lambda: self._set_ui_state(False))

        threading.Thread(target=worker, daemon=True).start()


def main():
    app = VisionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
