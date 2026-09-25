"""
Vision Studio — Multi-Mode Real-Time Computer Vision Suite
-----------------------------------------------------------
Features 3 distinct vision paradigms switchable via sleek top tabs:
1. 🟦 Segmentation (YOLO11-seg) — 80 COCO classes with alpha-blended instance masks
2. 🌍 Détection Universelle (YOLO-World) — Open-vocabulary detection with natural language text prompts
3. 🦴 Squelette & Posture (YOLO-Pose) — Real-time tracking of 17 body joints, arms, legs, and posture

Apple Pro / Linear minimalist design, instant reactivity, and robust AVFoundation webcam support.
"""

import os
import sys
import time
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import torch
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)


def get_default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_color_palette(num_classes=80):
    np.random.seed(42)
    return np.random.randint(60, 240, size=(num_classes, 3), dtype=np.uint8)


# COCO 17 Keypoints definitions for Pose
LIMBS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Shoulders & arms
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # Hips & legs
    (0, 1), (0, 2), (1, 3), (2, 4)            # Head & face
]

LIMB_COLORS = [
    (255, 140, 0), (255, 0, 128), (0, 240, 255), (0, 140, 255), (140, 0, 255),
    (0, 255, 140), (140, 255, 0), (255, 230, 0),
    (0, 220, 255), (0, 120, 255), (255, 120, 0), (255, 60, 0),
    (180, 180, 180), (180, 180, 180), (140, 140, 140), (140, 140, 140)
]

KEYPOINT_NAMES = [
    "Nez", "Oeil G", "Oeil D", "Oreille G", "Oreille D",
    "Epaule G", "Epaule D", "Coude G", "Coude D", "Poignet G", "Poignet D",
    "Hanche G", "Hanche D", "Genou G", "Genou D", "Cheville G", "Cheville D"
]


class VisionStudio(ctk.CTk):
    def __init__(self, initial_source=None, is_webcam=False):
        super().__init__()

        # Window Setup
        self.title("Vision Studio")
        self.geometry("1380x880")
        self.minsize(1160, 740)
        ctk.set_appearance_mode("dark")
        self.configure(fg_color="#09090b")

        self.device = get_default_device()

        # Multi-model management (lazy-loaded / cached)
        self.models = {}
        self.current_mode = "segmentation"  # "segmentation", "world", "pose"

        # Initialize primary segmentation model
        self.models["segmentation"] = YOLO("yolo11n-seg.pt")
        self.color_palette = generate_color_palette(80)

        # Media State
        self.is_webcam = is_webcam
        self.source_path = initial_source or os.path.join(PROJECT_DIR, "samples", "pedestrians.avi")
        if not os.path.exists(self.source_path) and not self.is_webcam:
            self.source_path = os.path.join(PROJECT_DIR, "samples", "pedestrians.avi")

        self.cap = None
        self.total_frames = 0
        self.source_fps = 25.0
        self.current_frame_idx = 0
        self.is_playing = False  # Start paused on pedestrian video by default
        self.speed_factor = 1.0
        self.mirror_mode = True

        # Mode 1: Segmentation settings
        self.seg_display_mode = "both"  # "both", "masks", "boxes"
        self.seg_targets = None
        self.seg_presets = {
            "Tous les objets (80)": None,
            "Personnes uniquement": ["person"],
            "Véhicules": ["car", "bicycle", "motorcycle", "bus", "truck"],
            "Électronique": ["cell phone", "laptop", "tv"],
            "Objets du quotidien": ["bottle", "cup", "chair", "backpack", "handbag"]
        }

        # Mode 2: YOLO-World settings
        self.world_classes = [
            "person", "jacket", "shirt", "pants", "shoes", "glasses", "watch",
            "chair", "table", "laptop", "phone", "bottle", "cup", "backpack", "bag"
        ]
        self.world_initialized = False

        # Mode 3: Pose settings
        self.show_skeleton = True
        self.show_joints = True
        self.show_pose_box = False

        # Global parameters
        self.show_labels = True
        self.conf_threshold = 0.35

        # Frame buffers & locks
        self.lock = threading.Lock()
        self.current_raw_frame = None
        self.latest_display_frame = None
        self.detected_count = 0
        self.fps_samples = []
        self.current_fps = 0.0
        self.camera_error_msg = None

        # Build UI and open stream
        self.running = True
        self._build_ui()
        self._open_stream()
        self.reprocess_current_frame()

        # Background stream thread
        self.worker_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.worker_thread.start()

        # Main GUI refresh timer
        self._gui_refresh()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _get_model(self, mode):
        if mode not in self.models:
            if mode == "world":
                m = YOLO("yolov8s-worldv2.pt")
                try:
                    m.set_classes(self.world_classes)
                except Exception:
                    pass
                self.models["world"] = m
                self.world_initialized = True
            elif mode == "pose":
                self.models["pose"] = YOLO("yolo11n-pose.pt")
        return self.models.get(mode)

    # ==============================================================
    # UI CONSTRUCTION (Top Tabs & Two-Pane Architecture)
    # ==============================================================
    def _build_ui(self):
        # Master grid: Row 0 = Top Tab Bar, Row 1 = Main Body
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # --------------------------------------------------------------
        # TOP BAR: Mode Tabs (Sleek Apple Pro Toolbar)
        # --------------------------------------------------------------
        top_bar = ctk.CTkFrame(self, fg_color="#121215", corner_radius=0, height=54, border_width=1, border_color="#27272a")
        top_bar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        top_bar.grid_propagate(False)

        # Brand Title
        brand_lbl = ctk.CTkLabel(
            top_bar,
            text="👁 Vision Studio",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#f4f4f5"
        )
        brand_lbl.pack(side="left", padx=(20, 24))

        # Mode Tabs Selector
        self.mode_tabs = ctk.CTkSegmentedButton(
            top_bar,
            values=["🟦 Segmentation (YOLO11)", "🌍 Détection Universelle (YOLO-World)", "🦴 Articulations & Squelette (YOLO-Pose)"],
            command=self._on_tab_switch,
            fg_color="#18181b",
            selected_color="#27272a",
            selected_hover_color="#3f3f46",
            unselected_color="#18181b",
            text_color="#f4f4f5",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34
        )
        self.mode_tabs.set("🟦 Segmentation (YOLO11)")
        self.mode_tabs.pack(side="left", padx=10)

        # Device pill on right
        device_pill = ctk.CTkLabel(
            top_bar,
            text=f"● {self.device.upper()} Accelerated",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#22c55e",
            fg_color="#18181b",
            corner_radius=12,
            padx=12,
            pady=4
        )
        device_pill.pack(side="right", padx=20)

        # --------------------------------------------------------------
        # MAIN BODY: Left Video & Right Side Settings Panel
        # --------------------------------------------------------------
        body_frame = ctk.CTkFrame(self, fg_color="#09090b", corner_radius=0)
        body_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        body_frame.grid_columnconfigure(0, weight=1)
        body_frame.grid_columnconfigure(1, weight=0, minsize=370)
        body_frame.grid_rowconfigure(0, weight=1)

        # Left: Video canvas & transport
        left_frame = ctk.CTkFrame(body_frame, fg_color="#09090b", corner_radius=0)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        self.display_container = ctk.CTkFrame(
            left_frame,
            fg_color="#000000",
            corner_radius=8,
            border_width=1,
            border_color="#27272a"
        )
        self.display_container.grid(row=0, column=0, sticky="nsew", padx=16, pady=(12, 8))
        self.display_container.grid_rowconfigure(0, weight=1)
        self.display_container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.display_container, bg="#000000", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Bottom Media Toolbar
        bottom_toolbar = ctk.CTkFrame(
            left_frame,
            fg_color="#121215",
            corner_radius=8,
            border_width=1,
            border_color="#27272a"
        )
        bottom_toolbar.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))

        # Timeline Scrubber
        timeline_box = ctk.CTkFrame(bottom_toolbar, fg_color="transparent")
        timeline_box.pack(fill="x", padx=16, pady=(10, 4))

        self.time_lbl_left = ctk.CTkLabel(
            timeline_box,
            text="00:00",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f4f4f5",
            width=50
        )
        self.time_lbl_left.pack(side="left")

        self.timeline_slider = ctk.CTkSlider(
            timeline_box,
            from_=0,
            to=100,
            command=self._on_seek,
            fg_color="#27272a",
            progress_color="#e4e4e7",
            button_color="#ffffff",
            button_hover_color="#e4e4e7",
            height=14
        )
        self.timeline_slider.set(0)
        self.timeline_slider.pack(side="left", fill="x", expand=True, padx=12)

        self.time_lbl_right = ctk.CTkLabel(
            timeline_box,
            text="00:00",
            font=ctk.CTkFont(size=12),
            text_color="#71717a",
            width=50
        )
        self.time_lbl_right.pack(side="right")

        # Transport Controls
        ctrl_bar = ctk.CTkFrame(bottom_toolbar, fg_color="transparent")
        ctrl_bar.pack(fill="x", padx=16, pady=(4, 10))

        self.btn_play = ctk.CTkButton(
            ctrl_bar,
            text="▶ Lecture",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#27272a",
            hover_color="#3f3f46",
            text_color="#f4f4f5",
            width=90,
            height=32,
            corner_radius=6,
            command=self.toggle_play
        )
        self.btn_play.pack(side="left", padx=(0, 8))

        self.btn_rewind = ctk.CTkButton(
            ctrl_bar,
            text="↺ Recommencer",
            font=ctk.CTkFont(size=12),
            fg_color="#18181b",
            hover_color="#27272a",
            text_color="#d4d4d8",
            width=110,
            height=32,
            corner_radius=6,
            command=self.rewind_video
        )
        self.btn_rewind.pack(side="left", padx=4)

        ctk.CTkLabel(ctrl_bar, text="Vitesse", font=ctk.CTkFont(size=12), text_color="#a1a1aa").pack(side="left", padx=(16, 8))
        self.seg_speed = ctk.CTkSegmentedButton(
            ctrl_bar,
            values=["0.25x", "0.5x", "1x", "1.5x", "2x"],
            command=self._on_speed_select,
            fg_color="#18181b",
            selected_color="#27272a",
            selected_hover_color="#3f3f46",
            unselected_color="#18181b",
            text_color="#f4f4f5",
            height=28
        )
        self.seg_speed.set("1x")
        self.seg_speed.pack(side="left", padx=4)

        self.btn_snap = ctk.CTkButton(
            ctrl_bar,
            text="Capture HD",
            font=ctk.CTkFont(size=12),
            fg_color="#18181b",
            hover_color="#27272a",
            text_color="#d4d4d8",
            width=90,
            height=32,
            corner_radius=6,
            command=self.save_snapshot
        )
        self.btn_snap.pack(side="right")

        # --------------------------------------------------------------
        # RIGHT PANE: Side Settings Sidebar
        # --------------------------------------------------------------
        self.right_frame = ctk.CTkScrollableFrame(
            body_frame,
            fg_color="#121215",
            corner_radius=0,
            border_width=1,
            border_color="#27272a",
            width=370
        )
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        # Source Selection Card
        card_source = self._make_card(self.right_frame, "Source d'Entrée")

        self.seg_source = ctk.CTkSegmentedButton(
            card_source,
            values=["Fichier Vidéo", "Webcam Direct"],
            command=self._on_source_change,
            fg_color="#18181b",
            selected_color="#27272a",
            selected_hover_color="#3f3f46",
            unselected_color="#18181b",
            height=32
        )
        self.seg_source.set("Webcam Direct" if self.is_webcam else "Fichier Vidéo")
        self.seg_source.pack(fill="x", padx=14, pady=(8, 8))

        self.btn_browse = ctk.CTkButton(
            card_source,
            text="Parcourir une vidéo (mp4, avi...)",
            font=ctk.CTkFont(size=12),
            fg_color="#1c1d22",
            hover_color="#27272a",
            text_color="#d4d4d8",
            height=30,
            command=self.browse_video
        )
        self.btn_browse.pack(fill="x", padx=14, pady=(0, 12))

        self.btn_fix_camera = ctk.CTkButton(
            card_source,
            text="Autoriser la caméra dans Réglages",
            font=ctk.CTkFont(size=11),
            fg_color="#3f3f46",
            hover_color="#52525b",
            text_color="#fafafa",
            height=28,
            command=self._open_macos_camera_settings
        )

        # Container for Mode-Specific Controls (Dynamically switched on Tab change)
        self.mode_controls_container = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.mode_controls_container.pack(fill="x", padx=0, pady=0)

        self._render_mode_controls()

        # Shared: Confidence Card
        card_conf = self._make_card(self.right_frame, "Seuil de Confiance")
        self.lbl_conf = ctk.CTkLabel(card_conf, text="Confiance : 35%", font=ctk.CTkFont(size=12), text_color="#a1a1aa")
        self.lbl_conf.pack(anchor="w", padx=14, pady=(8, 4))

        self.slider_conf = ctk.CTkSlider(
            card_conf,
            from_=0.10,
            to=0.90,
            command=self._on_conf_drag,
            fg_color="#27272a",
            progress_color="#e4e4e7",
            button_color="#ffffff",
            button_hover_color="#e4e4e7",
            height=14
        )
        self.slider_conf.set(0.35)
        self.slider_conf.pack(fill="x", padx=14, pady=(0, 14))

        # Shared: Telemetry Card
        card_telemetry = self._make_card(self.right_frame, "Télémétrie Système")
        self.lbl_fps = ctk.CTkLabel(card_telemetry, text=f"FPS : 0.0 ({self.device.upper()})", font=ctk.CTkFont(size=12, weight="bold"), text_color="#f4f4f5")
        self.lbl_fps.pack(anchor="w", padx=14, pady=(8, 2))

        self.lbl_det_count = ctk.CTkLabel(card_telemetry, text="Détections : 0", font=ctk.CTkFont(size=12), text_color="#a1a1aa")
        self.lbl_det_count.pack(anchor="w", padx=14, pady=(0, 2))

        self.lbl_src_name = ctk.CTkLabel(card_telemetry, text="Source : -", font=ctk.CTkFont(size=11), text_color="#71717a")
        self.lbl_src_name.pack(anchor="w", padx=14, pady=(0, 12))

        # Quit Button
        self.btn_close = ctk.CTkButton(
            self.right_frame,
            text="Quitter l'application",
            font=ctk.CTkFont(size=12),
            fg_color="#1c1d22",
            hover_color="#27272a",
            text_color="#a1a1aa",
            height=34,
            command=self.on_close
        )
        self.btn_close.pack(fill="x", padx=16, pady=(16, 24))

    def _make_card(self, parent, title):
        card = ctk.CTkFrame(
            parent,
            fg_color="#18181b",
            corner_radius=8,
            border_width=1,
            border_color="#27272a"
        )
        card.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#e4e4e7"
        ).pack(anchor="w", padx=14, pady=(10, 2))
        return card

    # ==============================================================
    # DYNAMIC MODE SWITCHER & SETTINGS CARDS
    # ==============================================================
    def _on_tab_switch(self, value):
        if "Segmentation" in value:
            self.current_mode = "segmentation"
        elif "Universelle" in value:
            self.current_mode = "world"
        elif "Squelette" in value:
            self.current_mode = "pose"

        # Lazy load model in background if needed
        self._get_model(self.current_mode)
        self._render_mode_controls()
        self.reprocess_current_frame()

    def _render_mode_controls(self):
        # Clear existing mode controls
        for widget in self.mode_controls_container.winfo_children():
            widget.destroy()

        if self.current_mode == "segmentation":
            # --- Segmentation Options ---
            card_seg = self._make_card(self.mode_controls_container, "Options de Segmentation")

            self.seg_display = ctk.CTkSegmentedButton(
                card_seg,
                values=["Masques + Boîtes", "Masques seuls", "Boîtes seules"],
                command=self._on_seg_display_change,
                fg_color="#18181b",
                selected_color="#27272a",
                selected_hover_color="#3f3f46",
                unselected_color="#18181b",
                height=30
            )
            mode_map = {"both": "Masques + Boîtes", "masks": "Masques seuls", "boxes": "Boîtes seules"}
            self.seg_display.set(mode_map.get(self.seg_display_mode, "Masques + Boîtes"))
            self.seg_display.pack(fill="x", padx=14, pady=(8, 10))

            card_targets = self._make_card(self.mode_controls_container, "Filtre d'Objets COCO (80)")
            self.opt_preset = ctk.CTkOptionMenu(
                card_targets,
                values=list(self.seg_presets.keys()),
                command=self._on_seg_preset_selected,
                fg_color="#18181b",
                button_color="#27272a",
                button_hover_color="#3f3f46",
                text_color="#f4f4f5",
                height=32
            )
            self.opt_preset.set("Tous les objets (80)")
            self.opt_preset.pack(fill="x", padx=14, pady=(8, 12))

        elif self.current_mode == "world":
            # --- YOLO-World (Open-Vocabulary) Options ---
            card_world = self._make_card(self.mode_controls_container, "Vocabulaire Ouvert (YOLO-World)")

            ctk.CTkLabel(
                card_world,
                text="Objets détectés en direct (langage naturel) :",
                font=ctk.CTkFont(size=11),
                text_color="#a1a1aa"
            ).pack(anchor="w", padx=14, pady=(6, 4))

            self.txt_world_classes = ctk.CTkTextbox(
                card_world,
                height=80,
                fg_color="#18181b",
                border_width=1,
                border_color="#27272a",
                font=ctk.CTkFont(size=11)
            )
            self.txt_world_classes.pack(fill="x", padx=14, pady=(0, 8))
            self.txt_world_classes.insert("1.0", ", ".join(self.world_classes))

            btn_update_world = ctk.CTkButton(
                card_world,
                text="Mettre à jour les classes",
                font=ctk.CTkFont(size=11),
                fg_color="#27272a",
                hover_color="#3f3f46",
                height=30,
                command=self._apply_world_classes
            )
            btn_update_world.pack(fill="x", padx=14, pady=(0, 10))

            # Preset buttons
            presets_row = ctk.CTkFrame(card_world, fg_color="transparent")
            presets_row.pack(fill="x", padx=14, pady=(0, 12))

            ctk.CTkButton(
                presets_row,
                text="Vêtements",
                font=ctk.CTkFont(size=10),
                width=65,
                height=26,
                fg_color="#1c1d22",
                hover_color="#27272a",
                command=lambda: self._set_world_preset(["person", "jacket", "shirt", "shoes", "glasses", "watch", "backpack"])
            ).pack(side="left", padx=(0, 4))

            ctk.CTkButton(
                presets_row,
                text="Bureau",
                font=ctk.CTkFont(size=10),
                width=65,
                height=26,
                fg_color="#1c1d22",
                hover_color="#27272a",
                command=lambda: self._set_world_preset(["laptop", "phone", "keyboard", "mouse", "cup", "bottle", "chair", "pen"])
            ).pack(side="left", padx=4)

            ctk.CTkButton(
                presets_row,
                text="Général",
                font=ctk.CTkFont(size=10),
                width=65,
                height=26,
                fg_color="#1c1d22",
                hover_color="#27272a",
                command=lambda: self._set_world_preset([
                    "person", "jacket", "shirt", "pants", "shoes", "glasses", "watch",
                    "chair", "table", "laptop", "phone", "bottle", "cup", "backpack", "bag"
                ])
            ).pack(side="left", padx=4)

        elif self.current_mode == "pose":
            # --- Pose / Skeleton Options ---
            card_pose = self._make_card(self.mode_controls_container, "Squelette & Articulations")

            self.sw_skel = ctk.CTkSwitch(
                card_pose,
                text="Tracer les os / membres (lignes)",
                font=ctk.CTkFont(size=12),
                command=self._on_pose_toggles,
                progress_color="#e4e4e7"
            )
            if self.show_skeleton:
                self.sw_skel.select()
            self.sw_skel.pack(anchor="w", padx=14, pady=(8, 4))

            self.sw_joints = ctk.CTkSwitch(
                card_pose,
                text="Afficher les 17 articulations (points)",
                font=ctk.CTkFont(size=12),
                command=self._on_pose_toggles,
                progress_color="#e4e4e7"
            )
            if self.show_joints:
                self.sw_joints.select()
            self.sw_joints.pack(anchor="w", padx=14, pady=4)

            self.sw_pose_box = ctk.CTkSwitch(
                card_pose,
                text="Boîte englobante de la personne",
                font=ctk.CTkFont(size=12),
                command=self._on_pose_toggles,
                progress_color="#e4e4e7"
            )
            if self.show_pose_box:
                self.sw_pose_box.select()
            self.sw_pose_box.pack(anchor="w", padx=14, pady=(4, 12))

    # ==============================================================
    # MODE SPECIFIC CALLBACKS
    # ==============================================================
    def _on_seg_display_change(self, val):
        mapping = {"Masques + Boîtes": "both", "Masques seuls": "masks", "Boîtes seules": "boxes"}
        self.seg_display_mode = mapping.get(val, "both")
        self.reprocess_current_frame()

    def _on_seg_preset_selected(self, choice):
        self.seg_targets = self.seg_presets.get(choice, None)
        self.reprocess_current_frame()

    def _set_world_preset(self, classes):
        self.world_classes = classes
        self.txt_world_classes.delete("1.0", "end")
        self.txt_world_classes.insert("1.0", ", ".join(classes))
        self._apply_world_classes()

    def _apply_world_classes(self):
        text = self.txt_world_classes.get("1.0", "end").strip()
        classes = [c.strip().lower() for c in text.split(",") if c.strip()]
        if classes:
            self.world_classes = classes
            model = self._get_model("world")
            try:
                model.set_classes(self.world_classes)
            except Exception as e:
                print(f"[WARN] Error updating YOLO-World classes: {e}")
            self.reprocess_current_frame()

    def _on_pose_toggles(self):
        self.show_skeleton = self.sw_skel.get()
        self.show_joints = self.sw_joints.get()
        self.show_pose_box = self.sw_pose_box.get()
        self.reprocess_current_frame()

    # ==============================================================
    # STREAM ACQUISITION (AVFoundation on macOS)
    # ==============================================================
    def _open_stream(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.camera_error_msg = None

            if self.is_webcam:
                backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(0, backend)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                opened = self.cap.isOpened()
                if opened:
                    ret, test_frame = self.cap.read()
                    if not ret or test_frame is None:
                        opened = False

                if not opened:
                    self.camera_error_msg = "Accès Webcam refusé sur macOS.\nAutorisez Terminal dans Réglages Système > Confidentialité > Caméra."
                    self.btn_fix_camera.pack(fill="x", padx=14, pady=(0, 10))
                else:
                    self.btn_fix_camera.pack_forget()

                self.total_frames = 0
                self.source_fps = 30.0
                self.timeline_slider.configure(state="disabled")
                self.btn_rewind.configure(state="disabled")
                self.btn_browse.configure(state="disabled")
                self.time_lbl_left.configure(text="LIVE")
                self.time_lbl_right.configure(text="LIVE")
                self.lbl_src_name.configure(text="Source : Webcam HD (macOS)")
            else:
                self.btn_fix_camera.pack_forget()
                self.btn_browse.configure(state="normal")
                self.timeline_slider.configure(state="normal")
                self.btn_rewind.configure(state="normal")

                self.cap = cv2.VideoCapture(self.source_path)
                if not self.cap.isOpened():
                    self.lbl_src_name.configure(text="✕ Fichier introuvable")
                    return

                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
                self.timeline_slider.configure(to=self.total_frames - 1)

                dur_sec = int(self.total_frames / self.source_fps)
                self.time_lbl_right.configure(text=f"{dur_sec // 60:02d}:{dur_sec % 60:02d}")
                self.lbl_src_name.configure(text=f"Source : {os.path.basename(self.source_path)}")

                # Preload frame 0 so it displays on startup
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.current_raw_frame = frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.current_frame_idx = 0

    def _open_macos_camera_settings(self):
        try:
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"], check=False)
        except Exception:
            pass

    def _on_source_change(self, value):
        self.is_webcam = ("Webcam" in value)
        if self.is_webcam:
            self.is_playing = True
            self.btn_play.configure(text="⏸ Pause")
        else:
            self.is_playing = False
            self.btn_play.configure(text="▶ Lecture")
        self._open_stream()
        self.reprocess_current_frame()

    def browse_video(self):
        chosen = filedialog.askopenfilename(
            title="Choisir un fichier vidéo",
            filetypes=[("Vidéos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Tous", "*.*")],
            initialdir=os.path.join(PROJECT_DIR, "samples")
        )
        if chosen:
            self.source_path = chosen
            self.is_webcam = False
            self.seg_source.set("Fichier Vidéo")
            self._open_stream()
            self.reprocess_current_frame()

    # ==============================================================
    # MEDIA CONTROLS & REACTIVITY
    # ==============================================================
    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.configure(text="▶ Lecture" if not self.is_playing else "⏸ Pause")
        if not self.is_playing:
            self.reprocess_current_frame()

    def rewind_video(self):
        if self.is_webcam or not self.cap:
            return
        with self.lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.timeline_slider.set(0)
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_raw_frame = frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.reprocess_current_frame()

    def _on_seek(self, value):
        if self.is_webcam or not self.cap:
            return
        target = int(value)
        with self.lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.current_frame_idx = target
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_raw_frame = frame
        self.reprocess_current_frame()

    def _on_speed_select(self, speed_str):
        mapping = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "1.5x": 1.5, "2x": 2.0}
        self.speed_factor = mapping.get(speed_str, 1.0)

    def _on_conf_drag(self, val):
        self.conf_threshold = float(val)
        self.lbl_conf.configure(text=f"Confiance : {int(self.conf_threshold * 100)}%")
        self.reprocess_current_frame()

    def save_snapshot(self):
        with self.lock:
            frame = self.latest_display_frame
        if frame is not None:
            filename = f"capture_{int(time.time())}.jpg"
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr)
            messagebox.showinfo("Capture Sauvegardée", f"Image enregistrée sous :\n{filename}")

    # ==============================================================
    # INFERENCE PIPELINES (Segmentation, Open-Vocabulary, Pose)
    # ==============================================================
    def reprocess_current_frame(self):
        with self.lock:
            frame = self.current_raw_frame

        if frame is None:
            return

        annotated, count = self._infer_and_annotate(frame)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with self.lock:
            self.latest_display_frame = rgb
            self.detected_count = count

    def _infer_and_annotate(self, frame):
        h, w, _ = frame.shape
        display_frame = frame.copy()

        if self.is_webcam and self.mirror_mode:
            display_frame = cv2.flip(display_frame, 1)

        model = self._get_model(self.current_mode)
        if model is None:
            return display_frame, 0

        # --- 1. MODE SEGMENTATION ---
        if self.current_mode == "segmentation":
            target_ids = None
            if self.seg_targets is not None:
                all_names = model.names
                name_to_id = {v.lower(): k for k, v in all_names.items()}
                target_ids = [name_to_id[t] for t in self.seg_targets if t in name_to_id]
                if not target_ids and len(self.seg_targets) > 0:
                    target_ids = [-1]

            try:
                results = model.predict(display_frame, device=self.device, classes=target_ids, conf=self.conf_threshold, verbose=False)
                r = results[0]
            except Exception:
                return display_frame, 0

            overlay = display_frame.copy()
            if self.seg_display_mode in ["both", "masks"] and r.masks is not None and len(r.masks) > 0:
                for mask_coords, box in zip(r.masks.xy, r.boxes):
                    cls_id = int(box.cls[0].item())
                    color = [int(c) for c in self.color_palette[cls_id % len(self.color_palette)]]
                    if len(mask_coords) > 0:
                        poly = np.array(mask_coords, dtype=np.int32)
                        cv2.fillPoly(overlay, [poly], color)
                        cv2.polylines(display_frame, [poly], isClosed=True, color=color, thickness=2)

                alpha = 0.40
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

            count = len(r.boxes) if r.boxes is not None else 0
            if self.seg_display_mode in ["both", "boxes"] and r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = [int(c) for c in self.color_palette[cls_id % len(self.color_palette)]]

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                    by1 = max(0, y1 - th - base - 4)
                    bx2 = min(w, x1 + tw + 6)
                    cv2.rectangle(display_frame, (x1, by1), (bx2, y1), color, -1)
                    cv2.putText(display_frame, label, (x1 + 3, y1 - base - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            return display_frame, count

        # --- 2. MODE YOLO-WORLD (Vocabulaire Ouvert) ---
        elif self.current_mode == "world":
            try:
                results = model.predict(display_frame, device=self.device, conf=self.conf_threshold, verbose=False)
                r = results[0]
            except Exception:
                return display_frame, 0

            count = len(r.boxes) if r.boxes is not None else 0
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = [int(c) for c in self.color_palette[cls_id % len(self.color_palette)]]

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    name = model.names.get(cls_id, f"obj_{cls_id}")
                    label = f"{name} {conf:.2f}"
                    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                    by1 = max(0, y1 - th - base - 4)
                    bx2 = min(w, x1 + tw + 6)
                    cv2.rectangle(display_frame, (x1, by1), (bx2, y1), color, -1)
                    cv2.putText(display_frame, label, (x1 + 3, y1 - base - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            return display_frame, count

        # --- 3. MODE POSE / SQUELETTE & ARTICULATIONS ---
        elif self.current_mode == "pose":
            try:
                results = model.predict(display_frame, device=self.device, conf=self.conf_threshold, verbose=False)
                r = results[0]
            except Exception:
                return display_frame, 0

            persons_count = len(r.keypoints) if r.keypoints is not None else 0

            if r.keypoints is not None and len(r.keypoints) > 0:
                for person_idx, kpts in enumerate(r.keypoints.xy):
                    kpts_np = kpts.cpu().numpy()  # (17, 2)
                    confs = r.keypoints.conf[person_idx].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)

                    # Optional Bounding Box
                    if self.show_pose_box and r.boxes is not None and person_idx < len(r.boxes):
                        box = r.boxes[person_idx]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                    # Draw Bones (Limbs)
                    if self.show_skeleton:
                        for limb_idx, (p1, p2) in enumerate(LIMBS):
                            if confs[p1] > 0.35 and confs[p2] > 0.35:
                                pt1 = (int(kpts_np[p1, 0]), int(kpts_np[p1, 1]))
                                pt2 = (int(kpts_np[p2, 0]), int(kpts_np[p2, 1]))
                                color = LIMB_COLORS[limb_idx % len(LIMB_COLORS)]
                                cv2.line(display_frame, pt1, pt2, color, 3, cv2.LINE_AA)

                    # Draw Joints (Keypoints)
                    if self.show_joints:
                        for joint_id in range(17):
                            if confs[joint_id] > 0.35:
                                jx, jy = int(kpts_np[joint_id, 0]), int(kpts_np[joint_id, 1])
                                # Glowing double-circle for joints
                                cv2.circle(display_frame, (jx, jy), 6, (0, 0, 0), -1)
                                cv2.circle(display_frame, (jx, jy), 4, (0, 255, 127), -1)

            return display_frame, persons_count

        return display_frame, 0

    # ==============================================================
    # CAPTURE LOOP & GUI REFRESH
    # ==============================================================
    def _stream_loop(self):
        while self.running:
            if not self.is_playing:
                time.sleep(0.04)
                continue

            t_start = time.perf_counter()

            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.05)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    if not self.is_webcam and self.total_frames > 0:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_idx = 0
                        ret, frame = self.cap.read()
                    if not ret or frame is None:
                        time.sleep(0.05)
                        continue

                self.current_raw_frame = frame
                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            annotated, count = self._infer_and_annotate(frame)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with self.lock:
                self.latest_display_frame = rgb
                self.detected_count = count

            cost = time.perf_counter() - t_start
            fps = 1.0 / cost if cost > 0 else 0
            self.fps_samples.append(fps)
            if len(self.fps_samples) > 20:
                self.fps_samples.pop(0)
            self.current_fps = sum(self.fps_samples) / len(self.fps_samples)

            target_delay = (1.0 / self.source_fps) / max(0.1, self.speed_factor)
            remaining = target_delay - cost
            if remaining > 0:
                time.sleep(remaining)

    def _gui_refresh(self):
        if not self.running:
            return

        with self.lock:
            frame = self.latest_display_frame
            cur_idx = self.current_frame_idx
            det_count = self.detected_count
            fps_val = self.current_fps
            err = self.camera_error_msg

        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())

        if err is not None:
            self.canvas.delete("all")
            self.canvas.create_text(
                canvas_w // 2, canvas_h // 2 - 20,
                text="📷 Accès Caméra Verrouillé sur macOS",
                fill="#f4f4f5", font=("Arial", 16, "bold"), justify="center"
            )
            self.canvas.create_text(
                canvas_w // 2, canvas_h // 2 + 20,
                text="Autorisez Terminal dans Réglages Système > Confidentialité & Sécurité > Caméra\npuis rechargez la source.",
                fill="#a1a1aa", font=("Arial", 12), justify="center"
            )
        elif frame is not None:
            fh, fw, _ = frame.shape
            scale = min(canvas_w / fw, canvas_h / fh)
            nw = int(fw * scale)
            nh = int(fh * scale)

            if nw > 0 and nh > 0:
                resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                img = Image.fromarray(resized)
                self.tk_photo = ImageTk.PhotoImage(image=img)
                self.canvas.delete("all")
                self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_photo, anchor="center")

        if not self.is_webcam and self.total_frames > 0:
            self.timeline_slider.set(cur_idx)
            sec = int(cur_idx / self.source_fps)
            self.time_lbl_left.configure(text=f"{sec // 60:02d}:{sec % 60:02d}")

        self.lbl_fps.configure(text=f"FPS : {fps_val:.1f} ({self.device.upper()})")
        if self.current_mode == "pose":
            self.lbl_det_count.configure(text=f"Personnes / Postures : {det_count}")
        else:
            self.lbl_det_count.configure(text=f"Détections : {det_count}")

        self.after(25, self._gui_refresh)

    def on_close(self):
        self.running = False
        with self.lock:
            if self.cap is not None:
                self.cap.release()
        self.destroy()


def main():
    initial_source = None
    is_webcam = False

    args = sys.argv[1:]
    if "--webcam" in args or "-w" in args:
        is_webcam = True
    elif len(args) > 0 and not args[0].startswith("-"):
        initial_source = args[0]

    app = VisionStudio(initial_source=initial_source, is_webcam=is_webcam)
    app.mainloop()


if __name__ == "__main__":
    main()
