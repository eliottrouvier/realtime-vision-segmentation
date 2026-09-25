"""
Vision Studio — Minimalist Real-Time Detection & Segmentation
--------------------------------------------------------------
Ultra-responsive, sleek Apple Pro / Linear styled interface.
Features:
- Robust macOS camera capture (AVFoundation) with permission diagnostics
- Instant reactivity: every setting (filters, mode, confidence) re-renders live even when paused
- Clean monochromatic dark palette (Deep Zinc #09090b & slate accents)
- Timeline scrubber with direct frame seek
- Real-time speed tuning (0.25x to 2x) and screenshot capture
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


class VisionStudio(ctk.CTk):
    def __init__(self, initial_source=None, is_webcam=False):
        super().__init__()

        # Window Setup (Sleek Apple Pro Theme)
        self.title("Vision Studio")
        self.geometry("1340x840")
        self.minsize(1120, 720)
        ctk.set_appearance_mode("dark")

        # Color Theme: Deep Zinc / Linear Style
        self.configure(fg_color="#09090b")

        # Runtime Engine
        self.device = get_default_device()
        self.model_name = "yolo11n-seg.pt"
        self.model = YOLO(self.model_name)
        self.color_palette = generate_color_palette(len(self.model.names))

        # Media State
        self.is_webcam = is_webcam
        self.source_path = initial_source or os.path.join(PROJECT_DIR, "samples", "pedestrians.avi")
        if not os.path.exists(self.source_path) and not self.is_webcam:
            self.source_path = os.path.join(PROJECT_DIR, "samples", "pedestrians.avi")

        self.cap = None
        self.total_frames = 0
        self.source_fps = 25.0
        self.current_frame_idx = 0
        self.is_playing = True
        self.speed_factor = 1.0
        self.mirror_mode = True

        # Pipeline Options
        self.display_mode = "both"  # "both", "masks", "boxes"
        self.show_labels = True
        self.conf_threshold = 0.35
        self.active_targets = None  # None = All 80 COCO classes

        self.target_presets = {
            "Tous les objets (80)": None,
            "Personnes uniquement": ["person"],
            "Véhicules": ["car", "bicycle", "motorcycle", "bus", "truck"],
            "Électronique": ["cell phone", "laptop", "tv"],
            "Objets du quotidien": ["bottle", "cup", "chair", "backpack", "handbag"]
        }

        # Frame buffers
        self.lock = threading.Lock()
        self.current_raw_frame = None
        self.latest_display_frame = None
        self.current_detections = []
        self.fps_samples = []
        self.current_fps = 0.0
        self.camera_error_msg = None

        # Threading control
        self.running = True
        self._build_ui()
        self._open_stream()

        # Background stream thread
        self.worker_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.worker_thread.start()

        # Main GUI refresh timer
        self._gui_refresh()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ==============================================================
    # UI CONSTRUCTION (Minimalist Apple Pro / Linear Design)
    # ==============================================================
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=370)
        self.grid_rowconfigure(0, weight=1)

        # --------------------------------------------------------------
        # LEFT PANE: Video Canvas & Transport Bar
        # --------------------------------------------------------------
        left_frame = ctk.CTkFrame(self, fg_color="#09090b", corner_radius=0)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # Video Display Container
        self.display_container = ctk.CTkFrame(
            left_frame,
            fg_color="#000000",
            corner_radius=8,
            border_width=1,
            border_color="#27272a"
        )
        self.display_container.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))
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

        # Playback Controls Bar
        ctrl_bar = ctk.CTkFrame(bottom_toolbar, fg_color="transparent")
        ctrl_bar.pack(fill="x", padx=16, pady=(4, 10))

        self.btn_play = ctk.CTkButton(
            ctrl_bar,
            text="⏸ Pause",
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

        # Speed Segment
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

        # Snapshot Button
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
        # RIGHT PANE: Modern Minimalist Settings Sidebar
        # --------------------------------------------------------------
        right_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="#121215",
            corner_radius=0,
            border_width=1,
            border_color="#27272a",
            width=370
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        # Header Title
        ctk.CTkLabel(
            right_frame,
            text="Vision Studio",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#f4f4f5"
        ).pack(anchor="w", padx=16, pady=(16, 2))

        self.lbl_stream_status = ctk.CTkLabel(
            right_frame,
            text="● Flux actif",
            font=ctk.CTkFont(size=12),
            text_color="#22c55e"
        )
        self.lbl_stream_status.pack(anchor="w", padx=16, pady=(0, 16))

        # --- Section 1: Source Switcher ---
        card_source = self._make_card(right_frame, "Source d'Entrée")

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

        # Camera Permission Fix Button (Hidden by default, shown if needed)
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

        # --- Section 2: Display & Overlay Mode ---
        card_mode = self._make_card(right_frame, "Rendu & Segmentation")

        self.seg_display = ctk.CTkSegmentedButton(
            card_mode,
            values=["Masques + Boîtes", "Masques seuls", "Boîtes seules"],
            command=self._on_display_mode_change,
            fg_color="#18181b",
            selected_color="#27272a",
            selected_hover_color="#3f3f46",
            unselected_color="#18181b",
            height=30
        )
        self.seg_display.set("Masques + Boîtes")
        self.seg_display.pack(fill="x", padx=14, pady=(8, 10))

        self.sw_labels = ctk.CTkSwitch(
            card_mode,
            text="Afficher labels & indices de confiance",
            font=ctk.CTkFont(size=12),
            command=self._on_param_modified,
            progress_color="#e4e4e7"
        )
        self.sw_labels.select()
        self.sw_labels.pack(anchor="w", padx=14, pady=4)

        self.sw_mirror = ctk.CTkSwitch(
            card_mode,
            text="Mode miroir horizontal",
            font=ctk.CTkFont(size=12),
            command=self._on_param_modified,
            progress_color="#e4e4e7"
        )
        self.sw_mirror.select()
        self.sw_mirror.pack(anchor="w", padx=14, pady=(4, 12))

        # --- Section 3: Live Target Filtering ---
        card_targets = self._make_card(right_frame, "Filtre d'Objets en Direct")

        self.opt_preset = ctk.CTkOptionMenu(
            card_targets,
            values=list(self.target_presets.keys()),
            command=self._on_preset_selected,
            fg_color="#18181b",
            button_color="#27272a",
            button_hover_color="#3f3f46",
            text_color="#f4f4f5",
            height=32
        )
        self.opt_preset.set("Tous les objets (80)")
        self.opt_preset.pack(fill="x", padx=14, pady=(8, 8))

        ctk.CTkLabel(card_targets, text="Filtre personnalisé :", font=ctk.CTkFont(size=11), text_color="#71717a").pack(anchor="w", padx=14, pady=(2, 2))
        
        entry_row = ctk.CTkFrame(card_targets, fg_color="transparent")
        entry_row.pack(fill="x", padx=14, pady=(0, 12))

        self.entry_custom = ctk.CTkEntry(
            entry_row,
            placeholder_text="ex: person, car, cup",
            fg_color="#18181b",
            border_color="#27272a",
            height=30
        )
        self.entry_custom.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.entry_custom.bind("<Return>", lambda e: self._apply_custom_filter())

        self.btn_apply = ctk.CTkButton(
            entry_row,
            text="OK",
            width=36,
            height=30,
            fg_color="#27272a",
            hover_color="#3f3f46",
            command=self._apply_custom_filter
        )
        self.btn_apply.pack(side="right")

        # --- Section 4: Confidence Tuning ---
        card_conf = self._make_card(right_frame, "Seuil de Confiance")

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

        # --- Section 5: Telemetry Status ---
        card_telemetry = self._make_card(right_frame, "Télémétrie Système")

        self.lbl_fps = ctk.CTkLabel(card_telemetry, text=f"FPS : 0.0 ({self.device.upper()})", font=ctk.CTkFont(size=12, weight="bold"), text_color="#f4f4f5")
        self.lbl_fps.pack(anchor="w", padx=14, pady=(8, 2))

        self.lbl_det_count = ctk.CTkLabel(card_telemetry, text="Détections : 0", font=ctk.CTkFont(size=12), text_color="#a1a1aa")
        self.lbl_det_count.pack(anchor="w", padx=14, pady=(0, 2))

        self.lbl_src_name = ctk.CTkLabel(card_telemetry, text="Source : -", font=ctk.CTkFont(size=11), text_color="#71717a")
        self.lbl_src_name.pack(anchor="w", padx=14, pady=(0, 12))

        # Quit Button
        self.btn_close = ctk.CTkButton(
            right_frame,
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
    # STREAM ACQUISITION & HARDWARE INITIALIZATION
    # ==============================================================
    def _open_stream(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.camera_error_msg = None

            if self.is_webcam:
                # Use macOS AVFoundation directly
                backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(0, backend)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                # Verify opened
                opened = self.cap.isOpened()
                if opened:
                    ret, test_frame = self.cap.read()
                    if not ret or test_frame is None:
                        opened = False

                if not opened:
                    self.camera_error_msg = "Accès Webcam refusé sur macOS.\nAutorisez Terminal dans Réglages Système > Confidentialité > Caméra."
                    self.lbl_stream_status.configure(text="✕ Caméra non autorisée", text_color="#ef4444")
                    self.btn_fix_camera.pack(fill="x", padx=14, pady=(0, 10))
                else:
                    self.lbl_stream_status.configure(text="● Webcam FaceTime HD active", text_color="#22c55e")
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
                    self.lbl_stream_status.configure(text="✕ Fichier introuvable", text_color="#ef4444")
                    return

                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
                self.timeline_slider.configure(to=self.total_frames - 1)

                dur_sec = int(self.total_frames / self.source_fps)
                self.time_lbl_right.configure(text=f"{dur_sec // 60:02d}:{dur_sec % 60:02d}")
                self.lbl_src_name.configure(text=f"Source : {os.path.basename(self.source_path)}")
                self.lbl_stream_status.configure(text="● Vidéo chargée", text_color="#22c55e")

            self.current_frame_idx = 0

    def _open_macos_camera_settings(self):
        try:
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"], check=False)
        except Exception:
            pass

    def _on_source_change(self, value):
        self.is_webcam = ("Webcam" in value)
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
    # CONTROLS & INSTANT REACTIVITY (Every click updates immediately)
    # ==============================================================
    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.configure(text="▶ Lecture" if not self.is_playing else "⏸ Pause")
        self.lbl_stream_status.configure(
            text="⏸ En pause" if not self.is_playing else "● En lecture",
            text_color="#f59e0b" if not self.is_playing else "#22c55e"
        )
        if not self.is_playing:
            self.reprocess_current_frame()

    def rewind_video(self):
        if self.is_webcam or not self.cap:
            return
        with self.lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.timeline_slider.set(0)
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

    def _on_display_mode_change(self, mode_str):
        mapping = {
            "Masques + Boîtes": "both",
            "Masques seuls": "masks",
            "Boîtes seules": "boxes"
        }
        self.display_mode = mapping.get(mode_str, "both")
        self.reprocess_current_frame()

    def _on_param_modified(self):
        self.show_labels = self.sw_labels.get()
        self.mirror_mode = self.sw_mirror.get()
        self.reprocess_current_frame()

    def _on_preset_selected(self, choice):
        self.active_targets = self.target_presets.get(choice, None)
        self.entry_custom.delete(0, "end")
        self.reprocess_current_frame()

    def _apply_custom_filter(self):
        text = self.entry_custom.get().strip()
        if not text:
            self.active_targets = None
            self.opt_preset.set("Tous les objets (80)")
        else:
            items = [i.strip().lower() for i in text.split(",") if i.strip()]
            self.active_targets = items if items else None
            self.opt_preset.set("Filtre personnalisé")
        self.reprocess_current_frame()

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
            messagebox.showinfo("Capture Sauvegardée", f"Image enregistrée avec succès :\n{filename}")

    # ==============================================================
    # IMMEDIATE REPROCESSING (Makes every control 100% active on click)
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

        # Resolve target IDs
        target_ids = None
        if self.active_targets is not None:
            all_names = self.model.names
            name_to_id = {v.lower(): k for k, v in all_names.items()}
            target_ids = [name_to_id[t] for t in self.active_targets if t in name_to_id]
            if not target_ids and len(self.active_targets) > 0:
                target_ids = [-1]

        try:
            results = self.model.predict(
                display_frame,
                device=self.device,
                classes=target_ids,
                conf=self.conf_threshold,
                verbose=False
            )
            r = results[0]
        except Exception:
            return display_frame, 0

        overlay = display_frame.copy()
        all_names = self.model.names

        # 1. Masks
        if self.display_mode in ["both", "masks"] and r.masks is not None and len(r.masks) > 0:
            for mask_coords, box in zip(r.masks.xy, r.boxes):
                cls_id = int(box.cls[0].item())
                color = [int(c) for c in self.color_palette[cls_id % len(self.color_palette)]]
                if len(mask_coords) > 0:
                    poly = np.array(mask_coords, dtype=np.int32)
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(display_frame, [poly], isClosed=True, color=color, thickness=2)

            alpha = 0.40
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

        # 2. Bounding Boxes & Badges
        count = len(r.boxes) if r.boxes is not None else 0
        if self.display_mode in ["both", "boxes"] and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = [int(c) for c in self.color_palette[cls_id % len(self.color_palette)]]

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                if self.show_labels:
                    label = f"{all_names[cls_id]} {conf:.2f}"
                    font_scale = 0.50
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    by1 = max(0, y1 - th - baseline - 4)
                    by2 = y1
                    bx2 = min(w, x1 + tw + 6)
                    cv2.rectangle(display_frame, (x1, by1), (bx2, by2), color, -1)
                    cv2.putText(display_frame, label, (x1 + 3, y1 - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return display_frame, count

    # ==============================================================
    # CONTINUOUS CAPTURE THREAD
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
                        # Auto loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_idx = 0
                        ret, frame = self.cap.read()
                    if not ret or frame is None:
                        time.sleep(0.05)
                        continue

                self.current_raw_frame = frame
                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Run detection
            annotated, count = self._infer_and_annotate(frame)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with self.lock:
                self.latest_display_frame = rgb
                self.detected_count = count

            # Calculate FPS
            cost = time.perf_counter() - t_start
            fps = 1.0 / cost if cost > 0 else 0
            self.fps_samples.append(fps)
            if len(self.fps_samples) > 20:
                self.fps_samples.pop(0)
            self.current_fps = sum(self.fps_samples) / len(self.fps_samples)

            # Throttling
            target_delay = (1.0 / self.source_fps) / max(0.1, self.speed_factor)
            remaining = target_delay - cost
            if remaining > 0:
                time.sleep(remaining)

    # ==============================================================
    # GUI REFRESH TIMER (Main Thread)
    # ==============================================================
    def _gui_refresh(self):
        if not self.running:
            return

        with self.lock:
            frame = self.latest_display_frame
            cur_idx = self.current_frame_idx
            det_count = self.detected_count
            fps_val = self.current_fps
            err = self.camera_error_msg

        # Display Frame or Error Message
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

        # Update Timeline & Telemetry
        if not self.is_webcam and self.total_frames > 0:
            self.timeline_slider.set(cur_idx)
            sec = int(cur_idx / self.source_fps)
            self.time_lbl_left.configure(text=f"{sec // 60:02d}:{sec % 60:02d}")

        self.lbl_fps.configure(text=f"FPS : {fps_val:.1f} ({self.device.upper()})")
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
