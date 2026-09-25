"""
Vision Studio — Real-Time Video & Webcam Segmentation with Side Controls
-------------------------------------------------------------------------
All-in-one studio interface with:
- Left pane: Live video playback with interactive timeline scrubber and transport controls
- Right pane: Rich side control panel (Display modes, Live target filtering,
  Playback speed, Confidence tuning, Telemetry HUD, and Source switcher).
"""

import os
import sys
import time
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
    return np.random.randint(50, 255, size=(num_classes, 3), dtype=np.uint8)


class VisionStudio(ctk.CTk):
    def __init__(self, initial_source=None, is_webcam=False):
        super().__init__()

        self.title("Vision Studio — Détection & Segmentation Temps Réel")
        self.geometry("1300x820")
        self.minsize(1100, 720)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Runtime State
        self.device = get_default_device()
        self.model_name = "yolo11n-seg.pt"
        self.model = None

        self.is_webcam = is_webcam
        self.source_path = initial_source or os.path.join(PROJECT_DIR, "samples", "pedestrians.avi")
        if not os.path.exists(self.source_path) and not self.is_webcam:
            self.source_path = os.path.join(PROJECT_DIR, "samples", "pedestrians.avi")

        self.cap = None
        self.total_frames = 0
        self.source_fps = 25.0
        self.current_frame_idx = 0
        self.is_playing = True
        self.is_seeking = False
        self.speed_factor = 1.0
        self.mirror_mode = True

        # Visualization settings
        self.display_mode = "both"  # "both", "masks", "boxes"
        self.show_labels = True
        self.conf_threshold = 0.35
        self.active_targets = None  # None = ALL COCO classes

        # Predefined target categories
        self.target_presets = {
            "Tous les objets (80)": None,
            "🚶 Personnes uniquement": ["person"],
            "🚗 Véhicules": ["car", "bicycle", "motorcycle", "bus", "truck"],
            "📱 Électronique": ["cell phone", "laptop", "tv"],
            "☕ Objets du quotidien": ["bottle", "cup", "chair", "backpack", "handbag"]
        }

        # Telemetry
        self.fps_meter = []
        self.current_fps = 0.0
        self.detected_count = 0

        # Video Threading
        self.running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        self.color_palette = None

        # Build UI
        self._init_model()
        self._build_layout()
        self._open_source()

        # Start background capture thread & GUI render timer
        self.capture_thread = threading.Thread(target=self._capture_and_infer_loop, daemon=True)
        self.capture_thread.start()

        self._render_loop()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_model(self):
        try:
            self.model = YOLO(self.model_name)
            self.color_palette = generate_color_palette(len(self.model.names))
        except Exception as e:
            messagebox.showerror("Erreur Modèle", f"Impossible de charger le modèle {self.model_name}:\n{e}")
            sys.exit(1)

    def _build_layout(self):
        # Main Grid: Left = Video & Media controls, Right = Side settings panel
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=380)
        self.grid_rowconfigure(0, weight=1)

        # ==============================================================
        # LEFT PANE: VIDEO SCREEN & PLAYBACK CONTROLS
        # ==============================================================
        left_pane = ctk.CTkFrame(self, corner_radius=0, fg_color="#121316")
        left_pane.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        left_pane.grid_rowconfigure(0, weight=1)
        left_pane.grid_columnconfigure(0, weight=1)

        # Video Canvas
        self.canvas_frame = ctk.CTkFrame(left_pane, fg_color="#08080a", corner_radius=12)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#08080a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Bottom Bar: Timeline & Transport
        bottom_bar = ctk.CTkFrame(left_pane, fg_color="#18191e", corner_radius=12)
        bottom_bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))

        # Timeline Scrubber (for video files)
        timeline_row = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        timeline_row.pack(fill="x", padx=16, pady=(10, 4))

        self.time_lbl_left = ctk.CTkLabel(timeline_row, text="00:00", font=ctk.CTkFont(size=12, weight="bold"), width=50)
        self.time_lbl_left.pack(side="left")

        self.timeline_slider = ctk.CTkSlider(
            timeline_row,
            from_=0,
            to=100,
            command=self._on_timeline_seek,
            progress_color="#3b82f6"
        )
        self.timeline_slider.set(0)
        self.timeline_slider.pack(side="left", fill="x", expand=True, padx=10)

        self.time_lbl_right = ctk.CTkLabel(timeline_row, text="00:00", font=ctk.CTkFont(size=12), text_color="#9da5b4", width=50)
        self.time_lbl_right.pack(side="right")

        # Transport Controls Row
        control_row = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        control_row.pack(fill="x", padx=16, pady=(4, 10))

        # Play/Pause & Restart
        self.btn_play_pause = ctk.CTkButton(
            control_row,
            text="⏸️ Pause",
            font=ctk.CTkFont(size=13, weight="bold"),
            width=105,
            height=34,
            fg_color="#3b82f6",
            hover_color="#2563eb",
            command=self.toggle_play_pause
        )
        self.btn_play_pause.pack(side="left", padx=(0, 8))

        self.btn_restart = ctk.CTkButton(
            control_row,
            text="🔄 Recommencer",
            font=ctk.CTkFont(size=13),
            width=125,
            height=34,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self.restart_video
        )
        self.btn_restart.pack(side="left", padx=4)

        # Speed selector
        ctk.CTkLabel(control_row, text="Vitesse :", font=ctk.CTkFont(size=12)).pack(side="left", padx=(18, 6))
        self.speed_seg = ctk.CTkSegmentedButton(
            control_row,
            values=["0.25x", "0.5x", "1x", "1.5x", "2x"],
            command=self._on_speed_changed,
            height=30
        )
        self.speed_seg.set("1x")
        self.speed_seg.pack(side="left", padx=2)

        # Snapshot button
        self.btn_snapshot = ctk.CTkButton(
            control_row,
            text="📸 Capture",
            font=ctk.CTkFont(size=13),
            width=100,
            height=34,
            fg_color="#10b981",
            hover_color="#059669",
            command=self.take_snapshot
        )
        self.btn_snapshot.pack(side="right")

        # ==============================================================
        # RIGHT PANE: SIDEBAR SETTINGS PANEL
        # ==============================================================
        right_pane = ctk.CTkScrollableFrame(self, corner_radius=0, width=380, fg_color="#1a1b20")
        right_pane.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        # Header Title
        ctk.CTkLabel(
            right_pane,
            text="⚙️ Réglages Vision",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(anchor="w", padx=16, pady=(16, 4))

        ctk.CTkLabel(
            right_pane,
            text="Contrôles dynamiques en direct",
            font=ctk.CTkFont(size=12),
            text_color="#9da5b4"
        ).pack(anchor="w", padx=16, pady=(0, 14))

        # --- Section 1: Source Selector ---
        sec_source = self._create_card(right_pane, "Source Vidéo")

        self.source_seg = ctk.CTkSegmentedButton(
            sec_source,
            values=["📁 Fichier Vidéo", "🎥 Webcam Direct"],
            command=self._on_source_switch,
            height=36
        )
        self.source_seg.set("🎥 Webcam Direct" if self.is_webcam else "📁 Fichier Vidéo")
        self.source_seg.pack(fill="x", padx=14, pady=(8, 8))

        self.btn_choose_video = ctk.CTkButton(
            sec_source,
            text="Choisir un autre fichier vidéo...",
            font=ctk.CTkFont(size=12),
            fg_color="#374151",
            hover_color="#4b5563",
            height=32,
            command=self.choose_video_file
        )
        self.btn_choose_video.pack(fill="x", padx=14, pady=(0, 12))

        # --- Section 2: Mode d'Affichage & Rendu ---
        sec_mode = self._create_card(right_pane, "Type de Détection & Rendu")

        ctk.CTkLabel(sec_mode, text="Mode de segmentation :", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=14, pady=(6, 4))
        self.display_seg = ctk.CTkSegmentedButton(
            sec_mode,
            values=["Masques + Boîtes", "Masques seuls", "Boîtes seules"],
            command=self._on_display_mode_changed,
            height=32
        )
        self.display_seg.set("Masques + Boîtes")
        self.display_seg.pack(fill="x", padx=14, pady=(0, 10))

        self.sw_labels = ctk.CTkSwitch(sec_mode, text="Afficher labels & scores de confiance", command=self._on_toggle_labels)
        self.sw_labels.select()
        self.sw_labels.pack(anchor="w", padx=14, pady=4)

        self.sw_mirror = ctk.CTkSwitch(sec_mode, text="Mode miroir (caméra horizontale)", command=self._on_toggle_mirror)
        self.sw_mirror.select()
        self.sw_mirror.pack(anchor="w", padx=14, pady=(4, 12))

        # --- Section 3: Filtres d'Objets en Direct ---
        sec_targets = self._create_card(right_pane, "Objets Cibles (Filtrage en Direct)")

        self.target_option = ctk.CTkOptionMenu(
            sec_targets,
            values=list(self.target_presets.keys()),
            command=self._on_target_preset_changed,
            height=34
        )
        self.target_option.set("Tous les objets (80)")
        self.target_option.pack(fill="x", padx=14, pady=(8, 8))

        ctk.CTkLabel(sec_targets, text="Ou tapez un filtre personnalisé :", font=ctk.CTkFont(size=11), text_color="#9da5b4").pack(anchor="w", padx=14, pady=(2, 2))
        self.entry_custom = ctk.CTkEntry(
            sec_targets,
            placeholder_text="ex: person, car, dog, backpack",
            height=32
        )
        self.entry_custom.pack(fill="x", padx=14, pady=(0, 6))
        self.entry_custom.bind("<Return>", self._on_custom_filter_enter)

        self.btn_apply_custom = ctk.CTkButton(
            sec_targets,
            text="Appliquer le filtre texte",
            font=ctk.CTkFont(size=11),
            fg_color="#374151",
            hover_color="#4b5563",
            height=28,
            command=self._apply_custom_filter
        )
        self.btn_apply_custom.pack(fill="x", padx=14, pady=(0, 12))

        # --- Section 4: Réglage Confiance & Paramètres ---
        sec_params = self._create_card(right_pane, "Confiance & Sensibilité")

        self.lbl_conf_text = ctk.CTkLabel(sec_params, text="Seuil de confiance : 35%", font=ctk.CTkFont(size=12))
        self.lbl_conf_text.pack(anchor="w", padx=14, pady=(8, 4))

        self.conf_slider = ctk.CTkSlider(
            sec_params,
            from_=0.10,
            to=0.90,
            command=self._on_conf_slider,
            progress_color="#10b981"
        )
        self.conf_slider.set(0.35)
        self.conf_slider.pack(fill="x", padx=14, pady=(0, 14))

        # --- Section 5: Télémétrie & Matériel ---
        sec_telemetry = self._create_card(right_pane, "Télémétrie en Direct")

        self.lbl_telemetry_fps = ctk.CTkLabel(
            sec_telemetry,
            text=f"FPS Réel : 0.0 ({self.device.upper()})",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#10b981"
        )
        self.lbl_telemetry_fps.pack(anchor="w", padx=14, pady=(8, 2))

        self.lbl_telemetry_det = ctk.CTkLabel(
            sec_telemetry,
            text="Objets détectés : 0",
            font=ctk.CTkFont(size=12)
        )
        self.lbl_telemetry_det.pack(anchor="w", padx=14, pady=(0, 2))

        self.lbl_telemetry_src = ctk.CTkLabel(
            sec_telemetry,
            text=f"Source : {os.path.basename(self.source_path)}",
            font=ctk.CTkFont(size=11),
            text_color="#9da5b4"
        )
        self.lbl_telemetry_src.pack(anchor="w", padx=14, pady=(0, 12))

        # --- Section 6: Quitter ---
        self.btn_quit = ctk.CTkButton(
            right_pane,
            text="❌ Fermer l'Interface",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            fg_color="#ef4444",
            hover_color="#dc2626",
            command=self.on_close
        )
        self.btn_quit.pack(fill="x", padx=16, pady=(16, 24))

    def _create_card(self, parent, title):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color="#212229")
        card.pack(fill="x", padx=16, pady=8)
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=14, pady=(10, 4))
        return card

    # ==============================================================
    # VIDEO SOURCE & STREAM MANAGEMENT
    # ==============================================================
    def _open_source(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()

            if self.is_webcam:
                self.cap = cv2.VideoCapture(0)
                self.total_frames = 0
                self.source_fps = 30.0
                self.btn_restart.configure(state="disabled")
                self.timeline_slider.configure(state="disabled")
                self.time_lbl_left.configure(text="LIVE")
                self.time_lbl_right.configure(text="LIVE")
                self.lbl_telemetry_src.configure(text="Source : Webcam FaceTime HD")
            else:
                self.cap = cv2.VideoCapture(self.source_path)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
                self.timeline_slider.configure(state="normal", to=self.total_frames - 1)
                self.btn_restart.configure(state="normal")
                dur_sec = int(self.total_frames / self.source_fps)
                self.time_lbl_right.configure(text=f"{dur_sec // 60:02d}:{dur_sec % 60:02d}")
                self.lbl_telemetry_src.configure(text=f"Source : {os.path.basename(self.source_path)}")

            self.current_frame_idx = 0

    def choose_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=[("Fichiers Vidéo", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Tous les fichiers", "*.*")],
            initialdir=os.path.join(PROJECT_DIR, "samples")
        )
        if file_path:
            self.source_path = file_path
            self.is_webcam = False
            self.source_seg.set("📁 Fichier Vidéo")
            self._open_source()

    def _on_source_switch(self, choice):
        if "Webcam" in choice:
            self.is_webcam = True
            self.btn_choose_video.configure(state="disabled")
        else:
            self.is_webcam = False
            self.btn_choose_video.configure(state="normal")
        self._open_source()

    # ==============================================================
    # CONTROLS CALLBACKS
    # ==============================================================
    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        self.btn_play_pause.configure(
            text="▶️ Lecture" if not self.is_playing else "⏸️ Pause",
            fg_color="#10b981" if not self.is_playing else "#3b82f6"
        )

    def restart_video(self):
        if self.is_webcam or not self.cap:
            return
        with self.lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.timeline_slider.set(0)

    def _on_timeline_seek(self, value):
        if self.is_webcam or not self.cap:
            return
        target_frame = int(value)
        with self.lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame_idx = target_frame

    def _on_speed_changed(self, speed_str):
        mapping = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "1.5x": 1.5, "2x": 2.0}
        self.speed_factor = mapping.get(speed_str, 1.0)

    def _on_display_mode_changed(self, mode_str):
        mapping = {
            "Masques + Boîtes": "both",
            "Masques seuls": "masks",
            "Boîtes seules": "boxes"
        }
        self.display_mode = mapping.get(mode_str, "both")

    def _on_toggle_labels(self):
        self.show_labels = self.sw_labels.get()

    def _on_toggle_mirror(self):
        self.mirror_mode = self.sw_mirror.get()

    def _on_target_preset_changed(self, choice):
        self.active_targets = self.target_presets.get(choice, None)
        self.entry_custom.delete(0, "end")

    def _on_custom_filter_enter(self, event):
        self._apply_custom_filter()

    def _apply_custom_filter(self):
        text = self.entry_custom.get().strip()
        if not text:
            self.active_targets = None
            self.target_option.set("Tous les objets (80)")
            return

        items = [i.strip().lower() for i in text.split(",") if i.strip()]
        self.active_targets = items if items else None
        self.target_option.set("Filtre personnalisé")

    def _on_conf_slider(self, val):
        self.conf_threshold = float(val)
        self.lbl_conf_text.configure(text=f"Seuil de confiance : {int(self.conf_threshold * 100)}%")

    def take_snapshot(self):
        with self.lock:
            frame_to_save = self.latest_frame
        if frame_to_save is not None:
            snap_name = f"snapshot_{int(time.time())}.jpg"
            # Convert RGB back to BGR for cv2 saving
            bgr = cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(snap_name, bgr)
            messagebox.showinfo("Capture Sauvegardée", f"Capture d'écran enregistrée avec succès sous :\n{snap_name}")

    # ==============================================================
    # CAPTURE & INFERENCE WORKER THREAD
    # ==============================================================
    def _capture_and_infer_loop(self):
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
                if not ret:
                    if not self.is_webcam and self.total_frames > 0:
                        # Auto-loop video file
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_idx = 0
                        ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue

                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            if self.is_webcam and self.mirror_mode:
                frame = cv2.flip(frame, 1)

            # Resolve target class IDs
            target_ids = None
            if self.active_targets is not None:
                all_names = self.model.names
                name_to_id = {v.lower(): k for k, v in all_names.items()}
                target_ids = [name_to_id[t] for t in self.active_targets if t in name_to_id]
                if not target_ids and len(self.active_targets) > 0:
                    target_ids = [-1]  # Match nothing

            # YOLO inference
            try:
                results = self.model.predict(
                    frame,
                    device=self.device,
                    classes=target_ids,
                    conf=self.conf_threshold,
                    verbose=False
                )
                r = results[0]
            except Exception as e:
                time.sleep(0.05)
                continue

            # Render overlay according to display_mode
            annotated = self._render_detection(frame, r)
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with self.lock:
                self.latest_frame = rgb_frame
                self.detected_count = len(r.boxes) if r.boxes is not None else 0

            # Calculate FPS
            t_cost = time.perf_counter() - t_start
            fps = 1.0 / t_cost if t_cost > 0 else 0
            self.fps_meter.append(fps)
            if len(self.fps_meter) > 20:
                self.fps_meter.pop(0)
            self.current_fps = sum(self.fps_meter) / len(self.fps_meter)

            # Frame rate throttle based on playback speed
            target_frame_time = (1.0 / self.source_fps) / max(0.1, self.speed_factor)
            remaining_sleep = target_frame_time - t_cost
            if remaining_sleep > 0:
                time.sleep(remaining_sleep)

    def _render_detection(self, frame, r):
        h, w, _ = frame.shape
        overlay = frame.copy()
        all_names = self.model.names

        # 1. Render Masks (if enabled)
        if self.display_mode in ["both", "masks"] and r.masks is not None and len(r.masks) > 0:
            for mask_coords, box in zip(r.masks.xy, r.boxes):
                cls_id = int(box.cls[0].item())
                color = [int(c) for c in self.color_palette[cls_id % len(self.color_palette)]]
                if len(mask_coords) > 0:
                    polygon = np.array(mask_coords, dtype=np.int32)
                    cv2.fillPoly(overlay, [polygon], color)
                    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)

            alpha = 0.45
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 2. Render Boxes and Labels (if enabled)
        if self.display_mode in ["both", "boxes"] and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = [int(c) for c in self.color_palette[cls_id % len(self.color_palette)]]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if self.show_labels:
                    label_text = f"{all_names[cls_id]} {conf:.2f}"
                    font_scale = 0.55
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    by1 = max(0, y1 - th - baseline - 4)
                    by2 = y1
                    bx2 = min(w, x1 + tw + 6)
                    cv2.rectangle(frame, (x1, by1), (bx2, by2), color, -1)
                    cv2.putText(frame, label_text, (x1 + 3, y1 - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    # ==============================================================
    # GUI RENDERING TIMER (MAIN THREAD)
    # ==============================================================
    def _render_loop(self):
        if not self.running:
            return

        with self.lock:
            frame = self.latest_frame
            cur_idx = self.current_frame_idx
            det_count = self.detected_count
            fps_val = self.current_fps

        # Display Frame in Tkinter Canvas
        if frame is not None:
            canvas_w = max(100, self.canvas.winfo_width())
            canvas_h = max(100, self.canvas.winfo_height())
            fh, fw, _ = frame.shape

            # Compute aspect ratio fit
            scale = min(canvas_w / fw, canvas_h / fh)
            new_w = int(fw * scale)
            new_h = int(fh * scale)

            if new_w > 0 and new_h > 0:
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                img = Image.fromarray(resized)
                self.tk_photo = ImageTk.PhotoImage(image=img)

                # Center image on canvas
                cx = canvas_w // 2
                cy = canvas_h // 2
                self.canvas.delete("all")
                self.canvas.create_image(cx, cy, image=self.tk_photo, anchor="center")

        # Update Timeline Scrubber
        if not self.is_webcam and self.total_frames > 0:
            self.timeline_slider.set(cur_idx)
            sec = int(cur_idx / self.source_fps)
            self.time_lbl_left.configure(text=f"{sec // 60:02d}:{sec % 60:02d}")

        # Update Telemetry HUD
        self.lbl_telemetry_fps.configure(text=f"FPS Réel : {fps_val:.1f} ({self.device.upper()})")
        self.lbl_telemetry_det.configure(text=f"Objets détectés : {det_count}")

        # Schedule next refresh (approx 30-40 Hz for GUI)
        self.after(25, self._render_loop)

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
