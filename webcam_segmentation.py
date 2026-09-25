"""
Real-Time Webcam Detection & Instance Segmentation
--------------------------------------------------
Performs live real-time object detection and instance segmentation
from your Mac webcam (FaceTime HD camera / external camera) using YOLO11-seg.

Features:
- Live streaming from camera index (default: 0)
- Hardware acceleration (Apple Silicon MPS)
- Mirror mode (horizontal flip) enabled by default for natural interaction
- Class filtering (e.g. --target person,cell phone,bottle or all 80 COCO classes)
- Alpha-blended polygonal instance masks + bounding boxes
- Real-time FPS telemetry and detection counts HUD
- Interactive keyboard controls:
    * 'q' or 'Esc' : Quit
    * 's'           : Save snapshot
    * 'p' or Space : Pause / freeze frame
    * 'f'           : Toggle horizontal mirror mode
"""

import argparse
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def get_default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_color_palette(num_classes=80):
    np.random.seed(42)
    return np.random.randint(50, 255, size=(num_classes, 3), dtype=np.uint8)


def run_webcam(
    cam_index: int = 0,
    model_name: str = "yolo11n-seg.pt",
    target_classes: list = None,
    conf_threshold: float = 0.35,
    width: int = 1280,
    height: int = 720,
    mirror: bool = True,
    device: str = None
):
    if device is None:
        device = get_default_device()

    print("=" * 65)
    print("Real-Time Webcam Detection & Instance Segmentation")
    print("=" * 65)
    print(f"Camera index:   {cam_index}")
    print(f"Requested res:  {width}x{height}")
    print(f"Model:          {model_name}")
    print(f"Target filter:  {target_classes if target_classes else 'ALL COCO classes (80)'}")
    print(f"Confidence:     {conf_threshold}")
    print(f"Compute device: {device.upper()}")
    print("=" * 65)

    # 1. Load YOLO model
    print(f"\n[INFO] Loading {model_name} on {device.upper()}...")
    model = YOLO(model_name)

    all_names = model.names
    name_to_id = {name.lower(): idx for idx, name in all_names.items()}
    target_ids = None
    if target_classes and "all" not in [t.lower() for t in target_classes]:
        target_ids = []
        for t in target_classes:
            t_clean = t.strip().lower()
            if t_clean in name_to_id:
                target_ids.append(name_to_id[t_clean])
            else:
                print(f"[WARNING] Unknown class '{t_clean}'.")
        if not target_ids:
            print("[ERROR] None of the requested target classes were recognized.")
            sys.exit(1)
        print(f"[INFO] Filtering active for: {[all_names[i] for i in target_ids]}")

    # 2. Open camera
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {cam_index}.")
        print("  Tips on macOS:")
        print("  1. Make sure your Terminal / IDE has Camera permissions in System Settings.")
        print("  2. If using an external camera, try --cam-index 1.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera stream opened: {actual_w}x{actual_h}")

    color_palette = generate_color_palette(len(all_names))
    fps_history = []
    paused = False
    snap_count = 0

    print("\n[INFO] Live camera active!")
    print("  [Controls] Space/p: Pause | s: Save snapshot | f: Toggle mirror | q/Esc: Exit\n")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)  # Horizontal mirror

            t_start = time.perf_counter()

            # YOLO inference
            results = model.predict(
                frame,
                device=device,
                classes=target_ids,
                conf=conf_threshold,
                verbose=False
            )
            r = results[0]

            # Mask overlay
            overlay = frame.copy()
            detected_count = 0

            # Render masks
            if r.masks is not None and len(r.masks) > 0:
                for mask_coords, box in zip(r.masks.xy, r.boxes):
                    cls_id = int(box.cls[0].item())
                    color = [int(c) for c in color_palette[cls_id % len(color_palette)]]

                    if len(mask_coords) > 0:
                        polygon = np.array(mask_coords, dtype=np.int32)
                        cv2.fillPoly(overlay, [polygon], color)
                        cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)

            alpha = 0.45
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Render bounding boxes and text labels
            if r.boxes is not None and len(r.boxes) > 0:
                detected_count = len(r.boxes)
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = [int(c) for c in color_palette[cls_id % len(color_palette)]]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label_text = f"{all_names[cls_id]} {conf:.2f}"
                    font_scale = 0.55
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    by1 = max(0, y1 - th - baseline - 4)
                    by2 = y1
                    bx2 = min(actual_w, x1 + tw + 6)
                    cv2.rectangle(frame, (x1, by1), (bx2, by2), color, -1)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1 + 3, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )

            # Performance telemetry
            t_cost = time.perf_counter() - t_start
            fps = 1.0 / t_cost if t_cost > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # HUD Display
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (10, 10), (330, 80), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, f"LIVE FPS: {avg_fps:.1f} ({device.upper()})", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 127), 2, cv2.LINE_AA)
            target_info = f"Filter: {','.join([all_names[i] for i in target_ids])}" if target_ids else "Filter: ALL"
            cv2.putText(frame, f"{target_info} | Detected: {detected_count}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow("Live Webcam Segmentation (YOLO11-seg)", frame)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in [ord("q"), 27]:
            print("\n[INFO] Stopped by user.")
            break
        elif key in [ord("p"), 32]:
            paused = not paused
            print(f"[INFO] {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord("f"):
            mirror = not mirror
            print(f"[INFO] Mirror mode: {'ON' if mirror else 'OFF'}")
        elif key == ord("s"):
            snap_count += 1
            snap_path = f"webcam_snapshot_{snap_count:03d}.jpg"
            cv2.imwrite(snap_path, frame)
            print(f"[INFO] Snapshot saved to: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera closed cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Webcam Object Detection and Instance Segmentation using YOLO11-seg"
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=0,
        help="Camera device index (default: 0 for built-in Mac FaceTime HD camera)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-seg.pt",
        help="YOLO model path or name (default: yolo11n-seg.pt)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        help="Comma-separated target classes to detect/segment (e.g. 'person', 'cell phone,cup', or 'all')"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold between 0.0 and 1.0 (default: 0.35)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width in pixels (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height in pixels (default: 720)"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror flip"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Execution device (defaults to auto-detecting MPS on Apple Silicon, CUDA, or CPU)"
    )

    args = parser.parse_args()

    targets = [t.strip() for t in args.target.split(",")] if args.target else None

    run_webcam(
        cam_index=args.cam_index,
        model_name=args.model,
        target_classes=targets,
        conf_threshold=args.conf,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
        device=args.device
    )


if __name__ == "__main__":
    main()
