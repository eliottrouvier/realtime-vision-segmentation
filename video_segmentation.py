"""
Real-Time Video Detection & Instance Segmentation
--------------------------------------------------
Uses YOLO11-seg (or YOLOv8-seg) for high-framerate object detection
and pixel-level instance segmentation on video streams.

Features:
- Hardware acceleration (Apple Silicon MPS / CUDA / CPU)
- Class filtering (e.g. --target person,car or all 80 COCO classes)
- Alpha-blended polygonal instance masks + bounding boxes
- Real-time FPS overlay & performance telemetry
- Interactive controls: Pause (Space/p), Save frame (s), Quit (q/Esc)
- Video export option (--output output.mp4)
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def get_default_device():
    """Detect the fastest available hardware accelerator."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_color_palette(num_classes=80):
    """Generate consistent, distinct colors for visualization."""
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors


def process_video(
    source_path: str,
    model_name: str = "yolo11n-seg.pt",
    target_classes: list = None,
    conf_threshold: float = 0.35,
    output_path: str = None,
    device: str = None,
    display: bool = True
):
    if device is None:
        device = get_default_device()

    print("=" * 65)
    print("Real-Time Video Detection & Instance Segmentation")
    print("=" * 65)
    print(f"Source video:   {source_path}")
    print(f"Model:          {model_name}")
    print(f"Target filter:  {target_classes if target_classes else 'ALL COCO classes (80)'}")
    print(f"Confidence:     {conf_threshold}")
    print(f"Compute device: {device.upper()}")
    if output_path:
        print(f"Saving output:  {output_path}")
    print("=" * 65)

    if not os.path.exists(source_path):
        print(f"Error: Source video file '{source_path}' does not exist.")
        sys.exit(1)

    # 1. Load YOLO segmentation model
    print(f"\n[INFO] Loading {model_name} on {device.upper()}...")
    model = YOLO(model_name)

    # Resolve target classes to class IDs if specified
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
                print(f"[WARNING] Unknown class '{t_clean}'. Available classes include:")
                print(f"          {list(all_names.values())[:10]}... (80 total)")
        if not target_ids:
            print("[ERROR] None of the requested target classes were recognized.")
            sys.exit(1)
        print(f"[INFO] Filtering active for class IDs: {target_ids} ({[all_names[i] for i in target_ids]})")

    # 2. Open video source
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {source_path}")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video resolution: {frame_width}x{frame_height} | FPS: {source_fps:.1f} | Frames: {total_frames}")

    # 3. Setup VideoWriter if saving output
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, source_fps, (frame_width, frame_height))

    color_palette = generate_color_palette(len(all_names))
    fps_history = []
    paused = False
    frame_idx = 0
    start_total_time = time.time()

    print("\n[INFO] Starting playback...")
    if display:
        print("  [Controls] Space/p: Pause | s: Save screenshot | q/Esc: Exit\n")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of video stream reached.")
                break
            frame_idx += 1
            t_start = time.perf_counter()

            # Run inference
            results = model.predict(
                frame,
                device=device,
                classes=target_ids,
                conf=conf_threshold,
                verbose=False
            )
            r = results[0]

            # Create mask overlay
            overlay = frame.copy()
            detected_count = 0

            # Render segmentation masks if available
            if r.masks is not None and len(r.masks) > 0:
                for mask_coords, box in zip(r.masks.xy, r.boxes):
                    cls_id = int(box.cls[0].item())
                    color = [int(c) for c in color_palette[cls_id % len(color_palette)]]

                    if len(mask_coords) > 0:
                        polygon = np.array(mask_coords, dtype=np.int32)
                        cv2.fillPoly(overlay, [polygon], color)
                        cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)

            # Alpha blend masks with frame (transparency factor 0.45)
            alpha = 0.45
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Render bounding boxes and clean label badges
            if r.boxes is not None and len(r.boxes) > 0:
                detected_count = len(r.boxes)
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = [int(c) for c in color_palette[cls_id % len(color_palette)]]

                    # Box outline
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Text label with background banner
                    label_text = f"{all_names[cls_id]} {conf:.2f}"
                    font_scale = 0.55
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    # Ensure banner stays within image bounds
                    by1 = max(0, y1 - th - baseline - 4)
                    by2 = y1
                    bx2 = min(frame_width, x1 + tw + 6)
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

            # Calculate FPS
            t_cost = time.perf_counter() - t_start
            fps = 1.0 / t_cost if t_cost > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Telemetry Banner Overlay (HUD)
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (10, 10), (330, 80), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)
            
            # FPS & Device info
            cv2.putText(frame, f"FPS: {avg_fps:.1f} ({device.upper()})", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 127), 2, cv2.LINE_AA)
            
            # Target count & filter status
            target_info = f"Filter: {','.join([all_names[i] for i in target_ids])}" if target_ids else "Filter: ALL"
            cv2.putText(frame, f"{target_info} | Detected: {detected_count}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

            # Write frame if saving
            if writer:
                writer.write(frame)

        # Display window
        if display:
            cv2.imshow("Real-Time Video Segmentation (YOLO11-seg)", frame)
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key in [ord("q"), 27]:  # q or Esc
                print("\n[INFO] Stopped by user.")
                break
            elif key in [ord("p"), 32]:  # p or Space
                paused = not paused
                print(f"[INFO] {'PAUSED' if paused else 'RESUMED'}")
            elif key == ord("s"):  # s for screenshot
                snap_path = f"screenshot_frame_{frame_idx:05d}.jpg"
                cv2.imwrite(snap_path, frame)
                print(f"[INFO] Screenshot saved to: {snap_path}")
        else:
            # Print periodic progress in headless mode
            if frame_idx % 50 == 0 or frame_idx == total_frames:
                print(f"  Frame {frame_idx}/{total_frames} | Processing FPS: {avg_fps:.1f} | Detections: {detected_count}")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
        print(f"\n[INFO] Annotated video saved successfully to: {output_path}")
    if display:
        cv2.destroyAllWindows()

    total_time = time.time() - start_total_time
    print(f"\nTotal frames processed: {frame_idx}")
    print(f"Total time:             {total_time:.2f}s")
    if frame_idx > 0:
        print(f"Average throughput:     {frame_idx / total_time:.1f} FPS")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Video Object Detection and Instance Segmentation using YOLO11-seg"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="samples/pedestrians.avi",
        help="Path to input video file (default: samples/pedestrians.avi)"
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
        help="Comma-separated target classes to detect/segment (e.g. 'person', 'car,bus', or 'all')"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold between 0.0 and 1.0 (default: 0.35)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to export the annotated video (e.g. output.mp4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Execution device (defaults to auto-detecting MPS on Apple Silicon, CUDA, or CPU)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying GUI window (useful for headless servers / batch processing)"
    )

    args = parser.parse_args()

    targets = [t.strip() for t in args.target.split(",")] if args.target else None

    process_video(
        source_path=args.source,
        model_name=args.model,
        target_classes=targets,
        conf_threshold=args.conf,
        output_path=args.output,
        device=args.device,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
