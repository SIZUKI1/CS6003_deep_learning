"""
Video Multi-Object Tracking with Fine-tuned YOLOv8.

This script performs:
1. Frame-by-frame object detection using fine-tuned YOLOv8
2. Multi-object tracking with BoT-SORT / ByteTrack
3. Tracking trajectory visualization
4. Output annotated video with bounding boxes, class labels, and tracking IDs

Usage:
    python track_video.py --source VIDEO_PATH [--model MODEL_PATH] [--tracker bytetrack.yaml]
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Multi-Object Tracking with YOLOv8")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (default: auto-discover best.pt)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        help="Tracker config: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--show", action="store_true",
                        help="Display results in window")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save individual annotated frames")
    args = parser.parse_args()

    from ultralytics import YOLO

    # ── Auto-discover model ──
    base_dir = Path(__file__).parent
    if args.model is None:
        model_path = base_dir / "runs" / "detect" / "yolov8s_road_vehicle" / "weights" / "best.pt"
        if model_path.exists():
            args.model = str(model_path)
        else:
            print("❌ No trained model found. Please train first or specify --model")
            sys.exit(1)

    print(f"📹 Source: {args.source}")
    print(f"🤖 Model: {args.model}")
    print(f"🔍 Tracker: {args.tracker}")
    print(f"🎯 Confidence: {args.conf}")

    # ── Load model ──
    model = YOLO(args.model)

    # ── Setup output ──
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Open video ──
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.source}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📐 Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Output video writer
    output_path = str(output_dir / "tracking_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ── Track history for trajectory visualization ──
    track_history = defaultdict(lambda: [])
    # Color palette for different track IDs
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
        (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    ]

    # ── Frame-by-frame processing ──
    frame_idx = 0
    all_track_data = []  # Store tracking data for analysis

    print("\n🚀 Starting tracking...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run tracking
        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        result = results[0]
        annotated_frame = frame.copy()

        # Process results
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            for i in range(len(boxes)):
                # Get box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = model.names[cls_id] if cls_id in model.names else f"cls_{cls_id}"

                # Get track ID (may not exist for untracked detections)
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu())
                else:
                    track_id = -1

                # Store tracking data
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                all_track_data.append({
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class": cls_name,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                })

                # Draw bounding box
                color = colors[track_id % len(colors)] if track_id >= 0 else (200, 200, 200)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"ID:{track_id} {cls_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_frame,
                              (x1, y1 - label_size[1] - 8),
                              (x1 + label_size[0], y1),
                              color, -1)
                cv2.putText(annotated_frame, label,
                            (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)

                # Update track history for trajectory
                if track_id >= 0:
                    track = track_history[track_id]
                    track.append((int(cx), int(cy)))
                    if len(track) > 50:
                        track.pop(0)

                    # Draw trajectory
                    if len(track) > 1:
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False,
                                      color=color, thickness=2)

        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # Write output
        writer.write(annotated_frame)

        # Save individual frames
        if args.save_frames:
            frame_dir = output_dir / "frames"
            frame_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}.jpg"), annotated_frame)

        # Progress
        if frame_idx % 30 == 0:
            print(f"  Processing frame {frame_idx}/{total_frames}...")

        frame_idx += 1

    cap.release()
    writer.release()

    # ── Save tracking data as JSON ──
    import json
    track_data_path = str(output_dir / "tracking_data.json")
    with open(track_data_path, "w") as f:
        json.dump(all_track_data, f, indent=2)

    # ── Summary ──
    unique_tracks = set(d["track_id"] for d in all_track_data if d["track_id"] >= 0)
    unique_classes = set(d["class"] for d in all_track_data)

    print(f"\n{'=' * 60}")
    print(f"✅ Tracking completed!")
    print(f"   Frames processed: {frame_idx}")
    print(f"   Unique tracks: {len(unique_tracks)}")
    print(f"   Classes detected: {unique_classes}")
    print(f"   Output video: {output_path}")
    print(f"   Tracking data: {track_data_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
