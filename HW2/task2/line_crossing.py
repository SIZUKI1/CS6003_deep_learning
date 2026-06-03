"""
Line-Crossing Counter for Multi-Object Tracking.

Sets a virtual line in the video and counts objects that cross it,
using detection box center coordinates and tracking ID continuity.

Usage:
    python line_crossing.py --source VIDEO --model MODEL [--line-y 0.5]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def main():
    parser = argparse.ArgumentParser(description="Line-Crossing Counter")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--line-y", type=float, default=0.5,
                        help="Virtual line Y position as fraction of frame height (0-1)")
    parser.add_argument("--line-direction", type=str, default="horizontal",
                        choices=["horizontal", "vertical"],
                        help="Line direction")
    parser.add_argument("--line-x", type=float, default=0.5,
                        help="Virtual line X position for vertical line (0-1)")
    parser.add_argument("--line-coords", type=int, nargs=4, default=None,
                        help="Virtual line coordinates as x1 y1 x2 y2. Overrides line-direction/y/x if set.")
    args = parser.parse_args()

    from ultralytics import YOLO

    base_dir = Path(__file__).parent
    if args.model is None:
        mp = base_dir / "runs/detect/yolov8s_road_vehicle/weights/best.pt"
        if mp.exists():
            args.model = str(mp)
        else:
            print("No trained model found!")
            sys.exit(1)

    model = YOLO(args.model)
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Cannot open: {args.source}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define virtual line
    if args.line_coords:
        x1, y1, x2, y2 = args.line_coords
        line_p1 = (x1, y1)
        line_p2 = (x2, y2)
        line_pos = None # not a single coordinate
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Line: custom coordinates {line_p1} -> {line_p2}")
    else:
        if args.line_direction == "horizontal":
            line_pos = int(height * args.line_y)
            line_p1 = (0, line_pos)
            line_p2 = (width, line_pos)
        else:
            line_pos = int(width * args.line_x)
            line_p1 = (line_pos, 0)
            line_p2 = (line_pos, height)
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Line: {args.line_direction} at {'y' if args.line_direction == 'horizontal' else 'x'}={line_pos}")

    # Output video
    out_path = str(output_dir / "counting_output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Tracking state
    prev_centers = {}  # track_id -> previous center position
    cross_count_down = 0  # Objects crossing downward/rightward
    cross_count_up = 0    # Objects crossing upward/leftward
    crossed_ids = set()   # Track IDs that have already crossed
    crossing_events = []  # Log of crossing events

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]

    frame_idx = 0
    print("\nStarting counting...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(
            frame, persist=True, tracker=args.tracker,
            conf=args.conf, verbose=False,
        )
        result = results[0]
        annotated = frame.copy()

        current_centers = {}

        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
                cls_id = int(result.boxes.cls[i].cpu())
                cls_name = model.names[cls_id]
                conf_val = float(result.boxes.conf[i].cpu())
                tid = int(result.boxes.id[i].cpu()) if result.boxes.id is not None else -1

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                current_centers[tid] = (cx, cy, cls_name)

                # Draw box
                color = colors[tid % len(colors)] if tid >= 0 else (200, 200, 200)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {cls_name}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw center point
                cv2.circle(annotated, (int(cx), int(cy)), 4, color, -1)

                # Check line crossing
                if tid >= 0 and tid in prev_centers:
                    prev_cx, prev_cy, _ = prev_centers[tid]

                    if args.line_coords:
                        p_prev = (prev_cx, prev_cy)
                        p_curr = (cx, cy)
                        if intersect(p_prev, p_curr, line_p1, line_p2):
                            if tid not in crossed_ids:
                                # Determine direction relative to line vector
                                prev_side = (prev_cx - line_p1[0]) * (line_p2[1] - line_p1[1]) - (prev_cy - line_p1[1]) * (line_p2[0] - line_p1[0])
                                crossed_ids.add(tid)
                                if prev_side > 0:
                                    cross_count_down += 1
                                    crossing_events.append({
                                        "frame": frame_idx, "track_id": tid,
                                        "class": cls_name, "direction": "down_or_right",
                                    })
                                else:
                                    cross_count_up += 1
                                    crossing_events.append({
                                        "frame": frame_idx, "track_id": tid,
                                        "class": cls_name, "direction": "up_or_left",
                                    })
                    else:
                        if args.line_direction == "horizontal":
                            # Crossing from above to below
                            if prev_cy < line_pos and cy >= line_pos:
                                if tid not in crossed_ids:
                                    cross_count_down += 1
                                    crossed_ids.add(tid)
                                    crossing_events.append({
                                        "frame": frame_idx, "track_id": tid,
                                        "class": cls_name, "direction": "down",
                                    })
                            # Crossing from below to above
                            elif prev_cy >= line_pos and cy < line_pos:
                                if tid not in crossed_ids:
                                    cross_count_up += 1
                                    crossed_ids.add(tid)
                                    crossing_events.append({
                                        "frame": frame_idx, "track_id": tid,
                                        "class": cls_name, "direction": "up",
                                    })
                        else:
                            # Vertical line crossing
                            if prev_cx < line_pos and cx >= line_pos:
                                if tid not in crossed_ids:
                                    cross_count_down += 1
                                    crossed_ids.add(tid)
                                    crossing_events.append({
                                        "frame": frame_idx, "track_id": tid,
                                        "class": cls_name, "direction": "right",
                                    })
                            elif prev_cx >= line_pos and cx < line_pos:
                                if tid not in crossed_ids:
                                    cross_count_up += 1
                                    crossed_ids.add(tid)
                                    crossing_events.append({
                                        "frame": frame_idx, "track_id": tid,
                                        "class": cls_name, "direction": "left",
                                    })

        prev_centers = current_centers

        # Draw virtual line
        cv2.line(annotated, line_p1, line_p2, (0, 0, 255), 3)

        # Draw count display
        total_count = cross_count_down + cross_count_up
        # Background panel
        cv2.rectangle(annotated, (10, height - 120), (350, height - 10), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, height - 120), (350, height - 10), (0, 255, 0), 2)

        if args.line_coords:
            cv2.putText(annotated, f"Down/Right: {cross_count_down}",
                        (20, height - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, f"Up/Left: {cross_count_up}",
                        (20, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif args.line_direction == "horizontal":
            cv2.putText(annotated, f"Down: {cross_count_down}",
                        (20, height - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, f"Up: {cross_count_up}",
                        (20, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(annotated, f"Right: {cross_count_down}",
                        (20, height - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, f"Left: {cross_count_up}",
                        (20, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(annotated, f"TOTAL CROSSINGS: {total_count}",
                    (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Frame info
        cv2.putText(annotated, f"Frame: {frame_idx}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(annotated)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total_frames} | Crossings: {total_count}")

        frame_idx += 1

    cap.release()
    writer.release()

    # Save results
    result_data = {
        "total_crossings": cross_count_down + cross_count_up,
        "crossings_down_or_right": cross_count_down,
        "crossings_up_or_left": cross_count_up,
        "line_direction": args.line_direction,
        "line_position": line_pos,
        "crossing_events": crossing_events,
        "total_frames": frame_idx,
    }

    result_path = str(output_dir / "counting_results.json")
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COUNTING RESULTS")
    print(f"{'='*60}")
    print(f"Total crossings: {result_data['total_crossings']}")
    print(f"  Down/Right: {cross_count_down}")
    print(f"  Up/Left: {cross_count_up}")
    print(f"Output video: {out_path}")
    print(f"Results JSON: {result_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
