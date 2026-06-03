"""
Complete Inference Pipeline: Tracking + Occlusion Analysis + Line Crossing.

Runs all three tasks in a single pass for efficiency.

Usage:
    python inference_pipeline.py --source VIDEO [--model MODEL] [--line-y 0.5]
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


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Complete Inference Pipeline")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--line-y", type=float, default=0.5)
    parser.add_argument("--line-direction", type=str, default="horizontal")
    parser.add_argument("--line-x", type=float, default=0.5)
    parser.add_argument("--line-coords", type=int, nargs=4, default=None,
                        help="Virtual line coordinates as x1 y1 x2 y2. Overrides line-direction/y/x if set.")
    parser.add_argument("--occlusion-frames", type=int, default=4,
                        help="Number of frames to extract for occlusion analysis")
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
    occlusion_dir = output_dir / "occlusion_frames"
    occlusion_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Cannot open: {args.source}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")
    print(f"Model: {args.model}")

    # Line setup
    if args.line_coords:
        x1, y1, x2, y2 = args.line_coords
        line_p1 = (x1, y1)
        line_p2 = (x2, y2)
        line_pos = None
        print(f"Line: custom coordinates {line_p1} -> {line_p2}")
    else:
        if args.line_direction == "horizontal":
            line_pos = int(h * args.line_y)
            line_p1, line_p2 = (0, line_pos), (w, line_pos)
        else:
            line_pos = int(w * args.line_x)
            line_p1, line_p2 = (line_pos, 0), (line_pos, h)
        print(f"Line: {args.line_direction} at {'y' if args.line_direction == 'horizontal' else 'x'}={line_pos}")

    # Video writers
    track_writer = cv2.VideoWriter(
        str(output_dir / "tracking_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    count_writer = cv2.VideoWriter(
        str(output_dir / "counting_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # State
    track_history = defaultdict(list)
    prev_centers = {}
    cross_down = 0
    cross_up = 0
    crossed_ids = set()
    crossing_events = []
    all_tracking = []
    frame_occlusion_scores = {}

    colors = [
        (255,0,0), (0,255,0), (0,0,255), (255,255,0),
        (255,0,255), (0,255,255), (128,0,0), (0,128,0),
        (0,0,128), (128,128,0), (128,0,128), (0,128,128),
        (255,128,0), (255,0,128), (128,255,0), (0,255,128),
    ]

    frame_idx = 0
    print("\nProcessing...")

    # === PASS 1: Run tracking and collect all data ===
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, tracker=args.tracker,
                              conf=args.conf, iou=args.iou, verbose=False)
        result = results[0]

        # Annotated frames
        track_frame = frame.copy()
        count_frame = frame.copy()

        current_centers = {}
        frame_boxes = []

        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
                conf_val = float(result.boxes.conf[i].cpu())
                cls_id = int(result.boxes.cls[i].cpu())
                cls_name = model.names[cls_id]
                tid = int(result.boxes.id[i].cpu()) if result.boxes.id is not None else -1

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                current_centers[tid] = (cx, cy, cls_name)
                frame_boxes.append([x1, y1, x2, y2])

                all_tracking.append({
                    "frame": frame_idx, "track_id": tid,
                    "class": cls_name, "conf": conf_val,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [cx, cy],
                })

                color = colors[tid % len(colors)] if tid >= 0 else (200, 200, 200)

                # --- Tracking frame ---
                cv2.rectangle(track_frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {cls_name} {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(track_frame, (x1, y1-th-6), (x1+tw, y1), color, -1)
                cv2.putText(track_frame, label, (x1, y1-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                if tid >= 0:
                    track_history[tid].append((int(cx), int(cy)))
                    if len(track_history[tid]) > 50:
                        track_history[tid].pop(0)
                    if len(track_history[tid]) > 1:
                        pts = np.array(track_history[tid], dtype=np.int32).reshape((-1,1,2))
                        cv2.polylines(track_frame, [pts], False, color, 2)

                # --- Counting frame ---
                cv2.rectangle(count_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(count_frame, f"ID:{tid} {cls_name}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.circle(count_frame, (int(cx), int(cy)), 4, color, -1)

                # Line crossing check
                if tid >= 0 and tid in prev_centers:
                    pcx, pcy, _ = prev_centers[tid]
                    if args.line_coords:
                        p_prev = (pcx, pcy)
                        p_curr = (cx, cy)
                        if intersect(p_prev, p_curr, line_p1, line_p2):
                            if tid not in crossed_ids:
                                # Determine direction relative to line vector
                                prev_side = (pcx - line_p1[0]) * (line_p2[1] - line_p1[1]) - (pcy - line_p1[1]) * (line_p2[0] - line_p1[0])
                                crossed_ids.add(tid)
                                if prev_side > 0:
                                    cross_down += 1
                                    crossing_events.append({"frame": frame_idx, "track_id": tid,
                                                             "class": cls_name, "direction": "down_or_right"})
                                else:
                                    cross_up += 1
                                    crossing_events.append({"frame": frame_idx, "track_id": tid,
                                                             "class": cls_name, "direction": "up_or_left"})
                    else:
                        if args.line_direction == "horizontal":
                            if pcy < line_pos <= cy and tid not in crossed_ids:
                                cross_down += 1
                                crossed_ids.add(tid)
                                crossing_events.append({"frame": frame_idx, "track_id": tid,
                                                         "class": cls_name, "direction": "down"})
                            elif pcy >= line_pos > cy and tid not in crossed_ids:
                                cross_up += 1
                                crossed_ids.add(tid)
                                crossing_events.append({"frame": frame_idx, "track_id": tid,
                                                         "class": cls_name, "direction": "up"})
                        else:
                            if pcx < line_pos <= cx and tid not in crossed_ids:
                                cross_down += 1
                                crossed_ids.add(tid)
                                crossing_events.append({"frame": frame_idx, "track_id": tid,
                                                         "class": cls_name, "direction": "right"})
                            elif pcx >= line_pos > cx and tid not in crossed_ids:
                                cross_up += 1
                                crossed_ids.add(tid)
                                crossing_events.append({"frame": frame_idx, "track_id": tid,
                                                         "class": cls_name, "direction": "left"})

        prev_centers = current_centers

        # Compute occlusion score for this frame
        overlap_count = 0
        for i in range(len(frame_boxes)):
            for j in range(i+1, len(frame_boxes)):
                if compute_iou(frame_boxes[i], frame_boxes[j]) > 0.15:
                    overlap_count += 1
        frame_occlusion_scores[frame_idx] = {
            "overlaps": overlap_count, "num_objects": len(frame_boxes),
            "score": overlap_count * 10 + len(frame_boxes) * 2,
        }

        # Frame info on tracking frame
        cv2.putText(track_frame, f"Frame: {frame_idx}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        track_writer.write(track_frame)

        # Counting frame decorations
        cv2.line(count_frame, line_p1, line_p2, (0, 0, 255), 3)
        total_count = cross_down + cross_up
        cv2.rectangle(count_frame, (10, h-110), (340, h-10), (0,0,0), -1)
        cv2.rectangle(count_frame, (10, h-110), (340, h-10), (0,255,0), 2)
        
        if args.line_coords:
            d_label, u_label = "Down/Right", "Up/Left"
        else:
            d_label = "Down" if args.line_direction == "horizontal" else "Right"
            u_label = "Up" if args.line_direction == "horizontal" else "Left"
            
        cv2.putText(count_frame, f"{d_label}: {cross_down}", (20, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(count_frame, f"{u_label}: {cross_up}", (20, h-55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(count_frame, f"TOTAL: {total_count}", (20, h-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(count_frame, f"Frame: {frame_idx}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        count_writer.write(count_frame)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total_frames} | Crossings: {total_count}")

        frame_idx += 1

    cap.release()
    track_writer.release()
    count_writer.release()

    # === PASS 2: Extract occlusion analysis frames ===
    print("\nExtracting occlusion analysis frames...")
    if frame_occlusion_scores:
        sorted_scores = sorted(frame_occlusion_scores.items(),
                               key=lambda x: x[1]["score"], reverse=True)
        best_frame = sorted_scores[0][0]
        analysis_frames = [best_frame + i for i in range(-1, args.occlusion_frames - 1)]
        analysis_frames = [f for f in analysis_frames if 0 <= f < frame_idx]
    else:
        mid = frame_idx // 2
        analysis_frames = list(range(mid - 1, mid + args.occlusion_frames - 1))

    # Re-run tracking for occlusion frames
    print(f"  Best occlusion candidate: frame {analysis_frames[0] if analysis_frames else 'N/A'}")
    model_fresh = YOLO(args.model)
    cap2 = cv2.VideoCapture(args.source)
    fidx = 0
    saved_frames = []
    target_set = set(analysis_frames)
    max_target = max(analysis_frames) + 1 if analysis_frames else 0

    while cap2.isOpened() and fidx <= max_target:
        ok, fr = cap2.read()
        if not ok:
            break
        res = model_fresh.track(fr, persist=True, tracker=args.tracker, conf=args.conf, verbose=False)
        if fidx in target_set:
            r = res[0]
            ann = fr.copy()
            if r.boxes is not None and len(r.boxes) > 0:
                for i in range(len(r.boxes)):
                    bx = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    tid = int(r.boxes.id[i].cpu()) if r.boxes.id is not None else -1
                    cname = model_fresh.names[int(r.boxes.cls[i].cpu())]
                    np.random.seed(tid * 42 if tid >= 0 else 0)
                    clr = tuple(np.random.randint(50, 255, 3).tolist())
                    cv2.rectangle(ann, (bx[0], bx[1]), (bx[2], bx[3]), clr, 3)
                    lbl = f"ID:{tid} {cname}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(ann, (bx[0], bx[1]-th-10), (bx[0]+tw+4, bx[1]), clr, -1)
                    cv2.putText(ann, lbl, (bx[0]+2, bx[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(ann, f"Frame #{fidx}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            fp = str(occlusion_dir / f"frame_{fidx:06d}.jpg")
            cv2.imwrite(fp, ann)
            saved_frames.append(fp)
        fidx += 1
    cap2.release()

    # Create comparison figure
    if saved_frames:
        imgs = [cv2.imread(p) for p in saved_frames]
        th = 480
        resized = [cv2.resize(im, (int(im.shape[1]*th/im.shape[0]), th)) for im in imgs]
        if len(resized) <= 2:
            combined = np.hstack(resized)
        else:
            r1 = np.hstack(resized[:2])
            r2l = resized[2:]
            while len(r2l) < 2:
                r2l.append(np.zeros_like(resized[0]))
            r2 = np.hstack(r2l)
            mw = max(r1.shape[1], r2.shape[1])
            if r1.shape[1] < mw:
                r1 = np.pad(r1, ((0,0),(0,mw-r1.shape[1]),(0,0)))
            if r2.shape[1] < mw:
                r2 = np.pad(r2, ((0,0),(0,mw-r2.shape[1]),(0,0)))
            combined = np.vstack([r1, r2])
        cv2.imwrite(str(output_dir / "occlusion_comparison.jpg"), combined)

    # === Save all results ===
    # Tracking data
    with open(str(output_dir / "tracking_data.json"), "w") as f:
        json.dump(all_tracking, f, indent=2)

    # Track lifecycles
    lifecycles = defaultdict(list)
    for d in all_tracking:
        if d["track_id"] >= 0:
            lifecycles[d["track_id"]].append(d["frame"])

    # ID jump analysis
    id_jumps = []
    for tid1, fr1 in lifecycles.items():
        if not fr1:
            continue
        end = max(fr1)
        lpos = None
        lcls = None
        for d in all_tracking:
            if d["track_id"] == tid1 and d["frame"] == end:
                lpos = d["center"]
                lcls = d["class"]
                break
        if lpos is None:
            continue
        for tid2, fr2 in lifecycles.items():
            if tid2 == tid1 or not fr2:
                continue
            start = min(fr2)
            if 0 < start - end <= 10:
                fpos = None
                for d in all_tracking:
                    if d["track_id"] == tid2 and d["frame"] == start:
                        fpos = d["center"]
                        fcls = d["class"]
                        break
                if fpos and lcls == fcls:
                    dist = np.sqrt((lpos[0]-fpos[0])**2 + (lpos[1]-fpos[1])**2)
                    if dist < 100:
                        id_jumps.append({
                            "old_id": tid1, "new_id": tid2,
                            "end_frame": end, "start_frame": start,
                            "distance": float(dist), "class": lcls,
                        })

    # Counting results
    counting = {
        "total_crossings": cross_down + cross_up,
        "crossings_down_or_right": cross_down,
        "crossings_up_or_left": cross_up,
        "line_position": line_pos,
        "line_direction": args.line_direction,
        "crossing_events": crossing_events,
    }
    with open(str(output_dir / "counting_results.json"), "w") as f:
        json.dump(counting, f, indent=2)

    # Full report
    report = {
        "video": args.source,
        "model": args.model,
        "total_frames": frame_idx,
        "total_tracks": len(lifecycles),
        "short_tracks": sum(1 for v in lifecycles.values() if len(v) < 10),
        "id_jumps": id_jumps,
        "counting": counting,
        "track_lifecycles": {
            str(k): {"start": min(v), "end": max(v),
                      "duration": max(v)-min(v)+1, "detections": len(v)}
            for k, v in lifecycles.items() if v
        },
        "analysis_frames": analysis_frames,
    }
    with open(str(output_dir / "full_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Frames processed: {frame_idx}")
    print(f"Unique tracks: {len(lifecycles)}")
    print(f"ID jumps: {len(id_jumps)}")
    for j in id_jumps:
        print(f"  ID {j['old_id']}->{j['new_id']} ({j['class']}, dist={j['distance']:.1f})")
    print(f"Line crossings: {counting['total_crossings']}")
    print(f"  Down/Right: {cross_down} | Up/Left: {cross_up}")
    print(f"\nOutputs:")
    print(f"  Tracking video:  {output_dir / 'tracking_output.mp4'}")
    print(f"  Counting video:  {output_dir / 'counting_output.mp4'}")
    print(f"  Occlusion frames: {occlusion_dir}")
    print(f"  Full report:     {output_dir / 'full_report.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
