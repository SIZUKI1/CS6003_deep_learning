"""
Occlusion and ID Jump Analysis for Multi-Object Tracking.

Analyzes tracking results to identify ID jumps, occlusion events,
and extracts consecutive frames for visual report.

Usage:
    python occlusion_analysis.py --source VIDEO --model MODEL [--tracking-data JSON]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def find_occlusion_frames(tracking_data):
    """Find frames where occlusion or dense intersection occurs."""
    frames_data = defaultdict(list)
    for d in tracking_data:
        frames_data[d["frame"]].append(d)

    track_lifecycles = defaultdict(list)
    for d in tracking_data:
        if d["track_id"] >= 0:
            track_lifecycles[d["track_id"]].append(d["frame"])

    short_tracks = {
        tid: frames for tid, frames in track_lifecycles.items()
        if len(frames) < 10
    }

    frame_scores = {}
    for frame_idx, detections in frames_data.items():
        if len(detections) < 2:
            continue
        score = 0
        boxes = [d["bbox"] for d in detections]
        overlap_count = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = compute_iou(boxes[i], boxes[j])
                if iou > 0.15:
                    overlap_count += 1
                    score += iou * 10
        if len(detections) >= 3:
            score += len(detections) * 2
        for d in detections:
            if d["track_id"] in short_tracks:
                score += 5
        frame_scores[frame_idx] = {
            "score": score, "num_objects": len(detections),
            "overlaps": overlap_count,
        }
    return frame_scores, track_lifecycles, short_tracks


def analyze_id_jumps(tracking_data, track_lifecycles):
    """Detect potential ID jumps."""
    id_jumps = []
    for tid1, frames1 in track_lifecycles.items():
        if not frames1:
            continue
        end_frame = max(frames1)
        last_pos, last_class = None, None
        for d in tracking_data:
            if d["track_id"] == tid1 and d["frame"] == end_frame:
                last_pos = d["center"]
                last_class = d["class"]
                break
        if last_pos is None:
            continue
        for tid2, frames2 in track_lifecycles.items():
            if tid2 == tid1 or not frames2:
                continue
            start_frame = min(frames2)
            if 0 < start_frame - end_frame <= 10:
                first_pos = None
                for d in tracking_data:
                    if d["track_id"] == tid2 and d["frame"] == start_frame:
                        first_pos = d["center"]
                        first_class = d["class"]
                        break
                if first_pos is None:
                    continue
                dist = np.sqrt((last_pos[0]-first_pos[0])**2 + (last_pos[1]-first_pos[1])**2)
                if dist < 100 and last_class == first_class:
                    id_jumps.append({
                        "old_id": tid1, "new_id": tid2,
                        "end_frame": end_frame, "start_frame": start_frame,
                        "distance": float(dist), "class": last_class,
                    })
    return id_jumps


def extract_frames(video_path, model, frame_indices, output_dir, tracker, conf):
    """Re-run tracking and extract annotated frames."""
    from ultralytics import YOLO
    cap = cv2.VideoCapture(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    saved = []
    target_set = set(frame_indices)
    max_frame = max(frame_indices) + 1

    while cap.isOpened() and frame_idx <= max_frame:
        success, frame = cap.read()
        if not success:
            break
        results = model.track(frame, persist=True, tracker=tracker, conf=conf, verbose=False)
        if frame_idx in target_set:
            result = results[0]
            ann = frame.copy()
            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    cv_conf = float(result.boxes.conf[i].cpu())
                    cls_id = int(result.boxes.cls[i].cpu())
                    cls_name = model.names[cls_id]
                    tid = int(result.boxes.id[i].cpu()) if result.boxes.id is not None else -1
                    np.random.seed(tid * 42 if tid >= 0 else 0)
                    color = tuple(np.random.randint(50, 255, 3).tolist())
                    cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)
                    label = f"ID:{tid} {cls_name}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(ann, (x1, y1-th-10), (x1+tw+4, y1), color, -1)
                    cv2.putText(ann, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(ann, f"Frame #{frame_idx}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            out_path = str(output_dir / f"occlusion_frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, ann)
            saved.append(out_path)
            print(f"  Saved: {out_path}")
        frame_idx += 1
    cap.release()
    return saved


def create_comparison(frame_paths, output_path):
    """Create comparison figure from frames."""
    frames = [cv2.imread(p) for p in frame_paths if os.path.exists(p)]
    if not frames:
        return
    target_h = 480
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        scale = target_h / h
        resized.append(cv2.resize(f, (int(w * scale), target_h)))
    if len(resized) <= 2:
        combined = np.hstack(resized)
    else:
        row1 = np.hstack(resized[:2])
        row2_f = resized[2:]
        while len(row2_f) < 2:
            row2_f.append(np.zeros_like(resized[0]))
        row2 = np.hstack(row2_f)
        max_w = max(row1.shape[1], row2.shape[1])
        if row1.shape[1] < max_w:
            row1 = np.pad(row1, ((0,0),(0,max_w-row1.shape[1]),(0,0)))
        if row2.shape[1] < max_w:
            row2 = np.pad(row2, ((0,0),(0,max_w-row2.shape[1]),(0,0)))
        combined = np.vstack([row1, row2])
    cv2.imwrite(output_path, combined)
    print(f"Comparison saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="outputs/occlusion_analysis")
    parser.add_argument("--tracking-data", type=str, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model is None:
        mp = base_dir / "runs/detect/yolov8s_road_vehicle/weights/best.pt"
        if mp.exists():
            args.model = str(mp)
        else:
            print("No trained model found!")
            sys.exit(1)

    model = YOLO(args.model)

    # Load or generate tracking data
    if args.tracking_data and os.path.exists(args.tracking_data):
        with open(args.tracking_data) as f:
            tracking_data = json.load(f)
    else:
        print("Running tracking...")
        cap = cv2.VideoCapture(args.source)
        tracking_data = []
        idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            results = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf, verbose=False)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for i in range(len(r.boxes)):
                    x1,y1,x2,y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    tid = int(r.boxes.id[i].cpu()) if r.boxes.id is not None else -1
                    tracking_data.append({
                        "frame": idx, "track_id": tid,
                        "class": model.names[int(r.boxes.cls[i].cpu())],
                        "conf": float(r.boxes.conf[i].cpu()),
                        "bbox": [int(x1),int(y1),int(x2),int(y2)],
                        "center": [(x1+x2)/2, (y1+y2)/2],
                    })
            idx += 1
        cap.release()
        with open(str(output_dir / "tracking_data.json"), "w") as f:
            json.dump(tracking_data, f, indent=2)

    # Analyze
    frame_scores, lifecycles, short_tracks = find_occlusion_frames(tracking_data)
    id_jumps = analyze_id_jumps(tracking_data, lifecycles)

    print(f"\n{'='*60}")
    print(f"Total tracks: {len(lifecycles)}")
    print(f"Short-lived tracks (<10 frames): {len(short_tracks)}")
    print(f"ID jumps detected: {len(id_jumps)}")
    for j in id_jumps:
        print(f"  ID {j['old_id']} -> {j['new_id']} ({j['class']}, "
              f"frame {j['end_frame']}->{j['start_frame']}, dist={j['distance']:.1f}px)")

    print(f"\nTrack lifecycles:")
    for tid in sorted(lifecycles.keys()):
        fr = lifecycles[tid]
        print(f"  ID {tid:3d}: frames {min(fr):4d}-{max(fr):4d} ({len(fr)} det)")

    # Find best frames for analysis
    if frame_scores:
        sorted_f = sorted(frame_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top = sorted_f[0][0]
        analysis_frames = [top + i for i in range(-1, 3)]
        analysis_frames = [f for f in analysis_frames if f >= 0]
    else:
        total = max(d["frame"] for d in tracking_data) if tracking_data else 0
        mid = total // 2
        analysis_frames = [mid-1, mid, mid+1, mid+2]

    print(f"\nExtracting frames: {analysis_frames}")
    model_fresh = YOLO(args.model)
    saved = extract_frames(args.source, model_fresh, analysis_frames, output_dir, args.tracker, args.conf)
    if saved:
        create_comparison(saved, str(output_dir / "occlusion_comparison.jpg"))

    # Save report
    report = {
        "total_tracks": len(lifecycles),
        "short_tracks": len(short_tracks),
        "id_jumps": id_jumps,
        "track_lifecycles": {
            str(k): {"start": min(v), "end": max(v), "duration": max(v)-min(v)+1, "detections": len(v)}
            for k, v in lifecycles.items() if v
        },
        "analysis_frames": analysis_frames,
    }
    with open(str(output_dir / "analysis_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_dir}")


if __name__ == "__main__":
    main()
