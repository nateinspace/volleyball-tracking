"""
VolleyVision Fine-Tuning Script
================================
Fine-tune the VolleyVision YOLOv8m action recognition model
with your own labeled Pepperdine volleyball practice data.

Pre-trained model classes (original): block, defense, serve, set, spike
Your pipeline classes:                block, defense, serve, set, attack, dig

Usage:
    # Step 1: Convert your JSON labels + extract frames from videos
    python finetune_volleyvision.py --mode convert \
        --json detections_labeled.json \
        --videos_dir /path/to/videos/

    # Step 2: Fine-tune
    python finetune_volleyvision.py --mode train \
        --data_yaml dataset/data.yaml --epochs 50

    # Step 3: Run inference on a video
    python finetune_volleyvision.py --mode predict \
        --model runs/finetune/weights/best.pt \
        --source /path/to/practice_video.mp4

    # Step 4: Run event detection with sliding window
    python finetune_volleyvision.py --mode events \
        --model runs/finetune/weights/best.pt \
        --source /path/to/practice_video.mp4
"""

import os
import sys
import shutil
import argparse
import yaml
import json
import random
from pathlib import Path

# ────────────────────────────────────────────────────────────
# 1. CLASS CONFIGURATION
# ────────────────────────────────────────────────────────────

# Final unified class list for training
# Index order matters — it defines the class IDs in YOLO labels
CLASSES = ["block", "serve", "set", "attack", "dig"]

# Map from VolleyVision original classes to ours
VOLLEYVISION_TO_OURS = {
    "block":   "block",
    "defense": "block",
    "serve":   "serve",
    "set":     "set",
    "spike":   "attack",   # renamed
}

# Map from YOUR JSON label names to unified classes
# "idle" and "ignore" are excluded (not trainable actions)
JSON_LABEL_TO_CLASS = {
    "block":  "block",
    "serve":  "serve",
    "set":    "set",
    "attack": "attack",
    "dig":    "dig",
    # Excluded:
    # "idle":   None,    <- no notable action, skip
    # "ignore": None,    <- not a player, skip
}

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


# ────────────────────────────────────────────────────────────
# 2. JSON -> YOLO CONVERTER
# ────────────────────────────────────────────────────────────

def convert_json_to_yolo(json_path: str, videos_dir: str = None,
                         frames_dir: str = None, output_dir: str = "dataset",
                         train_split: float = 0.80, val_split: float = 0.15):
    """
    Convert your detections_labeled.json into YOLO training format.

    This handles two scenarios:
      A) You have the original videos -> frames are extracted automatically
      B) You already extracted frames -> point --frames_dir to them

    Frame filenames are expected to match the "filename" field in the JSON
    (e.g., "v0_frame_000025.jpg").

    Args:
        json_path:   Path to detections_labeled.json
        videos_dir:  Directory containing the source .mp4 video files
        frames_dir:  Directory containing pre-extracted frame images
                     (if provided, videos_dir is ignored)
        output_dir:  Where to write the YOLO dataset
        train_split: Fraction of data for training
        val_split:   Fraction of data for validation (rest -> test)
    """
    output_dir = Path(output_dir)

    # -- Load JSON --
    with open(json_path) as f:
        data = json.load(f)

    videos = {v["video_id"]: v for v in data["videos"]}
    frames = data["frames"]
    print(f"Loaded {len(frames)} frames from {len(videos)} videos")
    print(f"Classes in JSON: {list(data['classes'].keys())}")

    # -- Extract frames from videos if needed --
    if frames_dir is None:
        if videos_dir is None:
            print("\nERROR: Provide either --videos_dir or --frames_dir")
            print("  --videos_dir : folder containing the .mp4 files")
            print("  --frames_dir : folder with pre-extracted frame jpgs")
            sys.exit(1)
        frames_dir = _extract_frames(data, videos_dir, output_dir / "_extracted_frames")

    frames_dir = Path(frames_dir)

    # -- Convert each frame's detections to YOLO format --
    converted = []  # list of (image_path, label_lines)
    label_counts = {c: 0 for c in CLASSES}
    skipped_idle = 0
    skipped_ignore = 0
    missing_frames = 0

    for frame in frames:
        img_path = frames_dir / frame["filename"]
        if not img_path.exists():
            missing_frames += 1
            continue

        img_w = frame["width"]
        img_h = frame["height"]

        yolo_lines = []
        for det in frame["detections"]:
            label = det["label"]

            # Skip non-action labels
            if label == "idle":
                skipped_idle += 1
                continue
            if label == "ignore":
                skipped_ignore += 1
                continue

            # Map to unified class
            unified = JSON_LABEL_TO_CLASS.get(label)
            if unified is None:
                continue

            class_id = CLASS_TO_ID[unified]

            # Convert absolute pixel bbox (x1,y1,x2,y2) -> YOLO (cx,cy,w,h) normalized
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            label_counts[unified] += 1

        # Only include frames that have at least one action detection
        if yolo_lines:
            converted.append((img_path, yolo_lines))

    print(f"\n-- Conversion Summary --")
    print(f"  Frames with actions: {len(converted)} / {len(frames)}")
    print(f"  Missing frame files: {missing_frames}")
    print(f"  Skipped 'idle':      {skipped_idle}")
    print(f"  Skipped 'ignore':    {skipped_ignore}")
    print(f"\n  Action label counts:")
    for cls, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {cls:12s}: {count}")

    # -- Split into train / val / test --
    random.seed(42)

    # Group by video_id to avoid data leakage between splits
    video_groups = {}
    frame_to_video = {f["filename"]: f["video_id"] for f in frames}
    for img_path, lines in converted:
        vid = frame_to_video.get(img_path.name, "unknown")
        video_groups.setdefault(vid, []).append((img_path, lines))

    video_ids = sorted(video_groups.keys())
    random.shuffle(video_ids)

    n_vids = len(video_ids)
    n_train = max(1, int(n_vids * train_split))
    n_val = max(1, int(n_vids * val_split))

    split_assignment = {}
    for i, vid in enumerate(video_ids):
        if i < n_train:
            split_assignment[vid] = "train"
        elif i < n_train + n_val:
            split_assignment[vid] = "val"
        else:
            split_assignment[vid] = "test"

    # If only a few videos, ensure at least train and val have data
    if n_vids <= 2:
        # Fall back to frame-level random split
        print(f"\n  NOTE: Only {n_vids} video(s) -- using frame-level split instead")
        random.shuffle(converted)
        n = len(converted)
        nt = int(n * train_split)
        nv = int(n * val_split)
        splits = {
            "train": converted[:nt],
            "val":   converted[nt:nt+nv],
            "test":  converted[nt+nv:],
        }
    else:
        splits = {"train": [], "val": [], "test": []}
        for vid, items in video_groups.items():
            splits[split_assignment[vid]].extend(items)

    # -- Write to disk --
    for split_name, items in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, yolo_lines in items:
            shutil.copy2(img_path, img_out / img_path.name)
            lbl_file = lbl_out / (img_path.stem + ".txt")
            lbl_file.write_text("\n".join(yolo_lines) + "\n")

        print(f"  {split_name:6s}: {len(items)} frames")

    # -- Write data.yaml --
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(CLASSES)},
        "nc": len(CLASSES),
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Dataset ready at: {output_dir}")
    print(f"   Config: {yaml_path}")
    print(f"   Classes: {CLASSES}")
    return str(yaml_path)


def _extract_frames(data: dict, videos_dir: str, out_dir: Path) -> Path:
    """Extract the specific frames referenced in the JSON from video files."""
    import cv2

    videos_dir = Path(videos_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group frames by video
    frames_by_video = {}
    for frame in data["frames"]:
        vid = frame["video_id"]
        frames_by_video.setdefault(vid, []).append(frame)

    videos_info = {v["video_id"]: v for v in data["videos"]}

    for vid_id, vid_frames in frames_by_video.items():
        vid_info = videos_info[vid_id]
        video_name = vid_info["filename"]
        video_path = videos_dir / video_name

        if not video_path.exists():
            print(f"  WARNING: Video not found: {video_path}")
            # Try without directory structure
            candidates = list(videos_dir.rglob(video_name))
            if candidates:
                video_path = candidates[0]
                print(f"           Found at: {video_path}")
            else:
                print(f"           Skipping {len(vid_frames)} frames")
                continue

        cap = cv2.VideoCapture(str(video_path))
        frame_indices = {f["frame_index"]: f["filename"] for f in vid_frames}

        extracted = 0
        frame_idx = 0
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            if frame_idx in frame_indices:
                out_path = out_dir / frame_indices[frame_idx]
                cv2.imwrite(str(out_path), img)
                extracted += 1
            frame_idx += 1

        cap.release()
        print(f"  Video {vid_id} ({video_name}): extracted {extracted}/{len(vid_frames)} frames")

    return out_dir


# ────────────────────────────────────────────────────────────
# 3. FINE-TUNING
# ────────────────────────────────────────────────────────────

def finetune(data_yaml: str, epochs: int = 50, imgsz: int = 1280,
             batch: int = 8, freeze_layers: int = 0,
             pretrained_weights: str = None):
    """
    Fine-tune starting from the VolleyVision pre-trained weights.

    Since we changed the number of classes (5 -> 6, added "dig" and
    renamed "spike" -> "attack"), the classification head gets
    re-initialized while backbone weights transfer over.

    Args:
        data_yaml:          Path to your data.yaml
        epochs:             Training epochs (50-100 recommended)
        imgsz:              Image size (1280 recommended -- volleyball
                            details get lost at 640)
        batch:              Batch size (lower if GPU memory is tight)
        freeze_layers:      Backbone layers to freeze (0 = train all,
                            10 = freeze early layers for speed)
        pretrained_weights: Path to starting weights
    """
    from ultralytics import YOLO

    # Default: VolleyVision pre-trained weights
    if pretrained_weights is None:
        pretrained_weights = _find_volleyvision_weights()

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    num_classes = data_cfg.get("nc", len(data_cfg["names"]))

    print(f"Target classes ({num_classes}): {list(data_cfg['names'].values())}")
    print(f"Pre-trained weights: {pretrained_weights}")

    # Class count changed (5 -> 6), so we start from base yolov8m
    # and transfer backbone weights from VolleyVision
    if pretrained_weights and Path(pretrained_weights).exists():
        # Check if class count matches
        probe = YOLO(pretrained_weights)
        old_nc = len(probe.names)
        if old_nc != num_classes:
            print(f"\n  Class count changed: {old_nc} -> {num_classes}")
            print("   Starting from yolov8m base with transferred backbone.")
            model = YOLO("yolov8m.pt")
        else:
            model = YOLO(pretrained_weights)
    else:
        print("No pre-trained volleyball weights found, using yolov8m.pt")
        model = YOLO("yolov8m.pt")

    device = "0,1,2,3,4,5,6,7" if _gpu_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: epochs={epochs}, imgsz={imgsz}, batch={batch}\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        save=True,
        save_period=10,
        device=device,
        project="runs",
        name="finetune",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        freeze=freeze_layers,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        verbose=True,
    )

    best_weights = Path("runs/finetune/weights/best.pt")
    print(f"\n  Training complete!")
    print(f"   Best weights: {best_weights}")
    print(f"   Results:      runs/finetune/results.png")
    return str(best_weights)


# ────────────────────────────────────────────────────────────
# 4. INFERENCE & EVENT DETECTION
# ────────────────────────────────────────────────────────────

def predict(model_path: str, source: str, conf: float = 0.5,
            imgsz: int = 1280, save: bool = True):
    """Run inference on an image or video."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(
        source=source,
        conf=conf,
        imgsz=imgsz,
        save=save,
        show_labels=True,
        show_conf=True,
        line_width=2,
        device="0" if _gpu_available() else "cpu",
    )
    print(f"Results saved to: {results[0].save_dir}")
    return results


def detect_events(model_path: str, source: str, conf: float = 0.4,
                  window_size: int = 8, threshold: int = 3,
                  imgsz: int = 1280, output_path: str = "output_events.mp4"):
    """
    Event detection using a sliding window over per-frame predictions.

    An event is declared only when an action appears >= `threshold`
    times within the last `window_size` frames. This filters out
    noisy single-frame false positives.

    Args:
        model_path:   Path to fine-tuned weights
        source:       Input video path
        conf:         Confidence threshold
        window_size:  Sliding window length (frames)
        threshold:    Min detections in window to declare an event
        imgsz:        Inference image size
        output_path:  Output annotated video path
    """
    import cv2
    from collections import deque
    from ultralytics import YOLO

    model = YOLO(model_path)
    device = "0,1,2,3,4,5,6,7" if _gpu_available() else "cpu"

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    window = deque(maxlen=window_size)
    frame_count = 0
    events_log = []

    print(f"Processing {source}...")
    print(f"  Window: {window_size}, Threshold: {threshold}, Conf: {conf}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        detections = results[0].boxes

        if len(detections) > 0:
            best_idx = detections.conf.argmax()
            class_id = int(detections.cls[best_idx])
            class_name = model.names[class_id]
            window.append(class_name)
        else:
            window.append(None)

        # Count actions in window
        action_counts = {}
        for action in window:
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1

        current_event = None
        for action, count in action_counts.items():
            if count >= threshold:
                current_event = action
                break

        annotated = results[0].plot(line_width=2, font_size=1)
        if current_event:
            timestamp = frame_count / fps
            cv2.putText(annotated, f"EVENT: {current_event.upper()}",
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (0, 255, 0), 4)
            if not events_log or events_log[-1][1] != current_event:
                events_log.append((frame_count, current_event, timestamp))

        out.write(annotated)
        frame_count += 1

        if frame_count % 200 == 0:
            print(f"  {frame_count} frames processed...")

    cap.release()
    out.release()

    print(f"\n  Event detection complete!")
    print(f"   Output: {output_path}")
    print(f"   Frames: {frame_count}")
    print(f"   Events: {len(events_log)}")

    if events_log:
        print("\n   Event Log:")
        for fnum, event, ts in events_log:
            mins, secs = divmod(ts, 60)
            print(f"     [{int(mins):02d}:{secs:05.2f}] {event}")

    return events_log


# ────────────────────────────────────────────────────────────
# 5. UTILITIES
# ────────────────────────────────────────────────────────────

def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _find_volleyvision_weights():
    """Locate the VolleyVision pre-trained weights."""
    candidates = [
        Path(__file__).parent.parent / "VolleyVision" / "Stage II - Players & Actions"
            / "actions" / "yV8_medium" / "weights" / "best.pt",
        Path("VolleyVision/Stage II - Players & Actions/actions/yV8_medium/weights/best.pt"),
        Path("best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def validate(model_path: str, data_yaml: str, imgsz: int = 1280):
    """Run validation to check metrics on val/test set."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device="0" if _gpu_available() else "cpu",
    )
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


# ────────────────────────────────────────────────────────────
# 6. CLI
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VolleyVision Fine-Tuning & Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert JSON labels + extract frames from videos
  python finetune_volleyvision.py --mode convert \\
      --json detections_labeled.json \\
      --videos_dir /path/to/videos/

  # Or if you already have extracted frame images
  python finetune_volleyvision.py --mode convert \\
      --json detections_labeled.json \\
      --frames_dir /path/to/extracted_frames/

  # Fine-tune
  python finetune_volleyvision.py --mode train \\
      --data_yaml dataset/data.yaml --epochs 50

  # Inference on video
  python finetune_volleyvision.py --mode predict \\
      --model runs/finetune/weights/best.pt \\
      --source practice.mp4

  # Event detection with sliding window
  python finetune_volleyvision.py --mode events \\
      --model runs/finetune/weights/best.pt \\
      --source practice.mp4

  # Validate
  python finetune_volleyvision.py --mode validate \\
      --model runs/finetune/weights/best.pt \\
      --data_yaml dataset/data.yaml
        """,
    )

    parser.add_argument("--mode", required=True,
                        choices=["convert", "train", "predict", "events", "validate"],
                        help="Operation mode")

    # Convert args
    parser.add_argument("--json", type=str,
                        help="Path to detections_labeled.json")
    parser.add_argument("--videos_dir", type=str,
                        help="Directory with source .mp4 video files")
    parser.add_argument("--frames_dir", type=str,
                        help="Directory with pre-extracted frame images")

    # Train args
    parser.add_argument("--data_yaml", type=str, default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=0)

    # Inference args
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--source", type=str)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="output_events.mp4")

    args = parser.parse_args()

    if args.mode == "convert":
        if not args.json:
            parser.error("--json is required for convert mode")
        convert_json_to_yolo(
            json_path=args.json,
            videos_dir=args.videos_dir,
            frames_dir=args.frames_dir,
            output_dir="dataset",
        )

    elif args.mode == "train":
        finetune(
            data_yaml=args.data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            freeze_layers=args.freeze,
            pretrained_weights=args.model,
        )

    elif args.mode == "predict":
        if not args.source:
            parser.error("--source is required for predict mode")
        predict(args.model or "runs/finetune/weights/best.pt",
                args.source, conf=args.conf, imgsz=args.imgsz)

    elif args.mode == "events":
        if not args.source:
            parser.error("--source is required for events mode")
        detect_events(args.model or "runs/finetune/weights/best.pt",
                      args.source, conf=args.conf, output_path=args.output,
                      imgsz=args.imgsz)

    elif args.mode == "validate":
        validate(args.model or "runs/finetune/weights/best.pt",
                 args.data_yaml, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
