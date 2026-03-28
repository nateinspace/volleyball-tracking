#!/usr/bin/env python3
"""
Detect a volleyball with Ultralytics YOLO-World (yolov8m-worldv2.pt), then track it with SAM 2
for all frames from the first good detection to the end of the video.

Requires: ultralytics, opencv-python, torch, SAM 2 from https://github.com/facebookresearch/sam2
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "video",
        type=Path,
        help="Input video path",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: <input>_tracked.mp4)",
    )
    p.add_argument(
        "--yolo-weights",
        default="yolov8m-worldv2.pt",
        help="YOLO-World weights (default: yolov8m-worldv2.pt)",
    )
    p.add_argument(
        "--prompt",
        default="volleyball",
        help='Open-vocabulary class text for YOLO-World set_classes (default: "volleyball")',
    )
    p.add_argument(
        "--coco-class-id",
        type=int,
        default=None,
        help="If set, skip set_classes/CLIP and keep only detections with this COCO id "
        "(e.g. 32 for sports ball). Useful if CLIP weights cannot download.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.22,
        help="Minimum YOLO confidence for a detection",
    )
    p.add_argument(
        "--air-only",
        action="store_true",
        help="Prefer detections whose bbox center lies in the upper fraction of the frame (in flight)",
    )
    p.add_argument(
        "--max-center-y-ratio",
        type=float,
        default=0.72,
        help="With --air-only: keep detections with center_y < H * this (default 0.72)",
    )
    p.add_argument(
        "--sam2-model",
        default="facebook/sam2.1-hiera-small",
        help="Hugging Face id for SAM 2.x (default: facebook/sam2.1-hiera-small)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda | cpu | mps (default: prefer mps, then cuda, then cpu)",
    )
    p.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Green mask overlay alpha",
    )
    return p.parse_args()


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def pick_device(explicit: Optional[str]) -> str:
    """Resolve device; prefer MPS on Apple Silicon when no device is given."""
    if explicit:
        d = explicit.lower().strip()
        if d == "mps" and not _mps_available():
            print(
                "Requested --device mps but MPS is not available "
                "(needs Apple Silicon PyTorch with MPS built in).",
                file=sys.stderr,
            )
            sys.exit(1)
        return d
    if _mps_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def inference_autocast(device: str):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device == "mps":
        return torch.autocast("mps", dtype=torch.float16)
    return nullcontext()


def best_box_in_frame(
    result,
    frame_shape: Tuple[int, ...],
    conf_min: float,
    coco_class_id: Optional[int],
    air_only: bool,
    max_center_y_ratio: float,
) -> Optional[Tuple[np.ndarray, float]]:
    """Return (xyxy, conf) for the best matching box, or None."""
    h, w = frame_shape[0], frame_shape[1]
    best_xyxy = None
    best_c = -1.0
    for b in result.boxes:
        ci = int(b.cls[0])
        if coco_class_id is not None and ci != coco_class_id:
            continue
        cf = float(b.conf[0])
        if cf < conf_min:
            continue
        xyxy = b.xyxy[0].detach().cpu().numpy()
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0
        if air_only and cy > h * max_center_y_ratio:
            continue
        if cf > best_c:
            best_c = cf
            best_xyxy = xyxy
    return (best_xyxy, best_c) if best_xyxy is not None else None


def find_first_detection(
    video_path: Path,
    model,
    conf_min: float,
    coco_class_id: Optional[int],
    air_only: bool,
    max_center_y_ratio: float,
) -> Tuple[int, np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=conf_min, verbose=False)[0]
            picked = best_box_in_frame(
                results,
                frame.shape,
                conf_min,
                coco_class_id,
                air_only,
                max_center_y_ratio,
            )
            if picked is not None:
                xyxy, cf = picked
                return idx, xyxy, cf
            idx += 1
    finally:
        cap.release()
    raise RuntimeError(
        "No volleyball (or matching COCO class) found. "
        "Try lowering --conf, dropping --air-only, or using --coco-class-id 32."
    )


def extract_video_to_indexed_jpegs(video_path: Path, out_dir: Path) -> int:
    """
    SAM2 accepts a directory of JPEGs named {frame_index}.jpg (sorted by int stem).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            p = out_dir / f"{i}.jpg"
            if not cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Failed to write {p}")
            i += 1
    finally:
        cap.release()
    if i == 0:
        raise RuntimeError(f"No frames read from {video_path}")
    return i


def sam2_input_path(video_path: Path) -> tuple[str, Optional[tempfile.TemporaryDirectory]]:
    """
    Return (path for SAM2 init_state, temp dir to clean up, or None).
    Uses the MP4 directly if `decord` is available; otherwise extracts JPEG frames (no decord required).
    """
    try:
        import decord  # noqa: F401

        _ = decord  # silence lint
        return str(video_path), None
    except ImportError:
        pass
    tmp = tempfile.TemporaryDirectory(prefix="sam2_frames_")
    n = extract_video_to_indexed_jpegs(video_path, Path(tmp.name))
    print(f"Using {n} extracted JPEG frames (install `decord` to load MP4 directly).")
    return tmp.name, tmp


def overlay_mask_bgr(frame_bgr: np.ndarray, mask_bool: np.ndarray, alpha: float) -> np.ndarray:
    out = frame_bgr.copy()
    color = np.array([0, 255, 0], dtype=np.float32)
    m = mask_bool.astype(bool)
    if not m.any():
        return out
    blended = out[m].astype(np.float32) * (1.0 - alpha) + color * alpha
    out[m] = blended.astype(np.uint8)
    return out


def mask_to_bool(video_res_mask: torch.Tensor) -> np.ndarray:
    """video_res_mask: typically (1, H, W) logits or probs."""
    m = video_res_mask
    if m.dim() == 3:
        m = m[0]
    elif m.dim() == 4:
        m = m[0, 0]
    return (m > 0.0).detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"File not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        out_path = video_path.with_name(video_path.stem + "_tracked.mp4")

    device = pick_device(args.device)
    print(f"Device: {device}")

    try:
        from ultralytics import YOLOWorld
    except ImportError as e:
        print("Install ultralytics: pip install ultralytics", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as e:
        print(
            "Install SAM 2: pip install git+https://github.com/facebookresearch/sam2.git",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    # --- YOLO-World: open vocabulary or fixed COCO id
    model = YOLOWorld(str(args.yolo_weights))
    model.to(device)
    if args.coco_class_id is None:
        try:
            model.set_classes([args.prompt])
        except Exception as e:
            print(
                "\nset_classes failed (CLIP text embeddings). "
                "If this is an SSL/cert issue, download CLIP ViT-B/32 weights manually, "
                "or re-run with e.g. --coco-class-id 32 for COCO 'sports ball'.\n",
                file=sys.stderr,
            )
            raise SystemExit(1) from e

    print("Scanning video for first detection…")
    det_frame, xyxy, det_conf = find_first_detection(
        video_path,
        model,
        args.conf,
        args.coco_class_id,
        args.air_only,
        args.max_center_y_ratio,
    )
    print(
        f"Detection at frame {det_frame} conf={det_conf:.3f} box xyxy="
        f"[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
    )

    # --- SAM 2 video predictor (Hugging Face checkpoint)
    predictor = SAM2VideoPredictor.from_pretrained(args.sam2_model, device=device)

    cap_info = cv2.VideoCapture(str(video_path))
    fps = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_info.release()

    # Box prompt uses pixel xyxy; SAM2 normalizes internally (keep float32 on CPU like the reference notebook).
    box_t = torch.tensor(xyxy, dtype=torch.float32)

    autocast = inference_autocast(device)

    masks_by_frame: dict[int, np.ndarray] = {}

    sam_path, sam_tmp = sam2_input_path(video_path)
    try:
        with torch.inference_mode(), autocast:
            state = predictor.init_state(
                sam_path,
                offload_video_to_cpu=(device == "cpu"),
                offload_state_to_cpu=(device == "cpu"),
            )
            # Box stays on CPU: SAM2 concatenates with CPU point tensors, then moves to inference_state["device"].
            predictor.add_new_points_or_box(
                state,
                frame_idx=det_frame,
                obj_id=0,
                box=box_t,
            )
            for fidx, obj_ids, video_res_masks in predictor.propagate_in_video(
                state,
                start_frame_idx=det_frame,
                max_frame_num_to_track=None,
                reverse=False,
            ):
                vm = video_res_masks[0]
                masks_by_frame[fidx] = mask_to_bool(vm)
    finally:
        if sam_tmp is not None:
            sam_tmp.cleanup()

    print(f"Tracked {len(masks_by_frame)} frames from {det_frame} to end (total frames ~{n_frames}).")

    # --- Write output: original frames; green overlay where SAM mask exists
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Could not open VideoWriter for {out_path}", file=sys.stderr)
        sys.exit(1)

    fi = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi in masks_by_frame:
                frame = overlay_mask_bgr(frame, masks_by_frame[fi], args.mask_alpha)
            writer.write(frame)
            fi += 1
    finally:
        cap.release()
        writer.release()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
