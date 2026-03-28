# Zaven's Notebooks

Four pipeline notebooks for volleyball video analysis — from video enhancement and 3D body estimation to player tracking and action detection.

---

## 1. sam3dbody_video.ipynb — 3D Body Mesh Viewer from Video

Takes `video.mp4` as input and produces an interactive, web-ready 3D human body mesh viewer using Facebook's SAM 3D Body model.

- Clones the `sam-3d-body` repo and installs dependencies (detectron2, MoGe, PyOpenGL, etc.)
- Loads the `facebook/sam-3d-body-dinov3` model along with optional human detector and FOV estimator
- Extracts video frames (with configurable frame-skip) and runs per-frame 3D body estimation
- Serializes mesh data (vertices, camera transforms, skeleton keypoints) into compact binary chunks with a JSON manifest
- Generates a self-contained HTML + Three.js viewer with playback controls, wireframe/solid toggle, skeleton bone overlay, person selection, and orbit camera

## 2. VideoEnhancementPipeline.ipynb — SeedVR2 Video Super-Resolution

GPU-intensive video restoration/upscaling pipeline using ByteDance's SeedVR2 7B model, tailored for enhancing low-bitrate college volleyball footage.

- Bootstraps a full conda environment (micromamba, PyTorch 2.4 w/ CUDA 12.1, flash-attn, apex) and clones the SeedVR repo
- Downloads the SeedVR2 7B checkpoint (~7 billion parameter diffusion model) from Hugging Face
- Creates a lossless constant-frame-rate mezzanine from the source video, then splits it into overlapping chunks sized to fit in GPU VRAM
- Runs the SeedVR2 "normal" restoration pass on each chunk using multi-GPU sequence-parallel and data-parallel scheduling (auto-scales across 4–16 GPUs)
- Reassembles the restored chunks into a single high-quality `tracking_master_cv.mp4` (H.264, CRF 8), with audio preserved and intermediate files cleaned up

## 3. YOLO26_PersonTracker.ipynb — Person Detection & Tracking

Persistent person detection and ID tracking on volleyball video using YOLO26x (Extra Large, 57.5 mAP) with BoT-SORT + ReID.

- Loads the video and applies preprocessing: slight rotation correction (-0.55 degrees) and court-side filtering (keeps only the "near" side below the net line)
- Uses `yolo26x.pt` for detection and `yolo26x-cls.pt` for ReID appearance embeddings, with a custom BoT-SORT config tuned to minimize ID swaps during player crossings
- Runs frame-by-frame tracking at IMGSZ=1920, filtering detections to the near court side
- Renders an annotated output video (`tracked.mp4`) with colored bounding boxes, person IDs, and confidence scores
- Exports `tracked_tracks.json` with per-frame bounding boxes and persistent track IDs, plus individual frame PNGs

## 4. IDandMovesCombinedPipeline.ipynb — Volleyball Move Detection & ID Mapping

Detects volleyball actions (block, serve, set, attack, dig) and maps them to existing tracked person IDs. Builds on the output of the tracking notebook.

- Loads a custom-trained YOLO model (`best.pt`) for action/move classification
- Processes 5 separate corrected volleyball videos, each with a corresponding `corrected_tracked_tracks.json` from the tracker
- Applies the same rotation/crop preprocessing as the tracker, plus masks out the far court side so only near-court actions are detected
- Runs batched inference, then maps each detected move to the nearest tracked person using a combined IoU + containment overlap metric
- Produces `combined_tracks_and_moves.json` per video — the original person IDs and boxes, enriched with a "moves" list per person per frame
- Renders overlay videos (`overlay_with_moves.mp4`) showing person boxes with IDs and action labels
