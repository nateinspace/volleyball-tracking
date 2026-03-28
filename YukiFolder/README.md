# Volleyball Vision: AI-Powered Practice Analysis Pipeline

An end-to-end pipeline for detecting and classifying volleyball actions from practice footage using YOLO26x for person detection and a custom labeling workflow for action classification.

Built for the **Waves Innovation Summit — Pepperdine × StatsPerform AI Hackathon**.

---

## The Problem

Volleyball coaching relies on manual video review to identify and analyze player actions — digs, sets, attacks, blocks, and serves. This is slow, inconsistent, and doesn't scale. Automating this requires solving two hard problems: detecting *who* is on court, and classifying *what* they're doing — including actions that look identical in a single frame but differ based on game context.

---

## Pipeline Components

### 1. Frame Extraction + Person Detection (`volleyball_extract-2.ipynb`)

The first half of the notebook processes multiple rally videos in a single run using YOLO26x for person detection.

**Key design decisions:**
- **YOLO26x** (extra-large variant) chosen over smaller models because players appear small in wide-angle practice footage — the higher parameter count improves detection at distance
- **1280px input resolution** instead of the default 640px, critical for detecting players that occupy only 50-80 pixels in a 1080p frame
- **FP16 inference** on CUDA for ~2x speedup with negligible accuracy loss
- **Low confidence threshold (0.25)** to over-detect rather than miss players — false positives are easily removed in the labeler, but missed detections require manual box drawing
- **Configurable frame sampling** (every N frames) with start/end skip to trim dead time
- **Video ID prefixing** (`v0_frame_000035.jpg`) to unify multiple videos into a single output without naming collisions

**Output:** A `frames/` directory of extracted JPEGs and a `detections.json` containing per-frame bounding boxes for every detected person, all defaulting to the `idle` label.

### 2. Action Labeler (`volleyball_action_labeler.html`)

A standalone HTML/JS application for manually classifying detected players into action categories. Designed specifically for the visual challenges of volleyball footage.

**UI features addressing volleyball-specific problems:**

- **Corner bracket annotations** instead of full bounding boxes — in volleyball, players cluster at the net during plays, and overlapping full rectangles obscure the body movements you need to see for classification. L-shaped corner markers show detection boundaries without covering the player
- **Side-by-side clean/annotated view** — left panel shows the raw frame for studying body position, right panel shows annotations for labeling context
- **Click-to-isolate selection** — when you select a player, all other annotations dim to 25% opacity, focusing attention on one detection at a time
- **Hover-to-highlight** — hovering over corner brackets fills the detection region with a light tint so you can quickly identify which corners belong together in crowded scenes
- **Toggle visibility** (`H` key) — instantly hides all annotations to see the raw frame, press again to restore
- **Opacity slider** — continuous control over annotation visibility from 0-100%
- **Draw mode** (`B` key) — click-and-drag to draw new bounding boxes for players YOLO missed
- **Keyboard-driven workflow** — `A`/`D` for frame navigation, `1-6` for label assignment, `Tab` to cycle through detections, `Delete` to remove false positives

**Action classes:**

| Key | Class | Description |
|-----|-------|-------------|
| 1 | **Dig** | Defensive forearm pass — also covers receives (first contact after serve). Visually identical to a receive; temporal post-processing distinguishes them. |
| 2 | **Set** | Overhead ball positioning for an attacking opportunity |
| 3 | **Attack** | Offensive strike directed over the net |
| 4 | **Block** | Defensive jump at the net to intercept an attack |
| 5 | **Serve** | Rally-initiating action. Label from the toss through follow-through, not just the contact frame. |
| 6 | **Idle** | No notable action — default for all unlabeled detections |

**Classes we intentionally excluded:**

- **Cover** — dropped because a player covering looks identical to a player in ready position. The distinguishing factor is entirely contextual (teammate just attacked, ball got blocked back), making it unsuitable for visual classification. Better handled as a post-processing rule.
- **Receive** — merged with Dig for training. Visually identical (both are forearm passes). Distinguished by temporal rule: if a serve was detected in the preceding 1-2 seconds from the opposing side, the first "dig" is reclassified as a receive.

**Output:** A `detections_labeled.json` with action labels assigned to each bounding box.

### 3. Clip Extraction (`volleyball_extract-2.ipynb`)

The second half of the notebook converts frame-level labels into player-centered video clips.

**How it works:**
- For each labeled detection, goes back to the **original source video** (not the extracted frames) and pulls a **2-second temporal window** centered on the labeled frame
- Samples **16 frames uniformly** across that window — this captures the full motion arc (approach → contact → follow-through) for any action
- Crops each frame around the player's bounding box with **1.5x padding**, then resizes to **224×224**
- Saves as both `.npz` (compressed NumPy, fast for training) and `.mp4` (for visual verification)

**Handling class imbalance:**

In volleyball practice footage, ~95% of detections in any frame are idle players. Without intervention, the training set would be overwhelmingly idle. The extractor addresses this with:

- **Idle capping** — randomly downsamples idle clips to a configurable ratio relative to total action clips (default: 1:1). This transforms a 25000:500 idle:action split into a balanced ~500:500 split
- **Per-class warnings** — flags any action class with fewer than 40 examples and recommends additional labeling

---

## Labeling Strategy

With limited data (5 rally videos, 13-30 seconds each), efficient labeling is critical.

**Label every frame of an action, not just the contact moment.** A serve spans ~1-1.5 seconds from toss to follow-through. At 5-frame extraction intervals, that's 12-18 frames where the server is visibly in motion. Labeling all of them multiplies your training data: 5 serve events become 60-90 serve training clips.

**Don't label decoys.** When multiple players run approach patterns but only one gets the set, only the player who contacts the ball gets an action label. Everyone else stays idle. A temporal classifier can learn that the ball entering a player's space is what makes an action real.

**Non-players can stay as idle.** Coaches and spectators detected by YOLO don't need special treatment. The idle cap randomly downsamples to ~1:1 with action clips, so bystanders become a small fraction of the surviving idle examples rather than the majority.

---

## Cover, and Receive vs Dig: Temporal Post-Processing

Cover (teammates positioning to recover a blocked ball) was excluded from visual classification because it is visually indistinguishable from ready position. A potential post-processing rule: if an attack was detected AND the ball trajectory reverses AND a nearby teammate makes contact, classify as cover.

The receive (first contact after a serve) and dig (defensive pass mid-rally) are visually identical — both are forearm platform passes. No single-frame classifier can distinguish them.

**Solution:** Merge them as a single "dig" class during training, then apply a temporal rule at inference:

```
IF a serve was detected in the preceding 1-2 seconds
   from the opposing side of the court
AND this player's predicted action is "dig"
THEN reclassify as "receive"
```

This works because serve detection is highly reliable (distinctive toss + endline court position), making the temporal rule robust.

---

# Volleyball Play Recognition 

Fine-tuned a YOLOv8m object detection model for recognizing volleyball actions in Pepperdine practice footage.

## Approach

Our approach uses transfer learning on a YOLOv8m object detection model. We started from VolleyVision (https://github.com/shukkkur/VolleyVision)'s pre-trained weights — a YOLOv8m model already fine-tuned on volleyball action data (block, defense, serve, set, spike) — which gave us a backbone that already understands volleyball-specific visual features like player poses, court context, and ball interactions. We then re-trained the model on hand-labeled frames from our available practice footage, replacing the detection head to output our 5 target action classes (block, serve, set, attack, dig). Because the VolleyVision backbone had already learned volleyball-relevant features, all layers transferred directly, and the model only needed to adapt its final detection head to our specific class definitions and practice footage domain.

## Files

- `volleybay_Play__Recognition.ipynb` — Full pipeline notebook
- `finetune_volleyvision.py` — Data conversion, training, and event detection CLI
- `volleyvision_actions_best.pt` — VolleyVision pre-trained weights (downloaded from GitHub)

**Not included in repo (private practice data):**
- `detections_labeled.json` — Hand-labeled annotations containing bounding boxes and action labels for each player across 1,134 frames from 5 practice video clips. Labels include block, serve, set, attack, dig, idle, and ignore.
- `volleyball_output.zip` — Extracted JPEG frames from the practice videos, referenced by the JSON annotations.

## Future Work & Approaches Explored

### VideoMAE v2 Temporal Classifier

We built a two-stage training pipeline using **VideoMAE v2** (ViT-Large, pretrained on Kinetics-400) as a temporal action classifier. The approach: YOLO26x handles person detection, then 2-second player-centered clips are fed to VideoMAE for action classification with full motion context.

The architecture is promising because it addresses several volleyball-specific problems a frame-level classifier cannot:
- **Decoy filtering** — the model sees whether the ball enters a player's space across the clip duration, distinguishing real attacks from fake approach runs
- **Serve vs attack** — visually identical at the contact frame, but the temporal window captures the toss (serve) vs approach jump (attack)
- **Covering vs not** — distinguish player covering for the attacker vs. somewhere else
- **Natural receive/dig distinction** — the 2-second clip preceding a receive would contain a serve; a dig clip would not


The training pipeline supports two-stage fine-tuning: Stage 1 on an external volleyball dataset, Stage 2 on labeled practice footage. Includes weighted cross-entropy for class imbalance, partial layer freezing, and cosine LR scheduling. This remains the recommended next step for improving classification accuracy.

### VNL-STES Dataset

The **VNL-STES dataset** (CVPR 2025 Workshop) contains 1,028 rally videos with 251,110 frames and 6,137 annotated events from the Volleyball Nations League. It covers serve, receive, set, spike, block, and score with frame-level temporal and spatial annotations. A preparation script (`vnl_stes_preparation.py`) was written to convert this into player-centered clips matching our format for VideoMAE pretraining. We were unable to complete the download and integration within the hackathon timeline, but this dataset would serve as an ideal Stage 1 pretraining source for the VideoMAE classifier.

**Data link:** https://hoangqnguyen.github.io/stes/