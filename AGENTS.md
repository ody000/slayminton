# Slayminton Agent Notes

## Architectural Constraints
- Keep the pipeline modular: video ingestion/masking, tracking, game-state, analysis, visualization.
- `models/dino.py` owns DINO training and frame-level object localization logic.
- A single DINO model should handle both tracked targets (player and shuttle) in one forward pass.
- Training supports self-supervised multi-crop DINO objectives plus supervised detection targets from COCO boxes.
- `utils/video_io.py` is an external dependency point and should not be modified from DINO implementation work.

## Repository-Specific Rules
- Input annotations are COCO-style JSON and currently include categories: badminton, person, racket, shuttle.
- Tracking targets are normalized to two canonical classes: `player` (from `person`) and `shuttle`.
- Dataset split policy in training loop is 80/20 train/validation.
- Validation metrics are reported using IoU and mAP@0.75.
- Tracking output format per object is `(timestamp, x, y, height, width)`.

## Expected Behaviors
- `DINODataset` should:
	- Parse images + annotations from `data/input/train` and `_annotations.coco.json`.
	- Produce DINO multi-crop views (global + local) for self-supervision.
	- Produce detector-ready image tensor and class-specific bbox targets.
	- Apply geometric augmentation that updates annotations consistently (e.g., horizontal flip).
- `DINOTracker` should:
	- Load/save checkpoint weights.
	- Run inference on one frame and return shuttle/player locations from one model.
	- Support CPU/GPU execution via runtime device selection.
- `train_dino` should:
	- Use student-teacher DINO training with centering and EMA updates.
	- Use cosine LR scheduling and EMA momentum scheduling.
	- Evaluate detection metrics periodically and persist checkpoints/history artifacts.
- `visualize_training` should:
	- Save sample prediction overlays for student + teacher.
	- Save loss and metric curves.

## Iteration Log (Current)
- Implemented full `models/dino.py`:
	- Added a complete `DINOTracker` class with ViT encoder, DINO projector, and multi-target detector head.
	- Added a complete `DINODataset` implementation with COCO parsing and annotation-aware augmentation.
	- Added `ViTEncoder` and robust class-token extraction for timm ViT variants.
	- Implemented end-to-end `train_dino` with centering, cosine schedules, EMA teacher updates, and validation (IoU/mAP@0.75).
	- Implemented `visualize_training` to export qualitative and metric plots.
- Implemented `main.py`:
	- Added CLI modes for training (`train`) and dry-run frame tracking (`track-frames`).
	- Wired dataset creation, model training, checkpoint writing, and result export to JSON.
	- Kept video integration decoupled so future `video_io.py` output tensors can feed directly into tracker inference.

## Integration Notes For Next Iteration
- Replace `track-frames` directory scan in `main.py` with direct frame tensor stream from `utils/video_io.py` once available.
- Optionally upgrade detector head to multi-instance predictions for two players when team-level tracking is needed.
- Add unit tests for:
	- COCO parsing and class mapping.
	- Box transformation correctness under horizontal flip.
	- Metric computations for IoU and mAP@0.75.
