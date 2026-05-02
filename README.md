# slayminton
Badminton game analysis project that uses computer vision to track shuttlecocks and players.

# Workflow
`main.py`: The CLI entrypoint. We have three modes:
	- `train`: runs DINO training using `models/dino.py` and COCO-style annotations in `data/input/train`.
	- `track-frames`: dry-run DINO over a folder of frames.
	- `track-video`: full pipeline (frame extraction, MOG2 masking, detection, rally analysis, visualization).

`utils/video_io.py`: Frame extraction and background subtraction helpers.
	- `extract_frames`: extracts JPG frames into `output_dir`.
	- `apply_mog2_to_frames`: creates cv.MOG2 mask frames.

`models/dino.py`: DINO model, `DINOTracker` class wrapper and training utilities.
	- `DINOTracker.detect`: returns player/shuttle detections for a frame.

`core/game_state.py`: Rally and game-state tracking using shuttle motion inactivity rules.

`core/analysis.py`: Compute rally statistics and generate metric visualizations (histogram PNGs).

`scripts/visualizations.py`: Includes many visualization utilities (replaces earlier `utils/visualization.py`).
	- `process_video`: builds a court-insert heatmap from mask frames and renders a visualization MP4.
	- `draw_dino_boxes_with_heatmap`: produces annotated MP4s with DINO boxes and a pasted, stationary court heatmap (bottom-right insert).
	- Along with other debugging diagnostic methods.

Output layout (per run through `track-video`):
	- `data/output/<video_name>_<timestamp>/`
		- `tracking_results.json` — per-frame detection output.
		- `rally_data.json` — rally segments.
		- `rally_statistics.json` — analysis outputs and paths to visualization artifacts.
		- `.temp/`
			- `original_frames/` — extracted original frames
			- `mask_frames/` — MOG2 mask frames
			- `masked_frames/` — masked RGB frames with boxes (auto-generated)

Run example (track-video):
```
python main.py --mode track-video --video-path data/input/match_clip.mp4 --weights dino_tracker.pt --fps 30
```

Notes:
- The DINO detector weights (`dino_tracker.pt`) may need retraining to improve per-frame localization — use `--mode train` to produce updated checkpoints.
- The visualization pipeline prefers mask-based player detection for stable P1/P2 assignment and constructs a stationary court coverage heatmap (no moving trail for the shuttle; shuttle shown as per-frame marker).

