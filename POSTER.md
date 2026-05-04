Introduction
Badminton is fast and demands precise shot placement and footwork. Manual replay analysis is slow and lacks automated metrics, leaving coaches without scalable tools for rally, movement, and shot analysis. Tracking is challenging due to a small, fast shuttlecock and dynamic play. We built Slayminton, a computer-vision pipeline that tracks players and shuttlecocks and extracts game statistics from video.

**Goals:** (1) Real-time player and shuttle detection, (2) automatic rally segmentation and statistics, (3) footwork heatmaps via homography, (4) annotated video exports for review.

Methods
Data Processing Pipeline
(flowchart placeholder)
We use MOG2 background subtraction to isolate moving objects (white) from background (black).
Citation for MOG2 paper:
[PUT PAPER CITATION HERE]
Player tracking uses DINOv3; shuttle tracking uses TrackNetv2.
Court corners are selected via a GUI and mapped with 2D homography to compute player positions and footwork.
The pipeline outputs annotated video, rally data, and player movement visualizations.
Data Collection
We collected 10,259 annotated frames from public video datasets. Annotations include players and shuttle; court corners were added manually for homography.
We augmented the dataset with horizontal flips.
Citation:
badminton Dataset by badminton, available at Roboflow Universe (https://universe.roboflow.com/badminton-rojkf/badminton-hehp8), licensed under CC BY 4.0.
Model Training
We fine-tuned a DINOv3 (pretrained with DINOv2-vitb14 head). In 12 epochs, IoU rose from 62% to 69% and mAP@0.5 from 63% to 74%.
A 100-epoch baseline raw DINO training reached IoU 33% and mAP@0.5 15%.
TrackNetv2 pretrained weights outperform DINOv3 on shuttle tracking without extra training.
Game Analysis
We detect rally activity, shot events, in/out placement, and compute player court-coverage heatmaps.
To reduce errors we use a 1–3 frame grace period for rally transitions, reject boxes with <5% white pixels in MOG2 masks, and disable stationary candidates.
Visualization
Model-generated bounding boxes are overlaid on each frame.
Player footwork heatmaps are generated via homography to a bird's-eye view.

Results
We built a system for real-time player and shuttle tracking that produces game statistics and annotated video.

Discussion
The framework provides automated replay analysis for players and coaches. Shuttle tracking needs further tuning, but the pipeline enables coaching feedback and raw training insights. Future work includes scorekeeping, multi-match analysis, shot classification, and performance prediction.

Limitations
Tracking errors remain, especially for the shuttle; DINOv3 underperformed so we used TrackNet for more reliable shuttle detection. Shot-type classification was not feasible without much more accurate shuttle/racket tracking or posture models.
Frame-rate assumptions were estimated for MOG2 and may reduce mask quality; stationary moments are less reliable. 
The dataset lacks full-match videos, limiting full-game state tracking and scorekeeping.

Project idea: Automated badminton replay analysis for shuttle and player trajectories, scoring, and game metrics.
Context: Computer-assisted replay simplifies scorekeeping and enables data-driven training.
Stakeholders: players, coaches, and audiences benefit from analytics, coaching insights, and clearer replays.