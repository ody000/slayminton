"""
Entry point for the Slayminton application.
The main execution loop implemented here.
"""

import argparse
from datetime import datetime
import json
import os
from typing import List
import shutil

import cv2
import numpy as np
from PIL import Image
import torch

from core.analysis import Analysis
from core.game_state import GameState
from models.dino import DINODataset, DINOTracker, train_dino
from utils.video_io import extract_frames, apply_mog2_to_frames


"""
Main execution method for the Slayminton application.
1. Define the paths for input video, output video, and CSV data. Handle
   edge cases, command-line arguments, or configuration files as needed.
2. Initialize the modules for video IO, DINOv3 for shuttle/player detection,
   homography, kinematics, and data writing.
3. Open CSV file for writing extracted data.
4. Start the main execution loop: for each video frame, perform shuttle and
   player detection, apply homography to get 3D coordinates, compute kinematics,
   and record the data.
5. Visualize the output with video IO. Handle graceful shutdown and resource
   cleanup after processing is complete.
"""

def main():
    # some CLI arguments for flexibility
    print("pre_argparse")
    parser = argparse.ArgumentParser(description="Slayminton DINOv3 training + tracking")
    parser.add_argument("--mode", choices=["train", "track-frames", "track-video"], default="train")
    parser.add_argument("--train-dir", default="data/input/train")
    parser.add_argument("--annotations", default="data/input/train/_annotations.coco.json")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--weights", default="data/output/dino_tracker.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # LoRA fine-tuning flags (can be enabled via env USE_LORA=1 in slurm_train.sh)
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=(os.environ.get("USE_LORA", "0") == "1"),
        help="Enable LoRA adapter fine-tuning on the encoder (lightweight)",
    )
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "4")), help="LoRA rank r")
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=int(os.environ.get("LORA_ALPHA", "16")),
        help="LoRA alpha scaling",
    )
    parser.add_argument(
        "--frames-dir",
        default="data/input/train",
        help="Directory containing RGB image frames for dry-run tracking mode",
    )
    parser.add_argument(
        "--video-path",
        help="Path to input MP4 video for track-video mode",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=120,
        help="Maximum number of frames to process in track-frames mode",
    )
    print("[MAIN] check 1")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rally-timeout-s", type=float, default=0.5)
    parser.add_argument("--min-shuttle-motion-px", type=float, default=2.0)
    args = parser.parse_args()
    print(f"[MAIN] mode={args.mode}")
    print(f"[MAIN] output_dir={args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    # GPU when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] device={device}")


    # If in training mode, run DINOv3 training loop and exit.
    if args.mode == "train":
        print("[MAIN] entering training mode")
        # Build dataset from COCO annotations
        dataset = DINODataset(
            device=device,
            data_dir=args.train_dir,
            annotations_file=args.annotations,
        )
        print(f"[MAIN] dataset_size={len(dataset)} train_dir={args.train_dir}")
        # full training loop
        _, _, history = train_dino(
            student=None,
            teacher=None,
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            checkpoint_name=os.path.basename(args.weights),
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )

        print("[MAIN] training_complete")
        print(f"Saved student checkpoint: {os.path.join(args.output_dir, os.path.basename(args.weights))}")
        print(f"Last train loss: {history.train_loss[-1]:.4f}")
        if history.val_iou:
            print(f"Last val IoU: {history.val_iou[-1]:.4f} | Last val mAP@0.75: {history.val_map75[-1]:.4f}")
        return


    # Actual tracking starts here (if not training mode).
    # Create one folder per input source/run under data/output.
    if args.mode == "track-frames":
        _run_track_frames(args, device)
    elif args.mode == "track-video":
        _run_track_video(args, device)
    else:
        print(f"[MAIN] unknown mode: {args.mode}")


def _run_track_frames(args, device):
    """Track detections in a directory of frame images."""
    input_tag = os.path.basename(os.path.normpath(args.frames_dir)) or "input"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"{input_tag}_{run_tag}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"[MAIN] run_output_dir={run_output_dir}")

    tracker = DINOTracker(weights_path=args.weights if os.path.exists(args.weights) else None, device=device)
    rally_tracker = GameState(
        inactive_timeout_s=args.rally_timeout_s,
        min_displacement_px=args.min_shuttle_motion_px,
    )
    analysis = Analysis()

    frame_files: List[str] = sorted(
        [
            os.path.join(args.frames_dir, p)
            for p in os.listdir(args.frames_dir)
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )[: args.frame_limit]
    print(f"[MAIN] track_frames count={len(frame_files)} fps={args.fps}")

    results = []
    progress_interval = max(1, int(args.fps))
    for i, frame_path in enumerate(frame_files):
        # Convert frame index to seconds so timeout logic is meaningful.
        timestamp = float(i) / max(args.fps, 1e-6)
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
        det = tracker.detect(frame, timestamp=timestamp)
        rally_active = rally_tracker.update(timestamp=timestamp, shuttle_det=det.get("shuttle"))
        results.append(
            {
                "frame_path": frame_path,
                "timestamp": timestamp,
                "player": det.get("player"),
                "shuttle": det.get("shuttle"),
                "rally_active": rally_active,
            }
        )

        if (i + 1) % progress_interval == 0 or (i + 1) == len(frame_files):
            print(
                f"[MAIN] frame={i + 1}/{len(frame_files)} "
                f"t={timestamp:.3f}s rally_active={rally_active}"
            )

    final_timestamp = (float(len(frame_files) - 1) / max(args.fps, 1e-6)) if frame_files else 0.0
    rally_tracker.finalize_rally_data(final_timestamp)
    rally_data = rally_tracker.get_rally_data()

    analysis_results = analysis.compute_rally_statistics(rally_data)
    analysis_results["output_dir"] = run_output_dir
    visualization_paths = analysis.visualize_results(analysis_results)

    output_path = os.path.join(run_output_dir, "tracking_results.json")
    rally_output_path = os.path.join(run_output_dir, "rally_data.json")
    stats_output_path = os.path.join(run_output_dir, "rally_statistics.json")

    # use JSON for output due to flexibility to store nested data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(rally_output_path, "w", encoding="utf-8") as f:
        json.dump(rally_data, f, indent=2)

    with open(stats_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "statistics": analysis_results,
                "visualizations": visualization_paths,
            },
            f,
            indent=2,
        )

    print(f"[MAIN] processed_frames={len(frame_files)}")
    print(f"[MAIN] saved_tracking={output_path}")
    print(f"[MAIN] saved_rally_data={rally_output_path}")
    print(f"[MAIN] saved_stats={stats_output_path}")
    print(f"[MAIN] run_complete dir={run_output_dir}")


def _run_track_video(args, device):
    """Track detections in an MP4 video, apply MOG2, and generate court visualizations."""
    if not args.video_path:
        print("[MAIN] error: --video-path required for track-video mode")
        return
    
    if not os.path.exists(args.video_path):
        print(f"[MAIN] error: video not found: {args.video_path}")
        return
    
    # Create output directory
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"{video_name}_{run_tag}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"[MAIN] run_output_dir={run_output_dir}")
    
    # Create temp directories for frames and masks
    temp_dir = os.path.join(run_output_dir, ".temp")
    original_frames_dir = os.path.join(temp_dir, "original_frames")
    mask_frames_dir = os.path.join(temp_dir, "mask_frames")
    
    try:
        # Extract frames from MP4
        print(f"[MAIN] extracting frames from {args.video_path}")
        frame_paths = extract_frames(args.video_path, original_frames_dir, fps=args.fps)
        
        if not frame_paths:
            print("[MAIN] error: no frames extracted from video")
            return
        
        # Apply MOG2 background subtraction
        print("[MAIN] applying MOG2 background subtraction")
        mask_paths = apply_mog2_to_frames(frame_paths, mask_frames_dir, fps=args.fps)
        
        if len(mask_paths) != len(frame_paths):
            print(f"[MAIN] warning: frame/mask count mismatch ({len(frame_paths)} vs {len(mask_paths)})")

        runtime_court_corners = None
        if frame_paths:
            try:
                print("[MAIN] opening court calibration GUI")
                from scripts.visualizations import set_court_points
                first_frame = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
                runtime_court_corners = set_court_points(first_frame, video_name, save=False)
            except Exception as e:
                print(f"[MAIN] warning: court calibration GUI failed ({e})")
        
        # Initialize tracker, game state, and analysis
        tracker = DINOTracker(weights_path=args.weights if os.path.exists(args.weights) else None, device=device)
        rally_tracker = GameState(
            inactive_timeout_s=args.rally_timeout_s,
            min_displacement_px=args.min_shuttle_motion_px,
        )
        analysis = Analysis()
        
        # Process frames with DINO tracking and game state
        print("[MAIN] running DINO tracking and game state analysis")
        results = []
        progress_interval = max(1, int(args.fps))
        
        for i, frame_path in enumerate(frame_paths):
            timestamp = float(i) / max(args.fps, 1e-6)
            frame = np.asarray(Image.open(frame_path).convert("RGB"))
            
            # Use MOG2 mask frame for detection (better contrast for motion)
            if i < len(mask_paths):
                mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Convert grayscale mask to RGB by repeating channels
                    frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            det = tracker.detect(frame, timestamp=timestamp)
            rally_active = rally_tracker.update(timestamp=timestamp, shuttle_det=det.get("shuttle"))
            results.append(
                {
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "player": det.get("player"),
                    "shuttle": det.get("shuttle"),
                    "rally_active": rally_active,
                }
            )
            
            if (i + 1) % progress_interval == 0 or (i + 1) == len(frame_paths):
                print(
                    f"[MAIN] frame={i + 1}/{len(frame_paths)} "
                    f"t={timestamp:.3f}s rally_active={rally_active}"
                )
        
        # Finalize rally data and compute statistics
        final_timestamp = (float(len(frame_paths) - 1) / max(args.fps, 1e-6)) if frame_paths else 0.0
        rally_tracker.finalize_rally_data(final_timestamp)
        rally_data = rally_tracker.get_rally_data()
        
        analysis_results = analysis.compute_rally_statistics(rally_data)
        analysis_results["output_dir"] = run_output_dir
        visualization_paths = analysis.visualize_results(analysis_results)
        
        # Save tracking and rally JSON outputs
        output_path = os.path.join(run_output_dir, "tracking_results.json")
        rally_output_path = os.path.join(run_output_dir, "rally_data.json")
        stats_output_path = os.path.join(run_output_dir, "rally_statistics.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        with open(rally_output_path, "w", encoding="utf-8") as f:
            json.dump(rally_data, f, indent=2)
        
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "statistics": analysis_results,
                    "visualizations": visualization_paths,
                },
                f,
                indent=2,
            )
        
        # Generate annotated video using draw_dino_boxes_with_heatmap from scripts/visualizations.py
        print("[MAIN] generating annotated video with DINO bboxes and shuttle heatmap")
        try:
            annotated_video_path = os.path.join(run_output_dir, f"{video_name}_annotated.mp4")
            # Import here to avoid circular dependency issues
            from scripts.visualizations import draw_dino_boxes_with_heatmap
            draw_dino_boxes_with_heatmap(
                frame_paths,
                results,
                annotated_video_path,
                fps=int(args.fps),
                court_corners=runtime_court_corners,
                mask_paths=mask_paths,
            )
        except Exception as e:
            print(f"[MAIN] warning: DINO visualization failed ({e})")
        
        # Generate court visualization MP4 using scripts/visualizations.py
        print("[MAIN] generating court overlay visualization")
        try:
            # Import and run visualization
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from scripts.visualizations import process_video as viz_process_video
            
            # Prepare visualizations: avoid copying masks when unnecessary.
            import scripts.visualizations as viz_module
            # Ensure visualizer will find original frames in our temp original_frames_dir
            original_train_path = viz_module.TRAIN_PATH
            viz_module.TRAIN_PATH = original_frames_dir
            viz_module.VIZ_OUT_PATH = run_output_dir

            # Create a folder with masked RGB frames annotated with boxes for inspection
            masked_frames_dir = os.path.join(temp_dir, "masked_frames")
            try:
                viz_module.save_masked_frames_with_boxes(frame_paths, mask_paths, results, masked_frames_dir)
            except Exception as e:
                print(f"[MAIN] warning: saving masked frames failed ({e})")

            # We no longer auto-generate the separate *_viz.mp4. If the user
            # wants the court overlay MP4, run the visualizer separately.

            # Restore original TRAIN_PATH
            viz_module.TRAIN_PATH = original_train_path
            
        except Exception as e:
            print(f"[MAIN] warning: visualization failed ({e}), continuing without MP4")
        
        # Summary output
        # Auto-generate masked_frames folder for this run (uses tracking_results.json)
        try:
            from scripts.visualizations import create_masked_frames_from_run
            masked_dir = create_masked_frames_from_run(run_output_dir)
            print(f"[MAIN] masked frames created: {masked_dir}")
        except Exception as e:
            print(f"[MAIN] warning: masked frames creation failed ({e})")

        print(f"[MAIN] processed_frames={len(frame_paths)}")
        print(f"[MAIN] saved_tracking={output_path}")
        print(f"[MAIN] saved_rally_data={rally_output_path}")
        print(f"[MAIN] saved_stats={stats_output_path}")
        print(f"[MAIN] run_complete dir={run_output_dir}")
    
    finally:
        # Clean up temp directory if desired (optional)
        # shutil.rmtree(temp_dir, ignore_errors=True)
        pass


print(__name__)
if __name__ == "__main__":
    
    main()
