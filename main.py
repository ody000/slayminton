"""
Entry point for the Slayminton application.
The main execution loop implemented here.
"""

import argparse
from datetime import datetime
import json
import os
from typing import List

import numpy as np
from PIL import Image
import torch

from core.analysis import Analysis
from core.game_state import GameState
from models.dino import DINODataset, DINOTracker, train_dino


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
    parser.add_argument("--mode", choices=["train", "track-frames"], default="train")
    parser.add_argument("--train-dir", default="data/input/train")
    parser.add_argument("--annotations", default="data/input/train/_annotations.coco.json")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--weights", default="data/output/dino_tracker.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--frames-dir",
        default="data/input/train",
        help="Directory containing RGB image frames for dry-run tracking mode",
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
        )

        print("[MAIN] training_complete")
        print(f"Saved student checkpoint: {os.path.join(args.output_dir, os.path.basename(args.weights))}")
        print(f"Last train loss: {history.train_loss[-1]:.4f}")
        if history.val_iou:
            print(f"Last val IoU: {history.val_iou[-1]:.4f} | Last val mAP@0.75: {history.val_map75[-1]:.4f}")
        return


    # Actual tracking starts here (if not training mode).
    # Create one folder per input source/run under data/output.
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


if __name__ == "__main__":
    print(__name__)
    main()
