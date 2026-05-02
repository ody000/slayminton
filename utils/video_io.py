"""Video I/O helper module.
Handles frame extraction, batching, and output writing."""

import cv2
import os
from typing import List
import numpy as np


def extract_frames(video_path: str, output_dir: str, fps: int = 30) -> List[str]:
    """
    Extract frames from an MP4 video and save them to output_dir.
    
    Args:
        video_path: Path to input MP4 file
        output_dir: Directory to save extracted frames
        fps: Target frames per second (downsample if video has higher fps)
    
    Returns:
        List of saved frame paths in sorted order
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    
    # Frame skip to match target fps
    skip_interval = max(1, int(video_fps / fps))
    frame_idx = 0
    saved_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip_interval == 0:
            # Save as 6-digit zero-padded filename
            frame_num = len(saved_frames)
            out_path = os.path.join(output_dir, f"{frame_num:06d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_frames.append(out_path)
        
        frame_idx += 1
    
    cap.release()
    print(f"[VIDEO_IO] extracted {len(saved_frames)} frames from {video_path}")
    return sorted(saved_frames)


def apply_mog2_to_frames(frame_paths: List[str], output_mask_dir: str, fps: int = 30) -> List[str]:
    """
    Apply MOG2 background subtraction to frames and save masks.
    
    Args:
        frame_paths: List of original frame paths
        output_mask_dir: Directory to save mask frames
        fps: Frames per second (used for MOG2 history)
    
    Returns:
        List of saved mask frame paths in same order as input
    """
    os.makedirs(output_mask_dir, exist_ok=True)
    
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg.setVarThreshold(200)
    fgbg.setHistory(1000)
    
    mask_paths = []
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[MOG2] warning: cannot read {frame_path}")
            continue
        
        fg_mask = fgbg.apply(frame)
        mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        # Save mask with same filename as original
        mask_path = os.path.join(output_mask_dir, os.path.basename(frame_path))
        cv2.imwrite(mask_path, mask_3ch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mask_paths.append(mask_path)
        
        if (i + 1) % max(1, int(fps)) == 0 or (i + 1) == len(frame_paths):
            print(f"[MOG2] frame {i + 1}/{len(frame_paths)}", end="\r")
    
    print(f"\n[MOG2] generated {len(mask_paths)} mask frames")
    return mask_paths
