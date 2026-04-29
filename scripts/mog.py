"""
Generate binary motion mask frames + videos using BackgroundSubtractorMOG2.

All JPGs live flat in data/input/train/. Filenames follow the pattern:
    <video_id>-<frame_number>_jpg.rf.<hash>.jpg
    e.g. 00011_mp4-103_jpg.rf.2XThjXxkBSFahPBmAmHp.jpg

Files are grouped by <video_id>, each group processed as one video sequence.

Outputs:
  - Frames: data/input/train_mog_frames/<video_id>/<original_filename>.jpg
  - Videos: data/input/train_mog_mp4s/<video_id>.mp4
"""

import cv2
import os
import glob
import json
from collections import defaultdict

TRAIN_PATH      = "data/input/train"
FRAMES_OUT_PATH = "data/input/train_mog_frames"
VIDEOS_OUT_PATH = "data/input/train_mog_mp4s"
DEFAULT_FPS     = 30
JPEG_QUALITY    = 95


def group_frames_by_video(folder: str) -> dict:
    """
    Group all JPGs in folder by their video-id prefix.
    Filename pattern: <video_id>-<frame_num>_jpg.rf.<hash>.jpg
    Returns {video_id: [sorted frame paths...]}.
    """
    groups = defaultdict(list)

    for fp in glob.glob(os.path.join(folder, "*.jpg")):
        name = os.path.basename(fp)
        # Everything before the first '-' is the video id
        # e.g. "00011_mp4-103_jpg.rf.hash.jpg" -> video_id = "00011_mp4"
        dash_idx = name.find("-")
        if dash_idx == -1:
            video_id  = "unknown"
            frame_num = 0
        else:
            video_id  = name[:dash_idx]
            remainder = name[dash_idx + 1:]   # "103_jpg.rf.hash.jpg"
            try:
                frame_num = int(remainder.split("_")[0])
            except ValueError:
                frame_num = 0

        groups[video_id].append((frame_num, fp))

    # Sort each group by frame number
    return {
        vid: [fp for _, fp in sorted(frames)]
        for vid, frames in sorted(groups.items())
    }


def process_video(video_id: str, frame_paths: list, fps: int = DEFAULT_FPS) -> None:
    print(f"\n[{video_id}]  {len(frame_paths)} frames")

    first = cv2.imread(frame_paths[0])
    if first is None:
        print(f"  [skip] Cannot read first frame")
        return
    h, w = first.shape[:2]
    print(f"  size: {w}x{h}  fps: {fps}")

    # Output dirs
    frame_out_dir = os.path.join(FRAMES_OUT_PATH, video_id)
    os.makedirs(frame_out_dir, exist_ok=True)
    os.makedirs(VIDEOS_OUT_PATH, exist_ok=True)

    video_out_path = os.path.join(VIDEOS_OUT_PATH, f"{video_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))

    # Fresh MOG2 instance per video
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg.setVarThreshold(200)
    fgbg.setHistory(1000)

    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            print(f"\n  [warn] Cannot read {fp}, skipping")
            continue

        fg_mask  = fgbg.apply(frame)
        mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        out_frame_path = os.path.join(frame_out_dir, os.path.basename(fp))
        cv2.imwrite(out_frame_path, mask_3ch, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        writer.write(mask_3ch)

        print(f"  frame {i+1}/{len(frame_paths)}", end="\r")

    writer.release()
    print(f"\n  ✓ video  -> {video_out_path}")
    print(f"  ✓ frames -> {frame_out_dir}/")

def coco_mog2():
    with open("data/input/train_mog_frames/_annotations.coco.json") as f:
        coco = json.load(f)

        for img in coco["images"]:
            name = img["file_name"]

            # extract video_id (same logic as your script)
            dash_idx = name.find("-")
            if dash_idx != -1:
                video_id = name[:dash_idx]
            else:
                video_id = "unknown"

            img["file_name"] = os.path.join(video_id, name)

    with open("_annotations.coco.json", "w") as f:
        json.dump(coco, f)

def main() -> None:
    groups = group_frames_by_video(TRAIN_PATH)

    if not groups:
        print(f"No JPGs found in {TRAIN_PATH}")
        return

    total_frames = sum(len(v) for v in groups.values())
    print(f"Found {len(groups)} video(s) across {total_frames} frames\n")

    for video_id, frame_paths in groups.items():
        process_video(video_id, frame_paths)

    print("\n\nAll done.")
    print(f"  Frames -> {FRAMES_OUT_PATH}/")
    print(f"  Videos -> {VIDEOS_OUT_PATH}/")



if __name__ == "__main__":
    coco_mog2()