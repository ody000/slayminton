

"""
Visualization: Dual-player footwork heatmap on a top-down court view.

Changes from previous version:
  - Track player FEET (bottom of bounding box) instead of centroid for accurate mapping
  - Added bounding box persistence (memory) to stop boxes flickering/disappearing
  - Stable P1/P2 ID assignment via nearest-previous-position
  - Single combined court-view insert instead of two separate corner boxes
  - Court drawn to real badminton proportions (13.4 m × 6.1 m) with proper lines

Outputs:
  data/input/train_viz_mp4s/<video_id>_viz.mp4
"""

import cv2
import numpy as np
import os
import glob
import json

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_PATH       = "data/input/train"
MASK_FRAMES_PATH = "data/input/train_mog_frames"
VIZ_OUT_PATH     = "data/input/train_viz_mp4s"
STORED_COURT_POINTS_PATH = "data/input/court_points.json"

# ── Detection thresholds & Tracking ──────────────────────────────────────────
MIN_BLOB_AREA   = 30
PLAYER_MIN_AREA = 1500
MAX_LOST_FRAMES = 15    # How many frames to keep the box alive if detection is lost

# ── Court-view insert ─────────────────────────────────────────────────────────
# Real court: 13.4 m long × 6.1 m wide (doubles)
COURT_LEN_M = 13.4
COURT_WID_M = 6.1
INSERT_H    = 300                                              # long axis = court length
INSERT_W    = int(INSERT_H * COURT_WID_M / COURT_LEN_M)        # ≈ 137 px → correct aspect
COURT_PAD   = 14        # pixel margin around court lines inside the insert

# Heatmap smoothing (in court-insert pixel space)
HM_BLUR      = 7        # → (15×15) Gaussian kernel
PLAYER_STAMP = 6        # footprint radius in court-insert pixels

INSERT_ALPHA = 0.82     # overlay opacity

# Court corners in VIDEO pixel coords: [TL, TR, BR, BL] clockwise.
# When set, a homography maps player positions to a true top-down view.
# Leave as None to use a simple linear (full-frame) mapping instead.
# Example: COURT_CORNERS_VIDEO = [(110,35), (610,35), (610,445), (110,445)] THIS NOW REQUIRES TWO MORE TUPLES FOR THE BOTTOM AND TOP OF THE MIDLINE
# COURT_CORNERS_VIDEO = None
# THIS IS NO LONGER NEEDED THEORETICALLY

# ── Per-player colours (BGR) ──────────────────────────────────────────────────
P1_COLOR = (57,  255,  20)   # neon green
P2_COLOR = (0,   165, 255)   # orange

# ── Video ─────────────────────────────────────────────────────────────────────
DEFAULT_FPS = 30

# ── Booleans ─────────────────────────────────────────────────────────────────────
SET_COURT_POINTS = False # turn true if you want to set all the court points manually even those that are already saved

# ─────────────────────────────────────────────────────────────────────────────
# Court geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

# Drawable court area bounds inside the insert (in pixels)
_CX0 = COURT_PAD
_CX1 = INSERT_W - COURT_PAD
_CY0 = COURT_PAD
_CY1 = INSERT_H - COURT_PAD
_CMX = (_CX1 - _CX0) / 2.0
_CW  = _CX1 - _CX0   # court pixel width
_CH  = _CY1 - _CY0   # court pixel height


def m_to_px(mx: float, my: float) -> tuple[int, int]:
    """Convert real court metres (x across, y along) → insert pixel coords."""
    px = _CX0 + int(mx * _CW / COURT_WID_M)
    py = _CY0 + int(my * _CH / COURT_LEN_M)
    return px, py


def draw_court_background() -> np.ndarray:
    """
    Return a fresh INSERT_H × INSERT_W BGR image with a top-down badminton
    court drawn to regulation proportions.
    """
    img = np.zeros((INSERT_H, INSERT_W, 3), dtype=np.uint8)
    img[:] = (28, 60, 28)   # dark green background

    WHITE  = (220, 220, 220)
    THIN   = 1

    def hline(y_m):
        y = m_to_px(0, y_m)[1]
        cv2.line(img, (_CX0, y), (_CX1, y), WHITE, THIN)

    def vline(x_m, y0_m=0.0, y1_m=COURT_LEN_M):
        x  = m_to_px(x_m, 0)[0]
        y0 = m_to_px(0, y0_m)[1]
        y1 = m_to_px(0, y1_m)[1]
        cv2.line(img, (x, y0), (x, y1), WHITE, THIN)

    # Outer doubles boundary
    cv2.rectangle(img, (_CX0, _CY0), (_CX1, _CY1), WHITE, THIN)

    # Singles sidelines
    vline(0.46)
    vline(5.64)

    # Net
    net_y = m_to_px(0, 6.7)[1]
    cv2.line(img, (_CX0, net_y), (_CX1, net_y), WHITE, 2)

    # Short service lines
    hline(4.72)
    hline(8.68)

    # Long service lines (doubles)
    hline(0.76)
    hline(12.64)

    # Centre line (within singles box, full length)
    vline(3.05, 0.0, COURT_LEN_M)

    # Net label
    cv2.putText(img, "NET", (m_to_px(2.6, 6.7)[0], net_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1, cv2.LINE_AA)

    # Thin outer border
    cv2.rectangle(img, (0, 0), (INSERT_W - 1, INSERT_H - 1), (90, 90, 90), 1)

    return img


def compute_homography(frame_h: int, frame_w: int, court_corners) -> np.ndarray | None:
    # """
    # If COURT_CORNERS_VIDEO is set, compute a homography that maps video
    # pixel coords → court-insert pixel coords.
    # """
    # if COURT_CORNERS_VIDEO is None:
    #     return None
    """
    compute a homography that maps video pixel coords → court-insert pixel coords.
    """
    src = np.array(court_corners, dtype=np.float32)
    dst = np.array([
        [_CX0, _CY0], # bottom left
        [_CX1, _CY0], # bottom right
        [_CX1, _CY1], # top right
        [_CX0, _CY1], # top left
        [_CMX, _CY0], # bottom of the midline
        [_CMX, _CY1] # top of the midline
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def video_to_insert(cx: int, cy: int, frame_w: int, frame_h: int,
                    H: np.ndarray | None) -> tuple[int, int]:
    """Map a video-pixel position to court-insert pixel coords."""
    if H is not None:
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]
        ix = int(np.clip(mapped[0], 0, INSERT_W - 1))
        iy = int(np.clip(mapped[1], 0, INSERT_H - 1))
    else:
        ix = _CX0 + int(cx * _CW / frame_w)
        iy = _CY0 + int(cy * _CH / frame_h)
        ix = int(np.clip(ix, 0, INSERT_W - 1))
        iy = int(np.clip(iy, 0, INSERT_H - 1))
    return ix, iy


# ─────────────────────────────────────────────────────────────────────────────
# Detection & stable ID assignment
# ─────────────────────────────────────────────────────────────────────────────

def detect_players(mask_gray: np.ndarray) -> list:
    """
    Return the top-2 largest blobs ≥ PLAYER_MIN_AREA as
    [(area, cx, cy, (x,y,w,h)), ...] sorted descending by area.
    """
    contours, _ = cv2.findContours(
        mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    players = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < PLAYER_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Use BOTTOM-CENTER (feet) instead of centroid for ground plane accuracy
        cx = x + w // 2
        cy = y + h
        players.append((area, cx, cy, (x, y, w, h)))

    players.sort(key=lambda b: b[0], reverse=True)
    return players[:2]


def _sq_dist(cx: int, cy: int, prev: tuple | None) -> float:
    if prev is None:
        return float("inf")
    return float((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2)


def assign_players_stable(
    blobs: list,
    prev_p1: tuple | None,
    prev_p2: tuple | None,
) -> tuple:
    """
    Assign blobs to P1 / P2 using nearest-previous-position matching.
    Falls back to X-sort on the very first frame (both prev are None).
    Returns (p1_blob_or_None, p2_blob_or_None).
    """
    if not blobs:
        return None, None

    # First frame: no history — assign left→P1, right→P2
    if prev_p1 is None and prev_p2 is None:
        s = sorted(blobs, key=lambda b: b[1])
        return (s[0] if len(s) > 0 else None,
                s[1] if len(s) > 1 else None)

    if len(blobs) == 1:
        _, cx, cy, _ = blobs[0]
        d1 = _sq_dist(cx, cy, prev_p1)
        d2 = _sq_dist(cx, cy, prev_p2)
        return (blobs[0], None) if d1 <= d2 else (None, blobs[0])

    # Two blobs: pick the permutation with lower total distance cost
    b0, b1 = blobs[0], blobs[1]
    _, cx0, cy0, _ = b0
    _, cx1, cy1, _ = b1

    cost_straight = _sq_dist(cx0, cy0, prev_p1) + _sq_dist(cx1, cy1, prev_p2)
    cost_swap     = _sq_dist(cx1, cy1, prev_p1) + _sq_dist(cx0, cy0, prev_p2)

    return (b0, b1) if cost_straight <= cost_swap else (b1, b0)


# ─────────────────────────────────────────────────────────────────────────────
# Court-view insert builder
# ─────────────────────────────────────────────────────────────────────────────

def build_court_insert(
    court_base: np.ndarray,
    hm_p1: np.ndarray,
    hm_p2: np.ndarray,
    p1_ipos: tuple | None,
    p2_ipos: tuple | None,
) -> np.ndarray:
    """
    Composite the two heatmaps onto a copy of the court background.
    P1 → red channel  |  P2 → blue channel
    Current positions marked with coloured dots.
    """
    canvas = court_base.copy()
    blur   = HM_BLUR * 2 + 1

    def overlay_hm(hm: np.ndarray, channel: int, color_bgr: tuple) -> None:
        if hm.max() == 0:
            return
        blurred = cv2.GaussianBlur(hm, (blur, blur), 0)
        norm    = cv2.normalize(blurred, None, 0.0, 1.0, cv2.NORM_MINMAX)
        # Blend into matching channel + a faint tint in the others for contrast
        for ch, strength in enumerate([0.15, 0.15, 0.15]):
            strength = 0.85 if ch == channel else 0.10
            canvas[:, :, ch] = np.clip(
                canvas[:, :, ch].astype(np.float32) + norm * 180 * strength,
                0, 255
            ).astype(np.uint8)

    # P1 → red (channel 2), P2 → blue (channel 0)
    overlay_hm(hm_p1, 2, P1_COLOR)
    overlay_hm(hm_p2, 0, P2_COLOR)

    # Current player dots
    for pos, color, label in [
        (p1_ipos, P1_COLOR, "P1"),
        (p2_ipos, P2_COLOR, "P2"),
    ]:
        if pos is not None:
            cv2.circle(canvas, pos, 5, color,         -1, cv2.LINE_AA)
            cv2.circle(canvas, pos, 5, (0, 0, 0),      1, cv2.LINE_AA)
            cv2.putText(canvas, label,
                        (pos[0] + 6, pos[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

    # Legend
    cv2.putText(canvas, "Footwork", (4, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "P1", (INSERT_W - 22, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, P1_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "P2", (INSERT_W - 10, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, P2_COLOR, 1, cv2.LINE_AA)

    return canvas


def paste_insert_bottom_right(frame: np.ndarray, insert: np.ndarray, alpha: float) -> None:
    H, W = frame.shape[:2]
    roi = frame[H - INSERT_H: H, W - INSERT_W: W]
    cv2.addWeighted(insert, alpha, roi, 1 - alpha, 0, roi)


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def group_mask_frames(mask_root: str) -> dict[str, list[str]]:
    groups = {}
    for vid in sorted(os.listdir(mask_root)):
        folder = os.path.join(mask_root, vid)
        if not os.path.isdir(folder):
            continue
        paths = sorted(
            glob.glob(os.path.join(folder, "*.jpg")),
            key=lambda p: _frame_num(os.path.basename(p))
        )
        if paths:
            groups[vid] = paths
    return groups


def _frame_num(name: str) -> int:
    dash = name.find("-")
    if dash == -1:
        return 0
    try:
        return int(name[dash + 1:].split("_")[0])
    except ValueError:
        return 0


def orig_path_for(mask_path: str) -> str:
    return os.path.join(TRAIN_PATH, os.path.basename(mask_path))


# ─────────────────────────────────────────────────────────────────────────────
# Setting court points
# ─────────────────────────────────────────────────────────────────────────────

def set_court_points(first_orig, video_id):
    points = []
    first_frame = first_orig.copy()

    labels = [
        "Bottom-Left",
        "Bottom-Right",
        "Top-Right",
        "Top-Left",
        "Midline Bottom",
        "Midline Top"
    ]
    
    def redraw():
        first_frame = first_orig.copy()
        for p in points:
            cv2.circle(first_frame, p, 5, (0, 255, 0), -1)

        if len(points) < 6:
            cv2.putText(first_frame, f"Click: {labels[len(points)]}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)
            
        cv2.imshow("frame", first_frame)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            redraw()

    redraw()
    cv2.setMouseCallback("frame", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('z'):  # undo
            if points:
                points.pop()
                redraw()

        # exit if 6 points collected
        if len(points) == 6:
            break

        # optional: press 'q' to quit early
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    print("Collected points:", points)

    # --- LOAD existing data ---
    if os.path.exists(STORED_COURT_POINTS_PATH):
        with open(STORED_COURT_POINTS_PATH, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # --- UPDATE this video ---
    all_data[video_id] = points

    # --- SAVE back ---
    with open(STORED_COURT_POINTS_PATH, "w") as f:
        json.dump(all_data, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# Per-video processing
# ─────────────────────────────────────────────────────────────────────────────

def process_video(video_id: str, mask_paths: list[str], fps: int = DEFAULT_FPS) -> None:
    print(f"\n[{video_id}]  {len(mask_paths)} frames")

    first_orig = cv2.imread(orig_path_for(mask_paths[0]))
    if first_orig is None:
        print("  [skip] Cannot read original frame")
        return
    H, W = first_orig.shape[:2]
    print(f"  size: {W}×{H}  fps: {fps}")
    print(f"  court insert: {INSERT_W}×{INSERT_H} px  (ratio {INSERT_W/INSERT_H:.3f}, "
          f"real {COURT_WID_M}/{COURT_LEN_M}={COURT_WID_M/COURT_LEN_M:.3f})")
    
    # if there is no saved data for the court points, set them manually
    if os.path.exists(STORED_COURT_POINTS_PATH):
        with open(STORED_COURT_POINTS_PATH, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}
    if video_id not in all_data or SET_COURT_POINTS:
        set_court_points(first_orig, video_id)

    # reload json and set COURT_CORNERS_VIDEO
    if os.path.exists(STORED_COURT_POINTS_PATH):
        with open(STORED_COURT_POINTS_PATH, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}
    court_corners = None
    if video_id in all_data:
        court_corners = np.array(all_data[video_id], dtype=np.float32)
    
    homography  = compute_homography(H, W, court_corners)
    court_base  = draw_court_background()

    # Heatmap accumulators in court-INSERT pixel space (not full-frame)
    hm_p1 = np.zeros((INSERT_H, INSERT_W), dtype=np.float32)
    hm_p2 = np.zeros((INSERT_H, INSERT_W), dtype=np.float32)

    prev_p1_pos: tuple | None = None
    prev_p2_pos: tuple | None = None
    
    # State tracking to prevent boxes disappearing
    last_p1_blob = None
    last_p2_blob = None
    p1_lost_count = 0
    p2_lost_count = 0

    os.makedirs(VIZ_OUT_PATH, exist_ok=True)
    out_path = os.path.join(VIZ_OUT_PATH, f"{video_id}_viz.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for i, mask_p in enumerate(mask_paths):
        orig_p   = orig_path_for(mask_p)
        frame    = cv2.imread(orig_p)
        mask_bgr = cv2.imread(mask_p)

        if frame is None:
            print(f"\n  [warn] Missing original: {orig_p}")
            continue
        if mask_bgr is None:
            print(f"\n  [warn] Missing mask: {mask_p}")
            writer.write(frame)
            continue

        mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)

        # ── Detect & assign ──────────────────────────────────────────────────
        blobs = detect_players(mask_gray)
        p1, p2 = assign_players_stable(blobs, prev_p1_pos, prev_p2_pos)

        # ── Handle Persistence (Memory) ──────────────────────────────────────
        # If P1 is missing but we saw them recently, reuse the old bounding box
        if p1 is None and last_p1_blob is not None and p1_lost_count < MAX_LOST_FRAMES:
            p1 = last_p1_blob
            p1_lost_count += 1
        elif p1 is not None:
            last_p1_blob = p1
            p1_lost_count = 0
            
        # Same for P2
        if p2 is None and last_p2_blob is not None and p2_lost_count < MAX_LOST_FRAMES:
            p2 = last_p2_blob
            p2_lost_count += 1
        elif p2 is not None:
            last_p2_blob = p2
            p2_lost_count = 0

        # ── Draw bounding boxes & accumulate heatmaps ────────────────────────
        p1_ipos: tuple | None = None
        p2_ipos: tuple | None = None

        for blob, color, label, hm_acc in [
            (p1, P1_COLOR, "P1", hm_p1),
            (p2, P2_COLOR, "P2", hm_p2),
        ]:
            if blob is None:
                continue
            _, cx, cy, (bx, by, bw, bh) = blob

            # Bounding box on main frame
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(frame, label, (bx, max(by - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Map position to court-insert space and stamp heatmap there
            ix, iy = video_to_insert(cx, cy, W, H, homography)
            cv2.circle(hm_acc, (ix, iy), PLAYER_STAMP, 1.0, -1)

            if label == "P1":
                p1_ipos = (ix, iy)
                prev_p1_pos = (cx, cy)
            else:
                p2_ipos = (ix, iy)
                prev_p2_pos = (cx, cy)

        # ── Build and paste the combined court insert ────────────────────────
        insert = build_court_insert(court_base, hm_p1, hm_p2, p1_ipos, p2_ipos)
        paste_insert_bottom_right(frame, insert, INSERT_ALPHA)

        writer.write(frame)
        print(f"  frame {i + 1}/{len(mask_paths)}", end="\r")

    writer.release()
    print(f"\n  ✓  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    groups = group_mask_frames(MASK_FRAMES_PATH)
    if not groups:
        print(f"No mask frame folders found in {MASK_FRAMES_PATH}")
        return
    print(f"Found {len(groups)} video group(s)\n")
    for video_id, mask_paths in groups.items():
        process_video(video_id, mask_paths)
    print(f"\nAll done.  Videos → {VIZ_OUT_PATH}/")


if __name__ == "__main__":
    main()