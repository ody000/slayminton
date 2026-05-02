"""Diagnostic script to verify ground truth bounding box correctness.

This script mimics the preprocessing logic from models/dino.py (DINODataset)
but instead of training, it visualizes ground truth boxes on images.

Usage:
    python diagnostic.py [--input_dir PATH] [--output_dir PATH] [--max_images N]

Output:
    Saves annotated images to /data/input/train_diagnostic/ showing:
    - Yellow bounding boxes for player (person)
    - Cyan bounding boxes for shuttle
    - Class name and confidence (1.0 for GT) as labels
"""

import json
import os
import argparse
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# COCO class mapping (matching dino.py)
COCO_NAME_TO_TRACK = {
    "person": "player",
    "player": "player",
    "shuttle": "shuttle",
    "shuttlecock": "shuttle",
}

TRACKED_CLASSES = ("player", "shuttle")
CLASS_COLORS = {
    "player": (0, 255, 0),  # Green (lime)
    "shuttle": (0, 255, 255),  # Cyan
}


def load_coco_annotations(annotations_file: str) -> Tuple[Dict, List[dict], Dict[int, List[dict]]]:
    """Load COCO JSON and extract metadata.

    Returns:
        (categories_dict, images_list, annotations_by_image_dict)
    """
    with open(annotations_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}
    images = sorted(coco.get("images", []), key=lambda x: x["id"])

    annotations_by_image: Dict[int, List[dict]] = {im["id"]: [] for im in images}
    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(ann)

    return categories, images, annotations_by_image


def get_representative_boxes(
    anns: List[dict],
    categories: Dict[int, str],
    img_w: float,
    img_h: float,
) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
    """Extract one representative box per tracked class (largest by area).

    Returns:
        Dict mapping class name ("player", "shuttle") to (x, y, w, h) in original image coords.
        If no annotation for a class, value is None.
    """
    buckets: Dict[str, List[Tuple[float, float, float, float]]] = {k: [] for k in TRACKED_CLASSES}

    for ann in anns:
        cat_id = ann.get("category_id")
        cname = categories.get(cat_id, "")
        mapped = COCO_NAME_TO_TRACK.get(cname)
        if mapped is None:
            continue

        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue

        x, y, w, h = [float(v) for v in bbox]
        # Clamp to image bounds
        x = max(0.0, min(x, img_w - 1.0))
        y = max(0.0, min(y, img_h - 1.0))
        w = max(0.0, min(w, img_w - x))
        h = max(0.0, min(h, img_h - y))
        buckets[mapped].append((x, y, w, h))

    out = {}
    for key, vals in buckets.items():
        if not vals:
            out[key] = None
        else:
            # Keep the largest box by area
            out[key] = max(vals, key=lambda b: b[2] * b[3])

    return out


def draw_boxes_on_image(
    image: Image.Image,
    boxes: Dict[str, Optional[Tuple[float, float, float, float]]],
    line_width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """Draw bounding boxes and labels on a PIL image.

    Args:
        image: PIL Image to annotate
        boxes: Dict mapping class name to (x, y, w, h) or None
        line_width: Thickness of bounding box lines
        font_size: Font size for labels

    Returns:
        Annotated PIL Image
    """
    draw = ImageDraw.Draw(image)

    # Try to use a nicer font; fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    for class_name in TRACKED_CLASSES:
        box = boxes.get(class_name)
        if box is None:
            continue

        x, y, w, h = box
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label with class name and confidence (GT is always 1.0)
        label = f"{class_name}: 1.0"
        bbox_text = draw.textbbox((x1, max(0, y1 - font_size - 4)), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        # Draw background for text
        draw.rectangle(
            [x1, max(0, y1 - text_height - 4), x1 + text_width + 4, max(0, y1)],
            fill=color,
        )
        # Draw text
        draw.text((x1 + 2, max(0, y1 - text_height - 2)), label, fill=(0, 0, 0), font=font)

    return image


def process_dataset(
    data_dir: str,
    annotations_file: str,
    output_dir: str,
    max_images: Optional[int] = None,
) -> None:
    """Process all (or a subset of) images in the dataset and save annotated versions.

    Args:
        data_dir: Directory containing images (e.g., train_mog_frames)
        annotations_file: Path to _annotations.coco.json
        output_dir: Output directory for annotated images
        max_images: If set, only process this many images. If None, process all.
    """
    os.makedirs(output_dir, exist_ok=True)

    categories, images, annotations_by_image = load_coco_annotations(annotations_file)

    print(f"[DIAGNOSTIC] Loading annotations from {annotations_file}")
    print(f"[DIAGNOSTIC] Found {len(images)} images")
    print(f"[DIAGNOSTIC] Categories: {categories}")

    # Optionally limit number of images processed
    if max_images is not None and max_images > 0:
        images = images[:max_images]
        print(f"[DIAGNOSTIC] Processing subset of {len(images)} images (max_images={max_images})")

    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, image_info in enumerate(images):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        img_path = os.path.join(data_dir, file_name)

        # Create output subdirectory structure matching input
        output_subdir = os.path.dirname(os.path.join(output_dir, file_name))
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)

        # Load image
        if not os.path.exists(img_path):
            print(f"[DIAGNOSTIC] WARN: Image not found {img_path}")
            skip_count += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[DIAGNOSTIC] ERROR loading image {img_path}: {e}")
            error_count += 1
            continue

        orig_w, orig_h = image.size

        # Get annotations for this image
        anns = annotations_by_image.get(image_id, [])
        boxes = get_representative_boxes(anns, categories, orig_w, orig_h)

        # Draw boxes on image
        try:
            annotated_image = draw_boxes_on_image(image, boxes)
            annotated_image.save(output_path)
            success_count += 1

            if (idx + 1) % 50 == 0:
                print(f"[DIAGNOSTIC] Processed {idx + 1}/{len(images)} images...")

        except Exception as e:
            print(f"[DIAGNOSTIC] ERROR annotating/saving {img_path}: {e}")
            error_count += 1

    print(f"\n[DIAGNOSTIC] Complete!")
    print(f"  Processed: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {output_dir}")

    # Summary statistics
    print(f"\n[DIAGNOSTIC] Summary Statistics:")
    total_boxes = {cls: 0 for cls in TRACKED_CLASSES}
    for image_id, anns in annotations_by_image.items():
        for ann in anns:
            cat_id = ann.get("category_id")
            cname = categories.get(cat_id, "")
            mapped = COCO_NAME_TO_TRACK.get(cname)
            if mapped is not None:
                total_boxes[mapped] += 1

    for cls_name in TRACKED_CLASSES:
        print(f"  Total {cls_name} annotations: {total_boxes[cls_name]}")

    if success_count > 0:
        print(f"\n[DIAGNOSTIC] Example output files:")
        output_files = []
        for root, dirs, files in os.walk(output_dir):
            for f in files[:3]:  # Show first 3 files as examples
                output_files.append(os.path.join(root, f))
        for f in output_files:
            print(f"    - {os.path.relpath(f, output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic script to visualize ground truth bounding boxes"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/input/train_mog_frames",
        help="Directory containing annotated images (default: data/input/train_mog_frames)",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default=None,
        help="Path to COCO annotations JSON (default: input_dir/_annotations.coco.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/input/train_diagnostic",
        help="Output directory for annotated images (default: data/input/train_diagnostic)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: None = process all)",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    # Auto-detect annotations file if not provided
    if args.annotations_file is None:
        annotations_file = os.path.join(input_dir, "_annotations.coco.json")
    else:
        annotations_file = os.path.abspath(args.annotations_file)

    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        exit(1)

    if not os.path.exists(annotations_file):
        print(f"ERROR: Annotations file not found: {annotations_file}")
        exit(1)

    print(f"[DIAGNOSTIC] Starting diagnostic run")
    print(f"  Input directory: {input_dir}")
    print(f"  Annotations file: {annotations_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max images: {args.max_images or 'all'}")
    print()

    process_dataset(input_dir, annotations_file, output_dir, max_images=args.max_images)


if __name__ == "__main__":
    main()
