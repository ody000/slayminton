"""Data augmentation script: generate horizontally reflected training images and annotations.

This script augments the training dataset by creating horizontally flipped versions
of all training images and updating the COCO annotations accordingly.

Usage:
    python utils/augment_data_reflect.py [--input_dir PATH] [--output_dir PATH]
    
Or from repo root:
    python -m utils.augment_data_reflect

The script will:
1. Load COCO annotations from the input directory
2. For each image, create a horizontally flipped version
3. Update bounding box x-coordinates for reflected images
4. Save augmented dataset with updated COCO JSON to output directory

Note: This effectively doubles the dataset size.
"""

import json
import os
import argparse
from typing import Dict, List
from pathlib import Path

from PIL import Image
import numpy as np


def load_coco_json(annotations_file: str) -> dict:
    """Load COCO JSON file."""
    with open(annotations_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_coco_json(coco_data: dict, output_file: str) -> None:
    """Save COCO JSON file."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)


def augment_dataset_with_reflection(
    input_dir: str,
    output_dir: str,
    annotations_file: str = "_annotations.coco.json",
) -> None:
    """Augment dataset by creating horizontally flipped versions of all images.

    Args:
        input_dir: Input directory with images and _annotations.coco.json
        output_dir: Output directory for augmented images and annotations
        annotations_file: Name of COCO JSON file (default: _annotations.coco.json)
    """
    input_annotations_path = os.path.join(input_dir, annotations_file)
    output_annotations_path = os.path.join(output_dir, annotations_file)

    if not os.path.exists(input_annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {input_annotations_path}")

    # Load original COCO data
    coco_data = load_coco_json(input_annotations_path)
    original_images = coco_data.get("images", [])
    original_annotations = coco_data.get("annotations", [])

    print(f"[AUGMENT] Loading {len(original_images)} images from {input_dir}")
    print(f"[AUGMENT] Loading {len(original_annotations)} annotations")

    os.makedirs(output_dir, exist_ok=True)

    # Track image and annotation IDs
    next_image_id = max([img["id"] for img in original_images], default=0) + 1
    next_ann_id = max([ann["id"] for ann in original_annotations], default=0) + 1

    augmented_images = []
    augmented_annotations = []

    # Process each original image
    for idx, img_info in enumerate(original_images):
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = os.path.join(input_dir, file_name)

        if not os.path.exists(img_path):
            print(f"[AUGMENT] WARN: Image not found {img_path}, skipping")
            continue

        # Create output subdirectory if needed
        output_subdir = os.path.dirname(os.path.join(output_dir, file_name))
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)

        try:
            # Load and save original image
            image = Image.open(img_path).convert("RGB")
            image.save(output_path)

            # Add original image to augmented list
            augmented_images.append(img_info)

            # Create reflected version
            img_width, img_height = image.size
            img_width = float(img_width)
            img_height = float(img_height)
            reflected_image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Generate reflected file name
            name_base, ext = os.path.splitext(file_name)
            reflected_file_name = f"{name_base}_reflected{ext}"
            reflected_output_path = os.path.join(output_dir, reflected_file_name)

            # Create output subdirectory for reflected image
            reflected_output_subdir = os.path.dirname(reflected_output_path)
            os.makedirs(reflected_output_subdir, exist_ok=True)

            # Save reflected image
            reflected_image.save(reflected_output_path)

            # Add reflected image metadata
            reflected_img_info = img_info.copy()
            reflected_img_info["id"] = next_image_id
            reflected_img_info["file_name"] = reflected_file_name
            augmented_images.append(reflected_img_info)

            # Get annotations for original image and create reflected versions
            img_annotations = [ann for ann in original_annotations if ann["image_id"] == image_id]

            # Add original annotations
            for ann in img_annotations:
                augmented_annotations.append(ann)

            # Create and add reflected annotations
            for ann in img_annotations:
                reflected_ann = ann.copy()
                reflected_ann["id"] = next_ann_id
                reflected_ann["image_id"] = next_image_id

                # Flip bounding box x-coordinate
                bbox = reflected_ann.get("bbox", None)
                if bbox and len(bbox) == 4:
                    x, y, w, h = [float(v) for v in bbox]
                    # Flip x-coordinate: new_x = img_width - x - w
                    reflected_bbox = [img_width - x - w, y, w, h]
                    reflected_ann["bbox"] = reflected_bbox

                augmented_annotations.append(reflected_ann)
                next_ann_id += 1

            next_image_id += 1

            if (idx + 1) % 100 == 0:
                print(f"[AUGMENT] Processed {idx + 1}/{len(original_images)} images...")

        except Exception as e:
            print(f"[AUGMENT] ERROR processing {img_path}: {e}")
            continue

    # Update COCO data with augmented images and annotations
    coco_data["images"] = augmented_images
    coco_data["annotations"] = augmented_annotations

    # Save updated COCO JSON
    save_coco_json(coco_data, output_annotations_path)

    print(f"\n[AUGMENT] Complete!")
    print(f"  Original images: {len(original_images)}")
    print(f"  Augmented images: {len(augmented_images)}")
    print(f"  Original annotations: {len(original_annotations)}")
    print(f"  Augmented annotations: {len(augmented_annotations)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output annotations: {output_annotations_path}")


def main():
    # Compute repo root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(
        description="Augment training data with horizontally reflected images"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(repo_root, "data/input/train_mog_frames"),
        help="Input directory with images and COCO JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(repo_root, "data/input/train_mog_reflect"),
        help="Output directory for augmented dataset",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="_annotations.coco.json",
        help="Name of COCO annotations file (default: _annotations.coco.json)",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        exit(1)

    print(f"[AUGMENT] Starting data augmentation")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print()

    augment_dataset_with_reflection(
        input_dir,
        output_dir,
        annotations_file=args.annotations_file,
    )


if __name__ == "__main__":
    main()
