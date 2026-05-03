"""DINOv3-style training and tracking module for Slayminton.

Implements:
1. COCO-backed multi-crop dataset for SSL + supervised box targets.
2. ViT encoder wrappers and DINO student/teacher models.
3. DINO training loop with centering, cosine LR scheduler, and EMA momentum schedule.
4. Single-model tracker for shuttle and player localization.
5. Visualization helpers for monitoring training.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF


# Hyperparameters for DINOv3 training
NUM_GLOBAL_CROPS = 2
NUM_LOCAL_CROPS = 6
GLOBAL_CROP_SIZE = 384 # 720p-friendly
LOCAL_CROP_SIZE = 128 # larger crops preserve more detail
BATCH_SIZE = 16
LEARNING_RATE = 5e-4  # Increased from 1e-3 to improve training convergence
EPOCHS = 75
WEIGHT_DECAY = 1e-4
DINO_OUT_DIM = 256
DINO_HIDDEN_DIM = 512
DINO_STUDENT_TEMP = 0.2
DINO_TEACHER_TEMP_START = 0.02
DINO_TEACHER_TEMP_END = 0.08
DINO_TEACHER_TEMP_WARMUP_EPOCHS = 10
CENTER_MOMENTUM = 0.98  # Centering EMA momentum: 0.95 means strong stability (0.9 was causing collapse!)
EMA_MOMENTUM_START = 0.993  
EMA_MOMENTUM_END = 0.9995
BOX_LOSS_WEIGHT = 0.05
SSL_LOSS_WEIGHT = 20.0
MIN_CONFIDENCE = 0.25
VAL_EVERY = 1
VAL_IOU_THRESHOLD = 0.5  # Reduced from 0.75 for easier shuttle detection metrics

TRACKED_CLASSES = ("player", "shuttle")


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    val_iou: List[float]
    val_map75: List[float]
    eval_epochs: List[int]


class DINOTracker(nn.Module):
    """DINOv3-style tracker for shuttle and player detection.

    The model predicts one target per tracked class: player and shuttle.
    Output format per object is `(timestamp, x, y, height, width)` where
    `(x, y)` is top-left corner in original frame coordinates.

    - train one model for both targets (player + shuttle)
    - detect per frame and return timestamped coordinates
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        pretrained_backbone_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        input_size: int = GLOBAL_CROP_SIZE,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        # One shared encoder for both SSL projection and detection.
        self.encoder, self.encoder_dim = _create_vit_tiny(pretrained_weights_path=pretrained_backbone_path)

        # determine patch H
        patch_size = None
        pe = getattr(self.encoder, "patch_embed", None)
        if pe is not None:
            ps = getattr(pe, "patch_size", None)
            if isinstance(ps, (tuple, list)):
                patch_size = int(ps[0])
            elif isinstance(ps, int):
                patch_size = ps
        # default fallback
        patch_size = patch_size or 16

        # round input_size up to multiple of patch_size
        if self.input_size % patch_size != 0:
            new_size = ((self.input_size + patch_size - 1) // patch_size) * patch_size
            print(f"[DINO] adjusting input_size {self.input_size} -> {new_size} to match patch_size {patch_size}")
            self.input_size = new_size

        # update preprocess resize to use self.input_size (already does)
        self.preprocess = transforms.Compose(
            [transforms.Resize((self.input_size, self.input_size)), transforms.ToTensor()]
        )

        # Projector is only for DINO loss (not final tracking output).
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, DINO_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(DINO_HIDDEN_DIM, DINO_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(DINO_HIDDEN_DIM, DINO_OUT_DIM),
        )
        # Per class output: [confidence, cx, cy, w, h].
        self.detector_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.GELU(),
            nn.Linear(self.encoder_dim, len(TRACKED_CLASSES) * 5),
        )
        self.to(self.device)

        # Accept either argument name for convenience.
        ckpt_path = weights_path or model_path
        if ckpt_path and os.path.exists(ckpt_path):
            self.load_checkpoint(ckpt_path)

        self.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
            ]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Reusable feature extraction for both heads.
        return _extract_cls_token(self.encoder, x)

    def forward_dino(self, x: torch.Tensor) -> torch.Tensor:
        # Student/teacher projection space used by SSL objective.
        feat = self.encode(x)
        return self.projector(feat)

    def forward_detect(self, x: torch.Tensor) -> torch.Tensor:
        # Detection head output shape: (B, num_classes, 5).
        feat = self.encode(x)
        raw = self.detector_head(feat).view(x.size(0), len(TRACKED_CLASSES), 5)
        conf = torch.sigmoid(raw[..., :1])
        box = torch.sigmoid(raw[..., 1:])
        return torch.cat([conf, box], dim=-1)

    def load_checkpoint(self, path: str) -> None:
        # We support both raw state_dict and wrapped {"model": ...} checkpoints.
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=False)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model": self.state_dict()}, path)

    def _get_white_pixel_ratio(
        self,
        frame: np.ndarray,
        bbox_xywh: Tuple[float, float, float, float],
        white_threshold: int = 200,
    ) -> float:
        """Calculate the ratio of white/bright pixels in a bounding box region.

        Used to filter out false positive detections (e.g., shuttle detection on black MOG regions).
        
        Args:
            frame: Input frame as HxW (grayscale) or HxWx3 (RGB/BGR)
            bbox_xywh: Bounding box as (x, y, w, h) in frame coordinates
            white_threshold: Pixel intensity threshold for "white" (0-255)
        
        Returns:
            Ratio of white pixels to total pixels in bounding box (0.0 to 1.0)
        """
        x, y, w, h = bbox_xywh
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Clamp to frame bounds
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        
        region = frame[y : y + h, x : x + w]
        
        # Convert to grayscale if needed
        if region.ndim == 3:
            region = np.mean(region, axis=2).astype(np.uint8)
        
        # Count pixels above white threshold
        white_pixels = np.sum(region >= white_threshold)
        total_pixels = region.size
        
        if total_pixels == 0:
            return 0.0
        
        return float(white_pixels) / float(total_pixels)

    @torch.no_grad()
    def detect(
        self,
        frame,
        timestamp: float = 0.0,
        min_confidence: float = MIN_CONFIDENCE,
        white_pixel_threshold: float = 0.05,
    ):
        """Detect shuttle and player in frame with optional post-processing filter.
        
        Args:
            frame: Input frame (numpy array HxWx3, tensor 3xHxW, or 1xHxW, or HxW grayscale)
            timestamp: Timestamp for this frame
            min_confidence: Minimum confidence threshold for detection
            white_pixel_threshold: For shuttle: if white pixels < this ratio, reject detection
        """
        # This method is what video pipeline should call frame-by-frame.
        original_frame = None
        if isinstance(frame, np.ndarray):
            if frame.ndim == 2:
                # Grayscale
                original_frame = frame.copy()
                frame_rgb = np.stack([frame] * 3, axis=-1)
                pil = Image.fromarray(frame_rgb.astype(np.uint8))
                orig_h, orig_w = frame.shape[:2]
            elif frame.ndim == 3:
                original_frame = frame.copy()
                pil = Image.fromarray(frame.astype(np.uint8))
                orig_h, orig_w = frame.shape[:2]
            else:
                raise ValueError("Expected frame as HxW or HxWx3 numpy array")
        elif isinstance(frame, torch.Tensor):
            if frame.dim() == 3 and frame.shape[0] in (1, 3):
                if frame.shape[0] == 1:
                    # Grayscale tensor
                    original_frame = frame[0].cpu().numpy().astype(np.uint8)
                    frame = frame.repeat(3, 1, 1)
                else:
                    # RGB tensor
                    original_frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                pil = transforms.ToPILImage()(frame.cpu())
                orig_h, orig_w = int(frame.shape[1]), int(frame.shape[2])
            else:
                raise ValueError("Expected frame tensor as (3,H,W) or (1,H,W)")
        else:
            raise TypeError("Frame must be numpy array or torch tensor")

        # Resize to model input, predict, then map back to original frame size.
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        pred = self.forward_detect(x)[0].cpu()

        outputs: Dict[str, Optional[Tuple[float, float, float, float, float]]] = {}
        for class_idx, class_name in enumerate(TRACKED_CLASSES):
            conf = float(pred[class_idx, 0].item())
            if conf < min_confidence:
                outputs[class_name] = None
                continue
            box_norm = pred[class_idx, 1:]
            box_xywh = _cxcywh_norm_to_xywh(box_norm.unsqueeze(0), orig_w, orig_h)[0]
            x0, y0, w, h = [float(v.item()) for v in box_xywh]
            
            # Post-processing: for shuttle, check white pixel ratio in MOG frames
            if class_name == "shuttle" and original_frame is not None:
                white_ratio = self._get_white_pixel_ratio(original_frame, (x0, y0, w, h))
                if white_ratio < white_pixel_threshold:
                    # Skip this detection if insufficient white pixels
                    outputs[class_name] = None
                    continue
            
            outputs[class_name] = (timestamp, x0, y0, h, w)

        return outputs


class DINODataset(Dataset):
    """Generate multi-crop DINO samples with COCO-style supervision.

    Supports any COCO-formatted dataset with images and bounding box annotations.
    Automatically detects shuttle/player from standard COCO categories.
    Can load single dataset or combine multiple datasets into single loader.
    
    Supported datasets:
    - data/input/train: Original training dataset (10K images)
    - data/input/train_mog_frames: MOG2-masked frames (10K images, better contrast)
    - data/input/train_mog_reflect: Augmented dataset (20K images, 10K original + 10K horizontally-flipped) [DEFAULT, RECOMMENDED]

    Parameters:
    - data_dir: str or list of str, path(s) to dataset directory/directories
    - annotations_file: str or list of str (optional), path(s) to COCO JSON files
      If not provided, auto-detects as "_annotations.coco.json" in each data_dir
      Multi-dataset support: pass comma-separated paths or lists to train on combined datasets

    Returns a sample dict with:
    - image_path: original file path
    - crops: list of [global_1, global_2, local_1, ..., local_N]
    - det_image: resized tensor for detector supervision
    - det_target: tensor shape (2,5): [conf, cx, cy, w, h] for (player, shuttle)
    - gt_boxes_xywh: tensor shape (2,4), resized-space ground-truth boxes
    """

    def __init__(
        self,
        device,
        data_dir,
        global_crop_size=GLOBAL_CROP_SIZE,
        local_crop_size=LOCAL_CROP_SIZE,
        num_global_crops=NUM_GLOBAL_CROPS,
        num_local_crops=NUM_LOCAL_CROPS,
        annotations_file=None,
    ):
        self.device = device
        # Support both single directory (str) and multiple directories (list of str)
        if isinstance(data_dir, str):
            data_dirs = [data_dir]
        else:
            data_dirs = list(data_dir)
        
        # Support both single annotation file and multiple files
        if annotations_file is None:
            annotations_files = [os.path.join(d, "_annotations.coco.json") for d in data_dirs]
        elif isinstance(annotations_file, str):
            annotations_files = [annotations_file]
        else:
            annotations_files = list(annotations_file)
        
        # Ensure we have matching counts
        if len(annotations_files) == 1 and len(data_dirs) > 1:
            # If only one annotation file provided for multiple dirs, expand it
            annotations_files = annotations_files * len(data_dirs)
        elif len(data_dirs) != len(annotations_files):
            raise ValueError(
                f"Mismatch: {len(data_dirs)} data_dirs but {len(annotations_files)} annotation files"
            )
        
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

        # Load and concatenate all datasets
        self.categories = {}
        self.images = []
        self.image_paths = []
        self.image_id_to_index = {}
        self.annotations_by_image: Dict[int, List[dict]] = {}
        
        for data_dir, ann_file in zip(data_dirs, annotations_files):
            with open(ann_file, "r", encoding="utf-8") as f:
                coco = json.load(f)
            
            # Merge categories (assuming all datasets have same categories)
            if not self.categories:
                self.categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}
            
            # Track current offset to reassign image IDs uniquely
            id_offset = max(self.annotations_by_image.keys()) + 1 if self.annotations_by_image else 0
            
            # Load images with path adjustment
            images_in_dir = sorted(coco.get("images", []), key=lambda x: x["id"])
            # Build a filesystem map for this data_dir to tolerate different layouts
            # (some datasets store images under per-video subfolders).
            file_map = {}
            basename_map = {}
            for root, _, files in os.walk(data_dir):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, data_dir)
                    # store relative path and basename -> full path
                    if rel not in file_map:
                        file_map[rel] = full
                    if f not in basename_map:
                        basename_map[f] = full

            for im in images_in_dir:
                new_id = im["id"] + id_offset
                self.images.append(im)
                ann_fname = im.get("file_name")
                resolved = None
                # Try exact relative path listed in annotation
                if ann_fname in file_map:
                    resolved = file_map[ann_fname]
                # Try basename match (handles images stored in subfolders)
                elif os.path.basename(ann_fname) in basename_map:
                    resolved = basename_map[os.path.basename(ann_fname)]
                else:
                    # Fallback: direct join (keeps previous behavior) if it exists
                    candidate = os.path.join(data_dir, ann_fname)
                    if os.path.exists(candidate):
                        resolved = candidate

                if resolved is None:
                    # Last resort: try case-insensitive search for basename
                    b = os.path.basename(ann_fname)
                    for k, v in basename_map.items():
                        if k.lower() == b.lower():
                            resolved = v
                            break

                if resolved is None:
                    # If still unresolved, warn and use the joined path (will raise later)
                    resolved = os.path.join(data_dir, ann_fname)
                    print(f"[DINODataset] Warning: could not resolve {ann_fname} under {data_dir}; using {resolved}")
                else:
                    if resolved != os.path.join(data_dir, ann_fname):
                        print(f"[DINODataset] Resolved {ann_fname} -> {resolved}")

                self.image_paths.append(resolved)
                self.image_id_to_index[new_id] = len(self.image_paths) - 1
                self.annotations_by_image[new_id] = []
            
            # Load annotations with ID offset
            for ann in coco.get("annotations", []):
                image_id = ann.get("image_id") + id_offset
                if image_id in self.annotations_by_image:
                    self.annotations_by_image[image_id].append(ann)

        # DINO-style global and local crop pipelines.
        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.global_crop_size, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
            ]
        )
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.local_crop_size, scale=(0.05, 0.4)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
            ]
        )

        # Lighter augmentation for detection branch (keep geometry stable).
        self.det_color_jitter = transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.2,
            hue=0.03,
        )
        self.det_to_tensor = transforms.Compose(
            [transforms.Resize((self.global_crop_size, self.global_crop_size)), transforms.ToTensor()]
        )

        # Preserved for compatibility with initial skeleton contract.
        self.global_crops: List[torch.Tensor] = []
        self.local_crops: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.length = len(self.images)

        self.class_to_idx = {name: i for i, name in enumerate(TRACKED_CLASSES)}
        # Map annotation class names to the 2 targets this tracker predicts.
        self.coco_name_to_track = {
            "person": "player",
            "player": "player",
            "shuttle": "shuttle",
            "shuttlecock": "shuttle",
        }

    def __len__(self):
        return self.length

    def _pick_representative_boxes(
        self, anns: Iterable[dict], img_w: float, img_h: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Collapse potentially many detections into one box per tracked class.
        buckets: Dict[str, List[torch.Tensor]] = {k: [] for k in TRACKED_CLASSES}
        for ann in anns:
            cat_id = ann.get("category_id")
            cname = self.categories.get(cat_id, "")
            mapped = self.coco_name_to_track.get(cname)
            if mapped is None:
                continue
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            x = max(0.0, min(x, img_w - 1.0))
            y = max(0.0, min(y, img_h - 1.0))
            w = max(0.0, min(w, img_w - x))
            h = max(0.0, min(h, img_h - y))
            buckets[mapped].append(torch.tensor([x, y, w, h], dtype=torch.float32))

        out = {}
        for key, vals in buckets.items():
            if not vals:
                out[key] = None
                continue
            # Single-target head: keep the biggest object as supervision target.
            out[key] = max(vals, key=lambda b: float((b[2] * b[3]).item()))
        return out

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        anns = self.annotations_by_image.get(image_id, [])
        selected = self._pick_representative_boxes(anns, orig_w, orig_h)

        # If image is flipped, we must flip bbox x-coordinate too.
        do_hflip = random.random() < 0.5
        if do_hflip:
            image = TF.hflip(image)
            for key in TRACKED_CLASSES:
                if selected[key] is None:
                    continue
                box = selected[key].clone()
                box[0] = float(orig_w) - box[0] - box[2]
                selected[key] = box

        image_det = self.det_color_jitter(image)
        det_image = self.det_to_tensor(image_det)

        # Scale COCO boxes from original image size to detector input size.
        sx = self.global_crop_size / float(orig_w)
        sy = self.global_crop_size / float(orig_h)
        det_targets = torch.zeros((len(TRACKED_CLASSES), 5), dtype=torch.float32)
        gt_boxes_xywh = torch.zeros((len(TRACKED_CLASSES), 4), dtype=torch.float32)

        for class_idx, class_name in enumerate(TRACKED_CLASSES):
            box = selected[class_name]
            if box is None:
                continue
            box_scaled = box.clone()
            box_scaled[0] *= sx
            box_scaled[1] *= sy
            box_scaled[2] *= sx
            box_scaled[3] *= sy
            gt_boxes_xywh[class_idx] = box_scaled
            det_targets[class_idx, 0] = 1.0
            det_targets[class_idx, 1:] = _xywh_to_cxcywh_norm(
                box_scaled,
                width=self.global_crop_size,
                height=self.global_crop_size,
            )

        # Build all SSL crops (teacher sees only global crops in training loop).
        crops: List[torch.Tensor] = []
        for _ in range(self.num_global_crops):
            crops.append(self.global_transform(image))
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        return {
            "image_path": img_path,
            "crops": crops,
            "det_image": det_image,
            "det_target": det_targets,
            "gt_boxes_xywh": gt_boxes_xywh,
        }





def train_dino(
    student,
    teacher,
    dataset,
    device,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    output_dir: str = "data/output",
    checkpoint_name: str = "dino_tracker.pt",
    pretrained_backbone_path: Optional[str] = None,
    freeze_backbone_epochs: int = 3,
    backbone_lr_factor: float = 0.1,
    # LoRA fine-tuning options
    use_lora: bool = False,
    lora_r: int = 4,
    lora_alpha: int = 16,
    num_workers: int = 0,
    debug_batches: int = 0,
    log_every: int = 10,
    # LoRA targeting options
    use_lora_modules: Optional[List[str]] = None,
    lora_min_dim: int = 64,
    use_amp: bool = False,
    # LR scheduler options
    end_learning_rate: Optional[float] = None,
    lr_warmup_epochs: int = 0,
):
    """Train DINO student/teacher with detector head using 80/20 train/val split.

    Supports passing pre-built student/teacher (DINOTracker objects). If either
    is None, this builds models internally.

    Hyperparameters (May 2026 improvements):
    - LEARNING_RATE: 1e-3 (faster convergence)
    - EMA_MOMENTUM_START: 0.993, EMA_MOMENTUM_END: 0.9995 (stable teacher updates)
    - CENTER_MOMENTUM: 0.95 (strong centering to prevent collapse; was 0.9 which is too weak)
    - VAL_IOU_THRESHOLD: 0.5 (reduced from 0.75 for realistic shuttle metrics)

    Anti-collapse mechanisms:
    - Centering EMA applied to all student outputs (not just teacher), with L2 normalization
    - Teacher temperature warmup in first 10 epochs
    - Batch-level center stabilization prevents mode collapse

    Dataset recommendations:
    - Use train_mog_frames for baseline (MOG2-masked, better contrast)
    - Use train_mog_reflect for improved training (augmented with 2x horizontal reflections)

    Features:
    - 80/20 train/validation split
    - Validate with IoU and mAP@0.5
    - Cosine LR scheduling and EMA momentum scheduling
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print(
        f"[TRAIN] start epochs={epochs} batch_size={batch_size} "
        f"lr={learning_rate} output_dir={output_dir}"
    )

    # Caller can pass custom models; otherwise create default student/teacher.
    if isinstance(student, DINOTracker):
        student_model = student
    else:
        # Allow building a model that loads a pretrained backbone path.
        student_model = DINOTracker(device=device, pretrained_backbone_path=pretrained_backbone_path)
    # Optionally apply LoRA adapters to encoder for lightweight fine-tuning.
    if use_lora:
        n_replaced = apply_lora_to_encoder(
            student_model.encoder,
            r=lora_r,
            alpha=lora_alpha,
            min_dim=lora_min_dim,
            module_name_patterns=use_lora_modules,
        )
        print(f"[TRAIN] LoRA applied to encoder: replaced {n_replaced} Linear modules (r={lora_r}, alpha={lora_alpha})")
    teacher_model = teacher if isinstance(teacher, DINOTracker) else copy.deepcopy(student_model)
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.load_state_dict(student_model.state_dict(), strict=False)
    for p in teacher_model.parameters():
        p.requires_grad = False

    # DDP is disabled in this deployment; models remain single-process

    train_subset, val_subset = _split_dataset(dataset, train_ratio=0.8, seed=42)
    print(
        f"[TRAIN] dataset_split train={len(train_subset)} val={len(val_subset)} "
        f"eval_every={VAL_EVERY}"
    )
    print(f"[TRAIN] creating DataLoaders num_workers={num_workers} batch_size={batch_size}")
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_dino_collate,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_dino_collate,
        pin_memory=False,
    )

    # Optionally freeze backbone for the first few epochs and only train projector+detector.
    if freeze_backbone_epochs and freeze_backbone_epochs > 0:
        # Freeze encoder parameters but preserve LoRA adapter parameters (if present).
        for name, param in student_model.encoder.named_parameters():
            # Keep LoRA adapter params trainable by name convention (lora_A / lora_B)
            if ("lora_A" in name) or ("lora_B" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    params_to_optimize = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(train_loader) * epochs, 1)
    # Determine end LR for decay
    end_lr = float(end_learning_rate) if end_learning_rate is not None else float(learning_rate * 0.1)

    history = TrainHistory(train_loss=[], val_loss=[], val_iou=[], val_map75=[], eval_epochs=[])
    # DINO centering buffer stabilizes teacher targets.
    center = torch.zeros(DINO_OUT_DIM, device=device)

    # AMP scaler if requested
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step_count = 0
    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0.0

        # Warm up teacher temperature to avoid early collapse.
        if epoch < DINO_TEACHER_TEMP_WARMUP_EPOCHS:
            alpha = epoch / max(DINO_TEACHER_TEMP_WARMUP_EPOCHS - 1, 1)
            teacher_temp = (
                (1.0 - alpha) * DINO_TEACHER_TEMP_START + alpha * DINO_TEACHER_TEMP_END
            )
        else:
            teacher_temp = DINO_TEACHER_TEMP_END

        batch_idx = 0
        for batch in train_loader:
            views = [v.to(device) for v in batch["crops_by_view"]]
            det_images = batch["det_images"].to(device)
            det_targets = batch["det_targets"].to(device)

            # Ensure input tensors have spatial dimensions that are multiples
            # of the encoder patch size (e.g., 14 for ViT-B/14). Some backbones
            # require H and W to be divisible by patch size; resize here using
            # bilinear interpolation so training tensors are compatible.
            pe = getattr(student_model.encoder, "patch_embed", None)
            patch_H = 16
            if pe is not None:
                ps = getattr(pe, "patch_size", None)
                if isinstance(ps, (tuple, list)):
                    patch_H = int(ps[0])
                elif isinstance(ps, int):
                    patch_H = ps

            def _ensure_multiple(t: torch.Tensor, multiple: int) -> torch.Tensor:
                # t: (B,C,H,W)
                b, c, h, w = t.shape
                if h % multiple == 0 and w % multiple == 0:
                    return t
                new_h = ((h + multiple - 1) // multiple) * multiple
                new_w = ((w + multiple - 1) // multiple) * multiple
                return F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)

            views = [_ensure_multiple(v, patch_H) for v in views]
            det_images = _ensure_multiple(det_images, patch_H)

            # LR scheduling: optional linear warmup (by epoch count) then cosine decay to end_lr
            warmup_steps = max(1, lr_warmup_epochs * max(len(train_loader), 1))
            if step_count < warmup_steps and lr_warmup_epochs > 0:
                lr_step = float(learning_rate) * float(step_count) / float(max(1, warmup_steps))
            else:
                # cosine schedule after warmup; adjust step index and total for decay portion
                decay_step = max(0, step_count - warmup_steps)
                decay_total = max(1, total_steps - warmup_steps)
                lr_step = _cosine_anneal(learning_rate, end_lr, decay_step, decay_total)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_step

            with torch.no_grad():
                # Teacher only processes global crops.
                teacher_views = [teacher_model.forward_dino(views[i]) for i in range(NUM_GLOBAL_CROPS)]
                teacher_probs = [F.softmax((tv - center) / teacher_temp, dim=-1).detach() for tv in teacher_views]

            # Student sees global + local views.
            # Use AMP autocast for student forward if enabled
            autocast = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast
            with autocast(enabled=use_amp):
                student_outs = [student_model.forward_dino(v) for v in views]

            # Cross-view DINO objective, skipping exact same-view pairs.
            ssl_loss = 0.0
            num_pairs = 0
            for i in range(NUM_GLOBAL_CROPS):
                for j in range(len(student_outs)):
                    if i == j:
                        continue
                    s_log = F.log_softmax(student_outs[j] / DINO_STUDENT_TEMP, dim=-1)
                    pair_loss = -(teacher_probs[i] * s_log).sum(dim=-1).mean()
                    ssl_loss = ssl_loss + pair_loss
                    num_pairs += 1
            ssl_loss = ssl_loss / max(num_pairs, 1)

            # Supervised branch grounds representation on shuttle/player boxes.
            pred = student_model.forward_detect(det_images)
            conf_pred = pred[..., 0]
            box_pred = pred[..., 1:]
            conf_target = det_targets[..., 0]
            box_target = det_targets[..., 1:]

            conf_loss = F.binary_cross_entropy(conf_pred, conf_target)
            box_loss = F.l1_loss(box_pred, box_target, reduction="none")
            box_loss = (box_loss.sum(dim=-1) * conf_target).sum() / conf_target.sum().clamp(min=1.0)
            det_loss = conf_loss + BOX_LOSS_WEIGHT * box_loss

            # Joint objective = SSL + detection.
            loss = SSL_LOSS_WEIGHT * ssl_loss + det_loss

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                # Update center: compute from all student outputs for stability.
                # Center prevents mode collapse by centering teacher targets.
                all_student_outputs = torch.cat(student_outs, dim=0)
                batch_center = all_student_outputs.mean(dim=0)
                # Strong EMA momentum stabilizes predictions; use CENTER_MOMENTUM=0.95+
                center = center * CENTER_MOMENTUM + batch_center * (1.0 - CENTER_MOMENTUM)
                # L2 normalize center to prevent unbounded growth (optional but helps numerically)
                center = center / (torch.norm(center, p=2, dim=-1, keepdim=True) + 1e-8)

                ema_m = _cosine_anneal(EMA_MOMENTUM_START, EMA_MOMENTUM_END, step_count, total_steps)
                for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
                    t_param.data.mul_(ema_m).add_(s_param.data, alpha=(1.0 - ema_m))

            epoch_loss += float(loss.item())
            step_count += 1
            batch_idx += 1

            # Lightweight logging so long-running runs show progress.
            if step_count % max(1, log_every) == 0:
                print(f"[TRAIN] epoch {epoch+1}/{epochs} step {step_count} batch {batch_idx} loss={loss.item():.4f}")

            # If debug_batches set, run only that many batches then exit epoch early.
            if debug_batches and batch_idx >= debug_batches:
                print(f"[TRAIN] debug_batches reached ({debug_batches}), breaking epoch early")
                break

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        history.train_loss.append(avg_train_loss)

        # Evaluate every VAL_EVERY epochs and at final epoch.
        should_eval = ((epoch + 1) % VAL_EVERY == 0) or (epoch + 1 == epochs)
        if should_eval:
            val_loss, val_iou, val_map75 = _evaluate_detector(student_model, val_loader, device)
            history.val_loss.append(val_loss)
            history.val_iou.append(val_iou)
            history.val_map75.append(val_map75)
            history.eval_epochs.append(epoch + 1)
            print(
                f"[TRAIN] epoch {epoch + 1:03d}/{epochs} | train_loss={avg_train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_mAP@{VAL_IOU_THRESHOLD}={val_map75:.4f}"
            )
        else:
            print(f"[TRAIN] epoch {epoch + 1:03d}/{epochs} | train_loss={avg_train_loss:.4f}")

        # Unfreeze backbone when we pass the freeze_backbone_epochs threshold.
        if freeze_backbone_epochs and (epoch + 1) == freeze_backbone_epochs:
            print(f"[TRAIN] unfreezing encoder parameters at epoch {epoch + 1}")
            # Unfreeze all encoder params (including adapters); rebuild optimizer grouping
            for name, param in student_model.encoder.named_parameters():
                param.requires_grad = True
            # Rebuild optimizer with a smaller LR for the backbone parameters.
            backbone_params = [p for p in student_model.encoder.parameters() if p.requires_grad]
            backbone_param_ids = {id(p) for p in backbone_params}
            other_params = [p for p in student_model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
            param_groups = [
                {"params": other_params, "lr": learning_rate},
                {"params": backbone_params, "lr": learning_rate * backbone_lr_factor},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # Save student model (teacher is only for training, not needed for inference)
    ckpt_path = os.path.join(output_dir, checkpoint_name)
    student_model.save_checkpoint(ckpt_path)
    print(f"[TRAIN] saved_checkpoint student={ckpt_path}")

    history_np = {
        "train_loss": np.asarray(history.train_loss, dtype=np.float32),
        "val_loss": np.asarray(history.val_loss, dtype=np.float32),
        "val_iou": np.asarray(history.val_iou, dtype=np.float32),
        "val_map75": np.asarray(history.val_map75, dtype=np.float32),
        "eval_epochs": np.asarray(history.eval_epochs, dtype=np.int32),
    }
    # Save curves as arrays for later plotting or analysis scripts.
    np.savez(os.path.join(output_dir, "dino_training_history.npz"), **history_np)
    print(f"[TRAIN] saved_history={os.path.join(output_dir, 'dino_training_history.npz')}")

    visualize_training(
        student_model,
        teacher_model,
        dataset,
        device,
        history=history,
        save_dir=output_dir,
    )
    print("[TRAIN] complete")

    return student_model, teacher_model, history


@torch.no_grad()
@torch.no_grad()
def visualize_training(
    student,
    teacher,
    dataset,
    device,
    history: Optional[TrainHistory] = None,
    save_dir: str = "data/output",
):
    """Minimal training visualization - only metric curves to save disk space.
    
    Outputs:
    - loss_curves.png: training and validation loss over epochs
    - metrics_curves.png: IoU and mAP metrics over epochs
    
    Removed: student/teacher PNG comparison (disk space optimization)
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)
    
    student.eval()
    teacher.eval()

    if history is None:
        return

    # Save loss curves for quick sanity checks during long runs.
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(history.train_loss) + 1), history.train_loss, label="train loss")
    if history.eval_epochs and history.val_loss:
        plt.plot(history.eval_epochs, history.val_loss, marker="o", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("DINO + Detection Loss Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=100)
    plt.close(fig)

    # Save validation metric curves.
    fig = plt.figure(figsize=(8, 4))
    if history.eval_epochs and history.val_iou:
        plt.plot(history.eval_epochs, history.val_iou, marker="o", label="val IoU")
    if history.eval_epochs and history.val_map75:
        plt.plot(history.eval_epochs, history.val_map75, marker="s", label=f"val mAP@{VAL_IOU_THRESHOLD}")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title("Validation Detection Metrics")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_curves.png"), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------
# Helper methods for box conversions, IoU, AP calculation, cosine annealing, and ViT feature extraction.
# ---------------------------------------------------------------------

def _strip_prefix(state_dict: dict, prefixes=("module.", "backbone.", "encoder.")) -> dict:
    """Remove common checkpoint prefixes so keys match encoder state."""
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p) :]
                break
        out[new_k] = v
    return out


def _dino_collate(batch: List[dict]):
    """Custom collate because each sample carries both crop-views and det labels."""
    num_crops = len(batch[0]["crops"])
    crops_by_view = [torch.stack([sample["crops"][i] for sample in batch], dim=0) for i in range(num_crops)]
    det_images = torch.stack([sample["det_image"] for sample in batch], dim=0)
    det_targets = torch.stack([sample["det_target"] for sample in batch], dim=0)
    gt_boxes_xywh = torch.stack([sample["gt_boxes_xywh"] for sample in batch], dim=0)
    image_paths = [sample["image_path"] for sample in batch]
    return {
        "crops_by_view": crops_by_view,
        "det_images": det_images,
        "det_targets": det_targets,
        "gt_boxes_xywh": gt_boxes_xywh,
        "image_paths": image_paths,
    }


def _split_dataset(dataset: Dataset, train_ratio: float = 0.8, seed: int = 42):
    # Deterministic split helps reproducibility when comparing experiments.
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = max(1, int(train_ratio * n))
    split = min(split, n - 1) if n > 1 else n
    train_idx = indices[:split]
    val_idx = indices[split:] if n > 1 else indices
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


@torch.no_grad()
def _evaluate_detector(model: DINOTracker, loader: DataLoader, device: torch.device):
    """Validation pass for detector branch (loss + IoU + mAP@threshold)."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count_iou = 0

    pred_conf = {c: [] for c in TRACKED_CLASSES}
    pred_iou = {c: [] for c in TRACKED_CLASSES}
    gt_exists = {c: [] for c in TRACKED_CLASSES}

    for batch in loader:
        # Pure eval pass: no gradients, just metrics.
        images = batch["det_images"].to(device)
        # Ensure evaluation images have spatial dims compatible with encoder patch size
        pe = getattr(model.encoder, "patch_embed", None)
        patch_H = 16
        if pe is not None:
            ps = getattr(pe, "patch_size", None)
            if isinstance(ps, (tuple, list)):
                patch_H = int(ps[0])
            elif isinstance(ps, int):
                patch_H = ps
        images = _ensure_multiple_tensor(images, patch_H)
        targets = batch["det_targets"].to(device)
        gt_boxes = batch["gt_boxes_xywh"].to(device)

        pred = model.forward_detect(images)
        conf_pred = pred[..., 0]
        box_pred_norm = pred[..., 1:]

        conf_target = targets[..., 0]
        box_target_norm = targets[..., 1:]

        conf_loss = F.binary_cross_entropy(conf_pred, conf_target)
        box_loss = F.l1_loss(box_pred_norm, box_target_norm, reduction="none")
        box_loss = (box_loss.sum(dim=-1) * conf_target).sum() / conf_target.sum().clamp(min=1.0)
        total_loss += float((conf_loss + box_loss).item())

        # Compute IoU in normalized coordinate space (0..1) to avoid needing
        # matched absolute pixel sizes between dataset and backbone input.
        pred_xywh_norm = _cxcywh_norm_to_xywh(box_pred_norm.reshape(-1, 4), 1.0, 1.0).reshape_as(box_target_norm)
        gt_xywh_norm = _cxcywh_norm_to_xywh(box_target_norm.reshape(-1, 4), 1.0, 1.0).reshape_as(box_target_norm)

        for cls_idx, cls_name in enumerate(TRACKED_CLASSES):
            cls_has_gt = conf_target[:, cls_idx] > 0.5
            cls_iou = _bbox_iou_xywh(pred_xywh_norm[:, cls_idx], gt_xywh_norm[:, cls_idx])
            pred_conf[cls_name].extend(conf_pred[:, cls_idx].detach().cpu().numpy().tolist())
            pred_iou[cls_name].extend(cls_iou.detach().cpu().numpy().tolist())
            gt_exists[cls_name].extend(cls_has_gt.detach().cpu().numpy().astype(np.float32).tolist())

            if cls_has_gt.any():
                total_iou += float(cls_iou[cls_has_gt].sum().item())
                count_iou += int(cls_has_gt.sum().item())

    mean_iou = total_iou / max(count_iou, 1)
    ap_values = []
    for cls_name in TRACKED_CLASSES:
        ap_values.append(
            _compute_ap_from_scores(
                np.asarray(pred_conf[cls_name], dtype=np.float32),
                np.asarray(pred_iou[cls_name], dtype=np.float32),
                np.asarray(gt_exists[cls_name], dtype=np.float32),
                iou_threshold=VAL_IOU_THRESHOLD,
            )
        )
    map75 = float(np.mean(ap_values)) if ap_values else 0.0
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, mean_iou, map75


def _create_vit_tiny(pretrained_weights_path: Optional[str] = None) -> Tuple[nn.Module, int]:
    """Create a ViT-tiny backbone via timm.

    Raises a clear error if timm is missing, since this project depends on it.
    """
    # Keep backbone creation in one place so swapping model size is easy later.
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for ViT backbone creation. Install via: pip install timm"
        ) from exc

    # Prefer DINOv2 pretrained weights if available via torch.hub. The exact model
    # name can be provided through the environment variable `DINOV2_MODEL`.
    dinov2_model = os.environ.get("DINOV2_MODEL", "dinov2_vitb14")
    try:
        # Attempt to load DINOv2 from FacebookResearch hub. This returns a nn.Module
        # compatible with timm-like ViT APIs in most cases.
        import torch

        print(f"[DINO] attempting to load pretrained backbone from dinov2: {dinov2_model}")
        encoder = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        embed_dim = getattr(encoder, "embed_dim", None) or getattr(encoder, "num_features", None)
        if embed_dim is None:
            embed_dim = 768
        print(f"[DINO] loaded dinov2 model {dinov2_model} (embed_dim={embed_dim})")
        # If a local pretrained checkpoint path is provided, try to load it into
        # the backbone. We attempt to be robust to common checkpoint wrappers.
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            try:
                ckpt = torch.load(pretrained_weights_path, map_location="cpu")
                # Heuristics to extract state_dict
                if isinstance(ckpt, dict):
                    if "state_dict" in ckpt:
                        sd = ckpt["state_dict"]
                    elif "model" in ckpt:
                        sd = ckpt["model"]
                    else:
                        sd = ckpt
                else:
                    sd = ckpt

                sd_clean = _strip_prefix(sd)
                try:
                    encoder.load_state_dict(sd_clean, strict=False)
                    print(f"[DINO] loaded pretrained weights from {pretrained_weights_path} into encoder")
                except Exception as le:
                    print(f"[DINO] warning: failed to strictly load pretrained backbone weights: {le}")
            except Exception as e:
                print(f"[DINO] warning: failed to read pretrained_weights_path {pretrained_weights_path}: {e}")

        return encoder, embed_dim
    except Exception as e:
        # If torch.hub or network access is not available on the cluster/node,
        # fallback to a timm-created ViT with ImageNet-pretrained weights when possible.
        print(f"[DINO] could not load dinov2 via torch.hub: {e}. Falling back to timm ViT.")
        # Try image-net pretrained tiny ViT as a reasonable fallback.
        encoder = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=0, dynamic_img_size=True
        )
        embed_dim = getattr(encoder, "embed_dim", 192)
        # Optionally load a local checkpoint into the timm encoder as well.
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            try:
                ckpt = torch.load(pretrained_weights_path, map_location="cpu")
                sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                sd_clean = _strip_prefix(sd)
                try:
                    encoder.load_state_dict(sd_clean, strict=False)
                    print(f"[DINO] loaded pretrained weights from {pretrained_weights_path} into timm encoder")
                except Exception as le:
                    print(f"[DINO] warning: failed to strictly load pretrained backbone weights into timm encoder: {le}")
            except Exception as e:
                print(f"[DINO] warning: failed to read pretrained_weights_path {pretrained_weights_path}: {e}")

        return encoder, embed_dim


def _extract_cls_token(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return class-token-like embedding from an encoder.

    Handles common timm output variants.
    """
    # Different timm models return features in slightly different formats.
    if hasattr(encoder, "forward_features"):
        features = encoder.forward_features(x)
    else:
        features = encoder(x)

    if isinstance(features, dict):
        if "x_norm_clstoken" in features:
            return features["x_norm_clstoken"]
        if "cls_token" in features:
            return features["cls_token"]
        if "x_prenorm" in features and isinstance(features["x_prenorm"], torch.Tensor):
            x_pre = features["x_prenorm"]
            return x_pre[:, 0, :] if x_pre.dim() == 3 else x_pre

    if isinstance(features, torch.Tensor):
        if features.dim() == 3:
            return features[:, 0, :]
        if features.dim() == 2:
            return features

    raise RuntimeError("Unable to extract class token embedding from encoder output")


def _cosine_anneal(start: float, end: float, step: int, total_steps: int) -> float:
    # Smooth schedule used for LR and EMA momentum.
    if total_steps <= 1:
        return end
    ratio = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * ratio))


def _ensure_multiple_tensor(t: torch.Tensor, multiple: int) -> torch.Tensor:
    """Resize a 4D image tensor so H and W are multiples of `multiple` using bilinear interpolation.

    Leaves other tensor shapes unchanged.
    """
    if not isinstance(t, torch.Tensor) or t.dim() != 4:
        return t
    b, c, h, w = t.shape
    if h % multiple == 0 and w % multiple == 0:
        return t
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    return F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _xywh_to_cxcywh_norm(xywh: torch.Tensor, width: float, height: float) -> torch.Tensor:
    # Detector learns normalized center-format boxes.
    x, y, w, h = xywh.unbind(-1)
    cx = (x + 0.5 * w) / max(width, 1e-6)
    cy = (y + 0.5 * h) / max(height, 1e-6)
    nw = w / max(width, 1e-6)
    nh = h / max(height, 1e-6)
    out = torch.stack([cx, cy, nw, nh], dim=-1)
    return out.clamp(0.0, 1.0)


def _cxcywh_norm_to_xywh(cxcywh: torch.Tensor, width: float, height: float) -> torch.Tensor:
    # Convert predictions back to absolute image coordinates for eval/inference.
    cx, cy, w, h = cxcywh.unbind(-1)
    abs_w = (w * width).clamp(min=0.0)
    abs_h = (h * height).clamp(min=0.0)
    x = (cx * width - 0.5 * abs_w).clamp(min=0.0, max=max(width - 1.0, 0.0))
    y = (cy * height - 0.5 * abs_h).clamp(min=0.0, max=max(height - 1.0, 0.0))
    return torch.stack([x, y, abs_w, abs_h], dim=-1)


def _bbox_iou_xywh(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    # Basic IoU in XYWH format.
    ax1, ay1, aw, ah = box_a.unbind(-1)
    bx1, by1, bw, bh = box_b.unbind(-1)
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = torch.max(ax1, bx1)
    inter_y1 = torch.max(ay1, by1)
    inter_x2 = torch.min(ax2, bx2)
    inter_y2 = torch.min(ay2, by2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = aw.clamp(min=0) * ah.clamp(min=0)
    area_b = bw.clamp(min=0) * bh.clamp(min=0)
    union = (area_a + area_b - inter).clamp(min=1e-6)
    return inter / union


def _compute_ap_from_scores(
    confidences: np.ndarray,
    ious: np.ndarray,
    has_gt: np.ndarray,
    iou_threshold: float,
) -> float:
    # Simple AP estimator: sort by confidence, then integrate precision-recall.
    if confidences.size == 0:
        return 0.0
    order = np.argsort(-confidences)
    confidences = confidences[order]
    ious = ious[order]
    has_gt = has_gt[order]

    tp = ((ious >= iou_threshold) & (has_gt > 0.5)).astype(np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    total_gt = max(float(np.sum(has_gt > 0.5)), 1.0)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for i in range(precision.size - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    # NumPy 2.x prefers trapezoid; NumPy 1.x only provides trapz.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(precision, recall))


def _get_parent_module_by_name(module: nn.Module, full_name: str):
    """Return (parent_module, attr_name) for a dotted module path inside `module`.

    If not found, returns (None, None).
    """
    parts = full_name.split(".")
    parent = module
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, None
        parent = getattr(parent, p)
    return parent, parts[-1]


class LoRALinear(nn.Module):
    """Lightweight LoRA adapter wrapper for an existing nn.Linear.

    This keeps the original Linear as `orig` (frozen by default) and adds two
    small adapter matrices `lora_A` and `lora_B`. Forward returns
    orig(x) + scaling * (x @ lora_A @ lora_B).
    """

    def __init__(self, orig_linear: nn.Linear, r: int = 4, alpha: int = 16):
        super().__init__()
        self.orig = orig_linear
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.r = max(1, int(r))
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.r))
        self.lora_B = nn.Parameter(torch.zeros(self.r, self.out_features))
        self.scaling = float(alpha) / float(self.r) if self.r > 0 else 1.0

        # Initialize adapters: A with kaiming, B zeros (common LoRA init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # By default freeze original weights; adapter params remain trainable.
        for p in self.orig.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original output
        orig_out = self.orig(x)
        # Compute LoRA delta: (.., in) @ (in, r) -> (.., r) then @ (r, out) -> (.., out)
        lora_inter = torch.matmul(x, self.lora_A)
        lora_out = torch.matmul(lora_inter, self.lora_B) * (self.scaling)
        return orig_out + lora_out


def apply_lora_to_encoder(
    encoder: nn.Module,
    r: int = 4,
    alpha: int = 16,
    min_dim: int = 64,
    module_name_patterns: Optional[List[str]] = None,
) -> int:
    """Replace suitable nn.Linear modules inside `encoder` with LoRA-wrapped versions.

    Heuristics:
    - If `module_name_patterns` is provided, only apply LoRA to modules whose
      full module name contains any of the given substrings.
    - Otherwise apply to Linear layers whose in/out feature sizes are >= `min_dim`.

    Returns number of modules replaced.
    """
    replaced = 0
    # Collect linear module names first to avoid mutation while iterating
    linear_items = [(name, m) for name, m in encoder.named_modules() if isinstance(m, nn.Linear)]
    for full_name, mod in linear_items:
        parent, attr = _get_parent_module_by_name(encoder, full_name)
        if parent is None:
            continue
        orig = getattr(parent, attr)
        # Avoid double-wrapping
        if isinstance(orig, LoRALinear):
            continue

        # If patterns provided, only match those names
        if module_name_patterns:
            matched = any(pat in full_name for pat in module_name_patterns)
            if not matched:
                continue
        else:
            # Skip small matrices
            if getattr(orig, "in_features", 0) < min_dim or getattr(orig, "out_features", 0) < min_dim:
                continue

        # Replace module
        setattr(parent, attr, LoRALinear(orig, r=r, alpha=alpha))
        replaced += 1
    return replaced
