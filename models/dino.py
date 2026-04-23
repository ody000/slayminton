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
LEARNING_RATE = 1e-4
EPOCHS = 100
WEIGHT_DECAY = 1e-4
DINO_OUT_DIM = 256
DINO_HIDDEN_DIM = 512
DINO_STUDENT_TEMP = 0.1
DINO_TEACHER_TEMP_START = 0.04
DINO_TEACHER_TEMP_END = 0.07
DINO_TEACHER_TEMP_WARMUP_EPOCHS = 10
EMA_MOMENTUM_START = 0.996
EMA_MOMENTUM_END = 0.9995
BOX_LOSS_WEIGHT = 1.0
SSL_LOSS_WEIGHT = 1.0
MIN_CONFIDENCE = 0.15
VAL_EVERY = 5

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
        device: Optional[torch.device] = None,
        input_size: int = GLOBAL_CROP_SIZE,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        # One shared encoder for both SSL projection and detection.
        self.encoder, self.encoder_dim = _create_vit_tiny()
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

    @torch.no_grad()
    def detect(self, frame, timestamp: float = 0.0, min_confidence: float = MIN_CONFIDENCE):
        # This method is what video pipeline should call frame-by-frame.
        if isinstance(frame, np.ndarray):
            if frame.ndim != 3:
                raise ValueError("Expected frame as HxWx3 numpy array")
            pil = Image.fromarray(frame.astype(np.uint8))
            orig_h, orig_w = frame.shape[:2]
        elif isinstance(frame, torch.Tensor):
            if frame.dim() == 3 and frame.shape[0] in (1, 3):
                if frame.shape[0] == 1:
                    frame = frame.repeat(3, 1, 1)
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
            outputs[class_name] = (timestamp, x0, y0, h, w)

        return outputs


class DINODataset(Dataset):
    """Generate multi-crop DINO samples with COCO-style supervision.

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
        self.data_dir = data_dir
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.annotations_file = (
            annotations_file
            if annotations_file is not None
            else os.path.join(self.data_dir, "_annotations.coco.json")
        )

        # Parse COCO once and cache indexes for fast __getitem__.
        with open(self.annotations_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}
        self.images = sorted(coco.get("images", []), key=lambda x: x["id"])
        self.image_paths = [os.path.join(self.data_dir, im["file_name"]) for im in self.images]
        self.image_id_to_index = {im["id"]: idx for idx, im in enumerate(self.images)}

        self.annotations_by_image: Dict[int, List[dict]] = {im["id"]: [] for im in self.images}
        for ann in coco.get("annotations", []):
            image_id = ann.get("image_id")
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


class ViTEncoder(nn.Module):
    """ViT backbone with a projection head on the class token embedding."""

    def __init__(self, head: nn.Module, encoder: Optional[nn.Module] = None):
        super().__init__()
        if encoder is None:
            self.encoder, self.encoder_dim = _create_vit_tiny()
        else:
            self.encoder = encoder
            self.encoder_dim = getattr(encoder, "embed_dim", 192)
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = _extract_cls_token(self.encoder, x)
        return self.head(cls_token)


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
):
    """Train DINO student/teacher with detector head using 80/20 train/val split.

    Supports passing pre-built student/teacher (DINOTracker objects). If either
    is None, this builds models internally.

    - 80/20 train/validation split
    - validate with IoU and mAP@0.75
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print(
        f"[TRAIN] start epochs={epochs} batch_size={batch_size} "
        f"lr={learning_rate} output_dir={output_dir}"
    )

    # Caller can pass custom models; otherwise create default student/teacher.
    student_model = student if isinstance(student, DINOTracker) else _build_tracker_model(device)
    teacher_model = teacher if isinstance(teacher, DINOTracker) else copy.deepcopy(student_model)
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.load_state_dict(student_model.state_dict(), strict=False)
    for p in teacher_model.parameters():
        p.requires_grad = False

    train_subset, val_subset = _split_dataset(dataset, train_ratio=0.8, seed=42)
    print(
        f"[TRAIN] dataset_split train={len(train_subset)} val={len(val_subset)} "
        f"eval_every={VAL_EVERY}"
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_dino_collate,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_dino_collate,
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(train_loader) * epochs, 1)

    history = TrainHistory(train_loss=[], val_loss=[], val_iou=[], val_map75=[], eval_epochs=[])
    # DINO centering buffer stabilizes teacher targets.
    center = torch.zeros(DINO_OUT_DIM, device=device)

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

        for batch in train_loader:
            views = [v.to(device) for v in batch["crops_by_view"]]
            det_images = batch["det_images"].to(device)
            det_targets = batch["det_targets"].to(device)

            # Cosine LR update each step.
            lr_step = _cosine_anneal(learning_rate, learning_rate * 0.1, step_count, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_step

            with torch.no_grad():
                # Teacher only processes global crops.
                teacher_views = [teacher_model.forward_dino(views[i]) for i in range(NUM_GLOBAL_CROPS)]
                teacher_probs = [
                    F.softmax((tv - center) / teacher_temp, dim=-1).detach() for tv in teacher_views
                ]

            # Student sees global + local views.
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
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Update center and teacher weights (EMA from student).
                batch_center = torch.cat(teacher_views, dim=0).mean(dim=0)
                center = center * 0.9 + batch_center * 0.1

                ema_m = _cosine_anneal(EMA_MOMENTUM_START, EMA_MOMENTUM_END, step_count, total_steps)
                for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
                    t_param.data.mul_(ema_m).add_(s_param.data, alpha=(1.0 - ema_m))

            epoch_loss += float(loss.item())
            step_count += 1

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
                f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_mAP@0.75={val_map75:.4f}"
            )
        else:
            print(f"[TRAIN] epoch {epoch + 1:03d}/{epochs} | train_loss={avg_train_loss:.4f}")

    # Save both student and teacher so experiments are easy to resume/compare.
    ckpt_path = os.path.join(output_dir, checkpoint_name)
    student_model.save_checkpoint(ckpt_path)
    teacher_ckpt_path = os.path.join(output_dir, f"teacher_{checkpoint_name}")
    teacher_model.save_checkpoint(teacher_ckpt_path)
    print(f"[TRAIN] saved_checkpoint student={ckpt_path}")
    print(f"[TRAIN] saved_checkpoint teacher={teacher_ckpt_path}")

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
def visualize_training(
    student,
    teacher,
    dataset,
    device,
    history: Optional[TrainHistory] = None,
    save_dir: str = "data/output",
):
    """Visualize qualitative predictions and metric curves.

    Outputs:
    - examples_student_teacher.png
    - loss_curves.png
    - metrics_curves.png

    - show sample predictions for student and teacher
    - show loss/metric curves over epochs
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)

    student.eval()
    teacher.eval()

    # Qualitative panel: show student vs teacher boxes with GT overlay.
    sample_count = min(9, len(dataset))
    sample_indices = random.sample(range(len(dataset)), k=sample_count) if len(dataset) > 0 else []

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for ax_idx in range(9):
        ax = axes[ax_idx]
        ax.axis("off")
        if ax_idx >= sample_count:
            continue

        sample = dataset[sample_indices[ax_idx]]
        img = sample["det_image"].unsqueeze(0).to(device)
        gt = sample["gt_boxes_xywh"].cpu()

        p_student = student.forward_detect(img)[0].cpu()
        p_teacher = teacher.forward_detect(img)[0].cpu()

        img_np = sample["det_image"].permute(1, 2, 0).cpu().numpy()
        ax.imshow(np.clip(img_np, 0.0, 1.0))

        colors = {"player": "lime", "shuttle": "cyan"}
        for cls_idx, cls_name in enumerate(TRACKED_CLASSES):
            gt_box = gt[cls_idx]
            if gt_box.sum() > 0:
                x, y, w, h = gt_box.tolist()
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="yellow", linewidth=2)
                ax.add_patch(rect)

            s_conf = float(p_student[cls_idx, 0].item())
            t_conf = float(p_teacher[cls_idx, 0].item())

            s_box = _cxcywh_norm_to_xywh(
                p_student[cls_idx, 1:].unsqueeze(0), student.input_size, student.input_size
            )[0]
            t_box = _cxcywh_norm_to_xywh(
                p_teacher[cls_idx, 1:].unsqueeze(0), teacher.input_size, teacher.input_size
            )[0]

            if s_conf >= MIN_CONFIDENCE:
                x, y, w, h = s_box.tolist()
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=colors[cls_name], linewidth=1.6)
                ax.add_patch(rect)
                ax.text(x, max(0, y - 2), f"S:{cls_name}:{s_conf:.2f}", color=colors[cls_name], fontsize=8)

            if t_conf >= MIN_CONFIDENCE:
                x, y, w, h = t_box.tolist()
                rect = plt.Rectangle(
                    (x, y),
                    w,
                    h,
                    fill=False,
                    edgecolor="magenta",
                    linewidth=1.2,
                    linestyle="--",
                )
                ax.add_patch(rect)
                ax.text(x, min(student.input_size - 8, y + h + 8), f"T:{t_conf:.2f}", color="magenta", fontsize=7)

        ax.set_title(f"sample #{sample_indices[ax_idx]}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "examples_student_teacher.png"), dpi=160)
    plt.close(fig)

    if history is None:
        return

    # Curves are intentionally simple for quick sanity checks during long runs.
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
    plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    if history.eval_epochs and history.val_iou:
        plt.plot(history.eval_epochs, history.val_iou, marker="o", label="val IoU")
    if history.eval_epochs and history.val_map75:
        plt.plot(history.eval_epochs, history.val_map75, marker="s", label="val mAP@0.75")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title("Validation Detection Metrics")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_curves.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------
# Helper methods for box conversions, IoU, AP calculation, cosine annealing, and ViT feature extraction.
# ---------------------------------------------------------------------


def _build_tracker_model(device: torch.device) -> DINOTracker:
    # Tiny factory so training setup stays clean.
    return DINOTracker(device=device)


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
    """Validation pass for detector branch (loss + IoU + mAP@0.75)."""
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

        # Convert normalized predictions back to XYWH in detector image space.
        pred_xywh = _cxcywh_norm_to_xywh(
            box_pred_norm.reshape(-1, 4), model.input_size, model.input_size
        ).reshape_as(gt_boxes)

        for cls_idx, cls_name in enumerate(TRACKED_CLASSES):
            cls_has_gt = conf_target[:, cls_idx] > 0.5
            cls_iou = _bbox_iou_xywh(pred_xywh[:, cls_idx], gt_boxes[:, cls_idx])
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
                iou_threshold=0.75,
            )
        )
    map75 = float(np.mean(ap_values)) if ap_values else 0.0
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, mean_iou, map75


def _create_vit_tiny() -> Tuple[nn.Module, int]:
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

    encoder = timm.create_model("vit_tiny_patch16_384", pretrained=False, num_classes=0)
    embed_dim = getattr(encoder, "embed_dim", 192)
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
    return float(np.trapz(precision, recall))
