"""
14-Class Skin Lesion Classifier
================================
Full pipeline: data loading → augmentation → EfficientNet-B4 backbone →
two-phase transfer learning → evaluation → Grad-CAM explainability.

Requirements:
    pip install torch torchvision timm albumentations opencv-python \
                scikit-learn matplotlib seaborn tqdm grad-cam

Dataset folder structure expected:
    data/
      train/
        melanoma/        *.jpg
        nevus/           *.jpg
        bcc/             *.jpg
        ... (14 class folders)
      val/
        melanoma/
        ...
      test/
        melanoma/
        ...
    manifest.csv          (image_path, label, split, patient_id)
"""

# ─────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────
import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm                          # EfficientNet, ConvNeXt, Swin etc.

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

# Grad-CAM


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ─────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────
class CFG:
    # ── Paths ──────────────────────────────────────────────
    data_root   = Path("..data/skin_dataset/")
    output_dir  = Path("outputs")
    model_path  = output_dir / "best_model.pth"

    # ── Classes (14 lesion types) ──────────────────────────
    # Adjust to match your exact folder/label names
    classes = [
        "melanoma",             # MEL
        "melanocytic_nevus",    # NV
        "basal_cell_carcinoma", # BCC
        "squamous_cell_carcinoma",  # SCC
        "actinic_keratosis",    # AK
        "benign_keratosis",     # BKL
        "dermatofibroma",       # DF
        "vascular_lesion",      # VASC
        "seborrheic_keratosis", # SK
        "lentigo_maligna",      # LM
        "lichen_planus",        # LP
        "psoriasis",            # PSO
        "tinea",                # TIN
        "urticaria",            # URT
    ]
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # ── Model ──────────────────────────────────────────────
    backbone    = "efficientnet_b4"   # or: resnet50, convnext_base, swin_base_patch4_window7_224
    image_size  = 224
    pretrained  = True

    # ── Training ───────────────────────────────────────────
    seed        = 42
    num_epochs_phase1 = 10    # frozen backbone
    num_epochs_phase2 = 30    # full fine-tune
    batch_size  = 32
    num_workers = 4

    # Phase 1: train head only
    lr_head     = 1e-3
    # Phase 2: fine-tune full model
    lr_backbone = 1e-5
    lr_head_ft  = 1e-4

    weight_decay = 1e-4
    patience     = 10         # early stopping

    # ── Device ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CFG.seed)
CFG.output_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 2. Dataset & DataLoaders
# ─────────────────────────────────────────────────────────────

def get_transforms(split: str) -> A.Compose:
    """
    Training:   heavy augmentation (flip, rotate, colour jitter, cutout)
    Val/Test:   deterministic resize + normalize only
    ImageNet statistics used to match EfficientNet pretrained weights.
    """
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    if split == "train":
        return A.Compose([
            A.Resize(CFG.image_size + 32, CFG.image_size + 32),
            A.RandomCrop(CFG.image_size, CFG.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                               rotate_limit=45, p=0.6),
            # Dermoscopy-specific: colour variation across cameras/devices
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.3),
            # Remove hair/ruler artefacts effect (coarse dropout)
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                            fill_value=0, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CFG.image_size, CFG.image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class SkinLesionDataset(Dataset):
    """
    Reads images from class-labelled subfolders:
        data/train/melanoma/img001.jpg
        data/train/nevus/img002.jpg  ...

    Falls back to manifest.csv if folder structure absent.
    """
    def __init__(self, root: Path, split: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        split_dir = root / split
        if split_dir.exists():
            for cls_name in CFG.classes:
                cls_dir = split_dir / cls_name
                if not cls_dir.exists():
                    continue
                label = CFG.class_to_idx[cls_name]
                for img_path in cls_dir.glob("*.jpg"):
                    self.samples.append((img_path, label))
                for img_path in cls_dir.glob("*.png"):
                    self.samples.append((img_path, label))
        else:
            # Fallback: read manifest.csv
            manifest = pd.read_csv(root / "manifest.csv")
            subset = manifest[manifest["split"] == split]
            for _, row in subset.iterrows():
                label = CFG.class_to_idx[row["label"]]
                self.samples.append((Path(row["image_path"]), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for split '{split}' in {root}")

        print(f"  [{split:5s}] {len(self.samples):,} images across "
              f"{len(set(s[1] for s in self.samples))} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    def class_counts(self) -> np.ndarray:
        """Count samples per class — used for weighted sampling and loss."""
        counts = np.zeros(CFG.num_classes, dtype=np.int64)
        for _, label in self.samples:
            counts[label] += 1
        return counts


def make_weighted_sampler(dataset: SkinLesionDataset) -> WeightedRandomSampler:
    """
    Oversamples rare classes so each epoch sees a more balanced distribution.
    Weight per sample = 1 / count_of_its_class.
    """
    counts = dataset.class_counts()
    weights_per_class = 1.0 / (counts + 1e-6)
    sample_weights = np.array([weights_per_class[label]
                                for _, label in dataset.samples])
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


def build_dataloaders(data_root: Path) -> dict[str, DataLoader]:
    print("\nBuilding dataloaders ...")
    datasets = {
        split: SkinLesionDataset(data_root, split,
                                 transform=get_transforms(split))
        for split in ("train", "val", "test")
    }

    sampler = make_weighted_sampler(datasets["train"])

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=CFG.batch_size,
                            sampler=sampler, num_workers=CFG.num_workers,
                            pin_memory=True),
        "val":   DataLoader(datasets["val"],   batch_size=CFG.batch_size,
                            shuffle=False, num_workers=CFG.num_workers,
                            pin_memory=True),
        "test":  DataLoader(datasets["test"],  batch_size=CFG.batch_size,
                            shuffle=False, num_workers=CFG.num_workers,
                            pin_memory=True),
    }
    return loaders, datasets


# ─────────────────────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────────────────────

class SkinLesionModel(nn.Module):
    """
    EfficientNet-B4 backbone (ImageNet pretrained) with a custom head:
        GlobalAvgPool → Dense(512) + BN + ReLU → Dropout(0.4) → Dense(14)

    GlobalAvgPool is preferred over Flatten:
      - far fewer parameters
      - produces clean Grad-CAM heatmaps
    """
    def __init__(self, num_classes: int = CFG.num_classes,
                 backbone: str = CFG.backbone):
        super().__init__()

        # timm gives us the backbone with pretrained weights
        self.backbone = timm.create_model(
            backbone,
            pretrained=CFG.pretrained,
            num_classes=0,      # remove original classifier
            global_pool="avg",  # apply global average pooling
        )
        feature_dim = self.backbone.num_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # shape: (B, feature_dim)
        logits   = self.head(features)  # shape: (B, num_classes)
        return logits

    def freeze_backbone(self):
        """Phase 1: only train the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Phase 2: fine-tune the full model."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ─────────────────────────────────────────────────────────────
# 4. Loss — weighted cross-entropy for class imbalance
# ─────────────────────────────────────────────────────────────

def compute_class_weights(dataset: SkinLesionDataset) -> torch.Tensor:
    """
    Weight = total_samples / (num_classes * count_per_class)
    Rare classes get higher weight → larger gradient signal.
    """
    counts = dataset.class_counts().astype(np.float32)
    total  = counts.sum()
    weights = total / (CFG.num_classes * counts + 1e-6)
    weights = weights / weights.sum() * CFG.num_classes   # normalize
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# 5. Training utilities
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = CFG.patience,
                 mode: str = "max", delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.delta     = delta
        self.best      = None
        self.counter   = 0
        self.triggered = False

    def step(self, metric: float, model: nn.Module, path: Path) -> bool:
        if self.best is None:
            self.best = metric
            self._save(model, path)
            return False

        improved = (metric > self.best + self.delta) if self.mode == "max" \
                   else (metric < self.best - self.delta)

        if improved:
            self.best    = metric
            self.counter = 0
            self._save(model, path)
        else:
            self.counter += 1
            print(f"    EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

    @staticmethod
    def _save(model: nn.Module, path: Path):
        torch.save(model.state_dict(), path)
        print(f"    ✓ Saved best model → {path}")


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients during fine-tune
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return {"loss": running_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs,
                            multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "loss":     running_loss / total,
        "acc":      correct / total,
        "macro_f1": macro_f1,
        "auc":      auc,
        "labels":   all_labels,
        "preds":    all_preds,
        "probs":    all_probs,
    }


# ─────────────────────────────────────────────────────────────
# 6. Two-phase training
# ─────────────────────────────────────────────────────────────

def run_phase(model, loaders, criterion, optimizer, scheduler,
              n_epochs: int, phase_name: str, early_stop: EarlyStopping,
              device) -> list[dict]:
    history = []
    print(f"\n{'─'*60}")
    print(f"  Phase: {phase_name}")
    print(f"{'─'*60}")

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device)
        val_metrics   = evaluate(
            model, loaders["val"],   criterion, device)

        if scheduler is not None:
            scheduler.step()

        log = {
            "epoch":       epoch,
            "train_loss":  train_metrics["loss"],
            "train_acc":   train_metrics["acc"],
            "val_loss":    val_metrics["loss"],
            "val_acc":     val_metrics["acc"],
            "val_macro_f1":val_metrics["macro_f1"],
            "val_auc":     val_metrics["auc"],
        }
        history.append(log)

        print(f"  train loss {log['train_loss']:.4f}  acc {log['train_acc']:.4f}")
        print(f"  val   loss {log['val_loss']:.4f}  acc {log['val_acc']:.4f}  "
              f"F1 {log['val_macro_f1']:.4f}  AUC {log['val_auc']:.4f}")

        stopped = early_stop.step(
            val_metrics["macro_f1"], model, CFG.model_path)
        if stopped:
            print(f"\n  ⚡ Early stopping triggered at epoch {epoch}")
            break

    return history


def train(loaders: dict, datasets: dict) -> list[dict]:
    model = SkinLesionModel().to(CFG.device)
    class_weights = compute_class_weights(datasets["train"]).to(CFG.device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    early_stop    = EarlyStopping(patience=CFG.patience, mode="max")
    all_history   = []

    # ── Phase 1: frozen backbone ──────────────────────────────
    model.freeze_backbone()
    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr_head, weight_decay=CFG.weight_decay
    )
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=CFG.num_epochs_phase1, eta_min=1e-6)

    h1 = run_phase(model, loaders, criterion,
                   optimizer_p1, scheduler_p1,
                   CFG.num_epochs_phase1, "Phase 1 — head only",
                   early_stop, CFG.device)
    all_history.extend(h1)

    # ── Phase 2: full fine-tune ───────────────────────────────
    model.unfreeze_backbone()
    early_stop_p2 = EarlyStopping(patience=CFG.patience, mode="max")

    # Different LR for backbone vs head (backbone gets much smaller LR)
    optimizer_p2 = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": CFG.lr_backbone},
        {"params": model.head.parameters(),     "lr": CFG.lr_head_ft},
    ], weight_decay=CFG.weight_decay)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=CFG.num_epochs_phase2, eta_min=1e-7)

    h2 = run_phase(model, loaders, criterion,
                   optimizer_p2, scheduler_p2,
                   CFG.num_epochs_phase2, "Phase 2 — full fine-tune",
                   early_stop_p2, CFG.device)
    all_history.extend(h2)

    return model, all_history


# ─────────────────────────────────────────────────────────────
# 7. Evaluation & visualisation
# ─────────────────────────────────────────────────────────────

def plot_training_history(history: list[dict]):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df["train_loss"], label="train")
    axes[0].plot(df["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(df["train_acc"], label="train")
    axes[1].plot(df["val_acc"],   label="val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    axes[2].plot(df["val_macro_f1"], label="macro F1")
    axes[2].plot(df["val_auc"],      label="macro AUC")
    axes[2].set_title("Val metrics"); axes[2].legend(); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(CFG.output_dir / "training_history.png", dpi=150)
    plt.close()
    print(f"  Saved training curves → {CFG.output_dir / 'training_history.png'}")


def plot_confusion_matrix(labels, preds, class_names: list):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(CFG.output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {CFG.output_dir / 'confusion_matrix.png'}")


def full_evaluation(model, loaders, datasets) -> dict:
    """Load best weights and evaluate on the test set (sealed until now)."""
    print("\n" + "─"*60)
    print("  Final evaluation on TEST set")
    print("─"*60)

    model.load_state_dict(torch.load(CFG.model_path, map_location=CFG.device))
    class_weights = compute_class_weights(datasets["train"]).to(CFG.device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    metrics = evaluate(model, loaders["test"], criterion, CFG.device)

    print(f"\n  Test accuracy  : {metrics['acc']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Macro AUC-ROC  : {metrics['auc']:.4f}")

    print("\n  Per-class report:")
    report = classification_report(
        metrics["labels"], metrics["preds"],
        target_names=CFG.classes, zero_division=0
    )
    print(report)

    plot_confusion_matrix(metrics["labels"], metrics["preds"], CFG.classes)

    # Save report to file
    report_path = CFG.output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved report → {report_path}")

    return metrics


# ─────────────────────────────────────────────────────────────
# 8. Grad-CAM explainability
# ─────────────────────────────────────────────────────────────

def gradcam_explain(model, image_path: str, true_label: str = None):
    """
    Generates a Grad-CAM heatmap for one image.
    The heatmap shows which pixels drove the prediction.
    Clinicians can verify the model is attending to the lesion,
    not an artefact (ruler mark, hair, vignette border).
    """
    model.eval()

    # Grad-CAM targets the last conv block of EfficientNet
    # timm model: backbone.conv_head is the last conv layer
    target_layer = [model.backbone.conv_head]

    cam = GradCAM(model=model, target_layers=target_layer)

    # Preprocess image
    transform = get_transforms("val")
    raw_image = np.array(Image.open(image_path).convert("RGB"))
    raw_resized = np.array(Image.fromarray(raw_image).resize(
        (CFG.image_size, CFG.image_size))) / 255.0

    input_tensor = transform(image=raw_image)["image"].unsqueeze(0).to(CFG.device)

    # Run Grad-CAM
    logits    = model(input_tensor)
    pred_idx  = logits.argmax(dim=1).item()
    pred_cls  = CFG.idx_to_class[pred_idx]
    pred_prob = torch.softmax(logits, dim=1)[0, pred_idx].item()

    targets   = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Overlay heatmap on image
    visualisation = show_cam_on_image(
        raw_resized.astype(np.float32), grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_resized)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(visualisation)
    title = f"Pred: {pred_cls} ({pred_prob:.1%})"
    if true_label:
        title += f"\nTrue: {true_label}"
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    out_path = CFG.output_dir / f"gradcam_{Path(image_path).stem}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Grad-CAM saved → {out_path}")
    return pred_cls, pred_prob


# ─────────────────────────────────────────────────────────────
# 9. Inference — single image or batch
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, image_path: str, top_k: int = 3) -> list[dict]:
    """
    Returns top-k predicted classes with their probabilities.
    Flags uncertain predictions (max prob < 0.6) for clinician review.
    """
    model.eval()
    transform = get_transforms("val")
    raw_image = np.array(Image.open(image_path).convert("RGB"))
    tensor    = transform(image=raw_image)["image"].unsqueeze(0).to(CFG.device)

    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_indices = probs.argsort()[::-1][:top_k]
    results = [
        {"class": CFG.idx_to_class[i], "probability": float(probs[i])}
        for i in top_indices
    ]

    if results[0]["probability"] < 0.6:
        print("  ⚠  Low confidence — flag for clinician review")

    return results


# ─────────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  14-Class Skin Lesion Classifier")
    print(f"  Device : {CFG.device}")
    print(f"  Model  : {CFG.backbone}")
    print(f"{'='*60}\n")

    # Build data
    loaders, datasets = build_dataloaders(CFG.data_root)

    # Train
    model, history = train(loaders, datasets)

    # Plot curves
    plot_training_history(history)

    # Evaluate on sealed test set
    metrics = full_evaluation(model, loaders, datasets)

    # Grad-CAM example (replace with a real image path)
    example_image = "data/test/melanoma/example.jpg"
    if Path(example_image).exists():
        gradcam_explain(model, example_image, true_label="melanoma")

    print("\n  All done. Outputs saved to:", CFG.output_dir)
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()