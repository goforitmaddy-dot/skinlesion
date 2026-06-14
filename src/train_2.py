"""
Skin Lesion Classifier - Training Script
Dataset: 14-class skin lesion dataset (HAM10000 + MSLDv2.0)
Approach: Transfer Learning with EfficientNetB3
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B3_Weights

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir": "../data/skin_dataset/",          # Root dir with train/ val/ test/ subfolders
    "output_dir": "./outputs",
    "model_name": "efficientnet_b3",
    "num_classes": 14,
    "image_size": 224,
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 42,
    "use_class_weights": True,        # Handles class imbalance
    "unfreeze_after_epoch": 5,        # Fine-tune backbone after this epoch
}

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like-lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "Healthy",
    "HFMD",
    "Measles",
    "Melanocytic nevi",
    "Melanoma",
    "Monkeypox",
    "Squamous cell carcinoma",
    "Vascular lesions",
]


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms(image_size: int):
    """
    Training uses aggressive augmentation to improve generalization.
    Validation/test uses only normalization (no augmentation).
    ImageNet mean/std used since we start from pretrained weights.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


# ─────────────────────────────────────────────
# DATASET & DATALOADER
# ─────────────────────────────────────────────
def get_dataloaders(cfg: dict):
    train_tf, val_tf = get_transforms(cfg["image_size"])

    train_ds = datasets.ImageFolder(os.path.join(cfg["data_dir"], "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(cfg["data_dir"], "val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(cfg["data_dir"], "test"),  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Test  samples : {len(test_ds)}")
    print(f"Classes       : {train_ds.classes}")

    return train_loader, val_loader, test_loader, train_ds


# ─────────────────────────────────────────────
# CLASS WEIGHTS (handles imbalance)
# ─────────────────────────────────────────────
def compute_class_weights(dataset, num_classes: int, device):
    """
    Inverse-frequency weighting so rare classes get more attention.
    """
    targets = np.array(dataset.targets)
    counts  = np.bincount(targets, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes          # normalize
    return torch.tensor(weights, dtype=torch.float).to(device)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int, freeze_backbone: bool = True):
    """
    EfficientNetB3 pretrained on ImageNet.
    - Phase 1 (frozen): only the new classifier head is trained.
    - Phase 2 (unfrozen): entire network fine-tuned with a low LR.
    """
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model


def unfreeze_backbone(model):
    """Call this after warm-up phase to fine-tune entire network."""
    for param in model.parameters():
        param.requires_grad = True
    print(">>> Backbone unfrozen — fine-tuning entire network.")


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_history(history: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Saved training curves → {out_dir}/training_curves.png")


def plot_confusion_matrix(labels, preds, class_names: list, out_dir: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {out_dir}/confusion_matrix.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ── Data ──────────────────────────────────
    train_loader, val_loader, test_loader, train_ds = get_dataloaders(CONFIG)

    # ── Class weights ─────────────────────────
    class_weights = None
    if CONFIG["use_class_weights"]:
        class_weights = compute_class_weights(train_ds, CONFIG["num_classes"], device)
        print("Class weights:", class_weights.cpu().numpy().round(3))

    # ── Model ─────────────────────────────────
    model = build_model(CONFIG["num_classes"], freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])

    # ── Training ──────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(CONFIG["output_dir"], "best_model.pth")

    for epoch in range(1, CONFIG["num_epochs"] + 1):

        # Unfreeze backbone after warm-up
        if epoch == CONFIG["unfreeze_after_epoch"] + 1:
            unfreeze_backbone(model)
            # Lower LR for fine-tuning entire network
            for g in optimizer.param_groups:
                g["lr"] = CONFIG["learning_rate"] * 0.1

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:>3}/{CONFIG['num_epochs']} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")

    # ── Evaluation on test set ─────────────────
    print("\n── Test Set Evaluation ──")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES)
    print("\nClassification Report:\n", report)

    report_path = os.path.join(CONFIG["output_dir"], "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # ── Save plots & history ───────────────────
    plot_history(history, CONFIG["output_dir"])
    plot_confusion_matrix(test_labels, test_preds, CLASS_NAMES, CONFIG["output_dir"])

    with open(os.path.join(CONFIG["output_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
