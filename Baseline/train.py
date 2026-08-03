import os
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torchvision.transforms as T
import numpy as np

from dataset import SkinLesionDataset
from model import get_model


def find_cnn_project():
    cur = os.path.abspath(__file__)
    while True:
        cur = os.path.dirname(cur)
        if os.path.basename(cur) == "Cnn_Project":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError("Could not find folder named 'Cnn_Project'")
        cur = parent


CNN_PROJECT = find_cnn_project()


# =========================
# Regularization knobs
# =========================
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 5
PATIENCE = 7
LR = 3e-5
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.30
FREEZE_BACKBONE_EPOCHS = 2
RESET_CLASSIFIER_ON_RESUME = True


MALIGNANT_CLASSES = {"MEL", "BCC", "AKIEC"}


def log_info(message):
    info_path = os.path.join(CNN_PROJECT, "info.txt")
    with open(info_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_classifier_module(model):
    for attr in ["fc", "classifier", "head"]:
        if hasattr(model, attr):
            return attr, getattr(model, attr)
    return None, None


def reset_classifier_head(model):
    attr, head = get_classifier_module(model)
    if head is None:
        print("Warning: could not find classifier head to reset.")
        return

    if isinstance(head, nn.Linear):
        nn.init.normal_(head.weight, mean=0.0, std=0.01)
        if head.bias is not None:
            nn.init.zeros_(head.bias)
    elif isinstance(head, nn.Sequential):
        for module in head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    else:
        print(f"Warning: unsupported head type for reset: {type(head)}")


def freeze_backbone(model):
    attr, head = get_classifier_module(model)
    head_param_ids = set()
    if head is not None:
        head_param_ids = {id(p) for p in head.parameters()}

    for param in model.parameters():
        param.requires_grad = id(param) in head_param_ids


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, label = self.base_dataset[base_idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def mixup_data(images, labels, alpha=0.2):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def format_confusion_matrix(cm, classes):
    header = "true\\pred," + ",".join(classes)
    rows = [header]
    for class_name, row in zip(classes, cm):
        rows.append(class_name + "," + ",".join(str(int(x)) for x in row))
    return "\n".join(rows)


def compute_cancer_recall(all_labels, all_preds, classes):
    malignant_indices = [i for i, name in enumerate(classes) if name in MALIGNANT_CLASSES]
    if not malignant_indices:
        return 0.0

    true_malignant = np.isin(all_labels, malignant_indices)
    pred_malignant = np.isin(all_preds, malignant_indices)

    tp = np.logical_and(true_malignant, pred_malignant).sum()
    fn = np.logical_and(true_malignant, np.logical_not(pred_malignant)).sum()
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def evaluate_model(model, val_loader, criterion, device, classes):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    cancer_recall = compute_cancer_recall(all_labels, all_preds, classes)

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "cancer_recall": cancer_recall,
    }


def main():
    set_seed(42)

    IMG_DIR = os.path.join(CNN_PROJECT, "isic2018", "ISIC2018_Task3_Training_Input")
    GT_CSV = os.path.join(CNN_PROJECT, "isic2018", "ISIC2018_Task3_Training_GroundTruth.csv")
    GROUPING_CSV = os.path.join(CNN_PROJECT, "isic2018", "ISIC2018_Task3_Training_LesionGroupings.csv")
    MODEL_PATH = os.path.join(CNN_PROJECT, "best_model.pth")

    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.20, scale=(0.02, 0.10), ratio=(0.3, 3.3), value='random'),
    ])

    # Validation must be deterministic.
    val_transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load base dataset without stochastic transforms, then wrap split-specific transforms.
    full_ds = SkinLesionDataset(
        img_dir=IMG_DIR,
        gt_csv=GT_CSV,
        grouping_csv=GROUPING_CSV,
        classes=classes,
        transform=None
    )

    df = full_ds.df

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["lesion_id"]))

    train_ds = TransformSubset(full_ds, train_idx, train_transform)
    val_ds = TransformSubset(full_ds, val_idx, val_transform)

    train_labels = df.iloc[train_idx]["label_idx"].values
    class_counts = np.bincount(train_labels, minlength=len(classes)).astype(np.float32)

    # Softer weighting: inverse sqrt is usually less unstable than raw inverse frequency.
    class_weights_np = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    class_weights_np = class_weights_np / class_weights_np.sum()
    sample_weights = class_weights_np[train_labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=len(classes), pretrained=True).to(device)

    if os.path.exists(MODEL_PATH):
        print("Found saved model.")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=True)

        if RESET_CLASSIFIER_ON_RESUME:
            print("Resetting classifier head to break out of an already-overfit solution.")
            reset_classifier_head(model)

    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    log_info("\n=== TRAINING RUN STARTED ===")
    log_info(
        f"Config | batch_size={BATCH_SIZE} | epochs={EPOCHS} | lr={LR} | weight_decay={WEIGHT_DECAY} "
        f"| label_smoothing={LABEL_SMOOTHING} | mixup_alpha={MIXUP_ALPHA} | freeze_backbone_epochs={FREEZE_BACKBONE_EPOCHS}"
    )

    if os.path.exists(MODEL_PATH):
        baseline_model = get_model(num_classes=len(classes), pretrained=True).to(device)
        baseline_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        baseline_metrics = evaluate_model(baseline_model, val_loader, criterion, device, classes)
        best_val_loss = baseline_metrics["val_loss"]
        print(
            "Existing saved model baseline | "
            f"val_loss={baseline_metrics['val_loss']:.4f} "
            f"val_acc={baseline_metrics['val_acc']:.4f} "
            f"precision={baseline_metrics['precision']:.4f} "
            f"recall={baseline_metrics['recall']:.4f} "
            f"f1={baseline_metrics['f1']:.4f} "
            f"cancer_recall={baseline_metrics['cancer_recall']:.4f}"
        )
        log_info(
            "Existing saved model baseline | "
            f"val_loss={baseline_metrics['val_loss']:.6f} | "
            f"val_acc={baseline_metrics['val_acc']:.6f} | "
            f"precision={baseline_metrics['precision']:.6f} | "
            f"recall={baseline_metrics['recall']:.6f} | "
            f"f1={baseline_metrics['f1']:.6f} | "
            f"cancer_recall={baseline_metrics['cancer_recall']:.6f}"
        )
        log_info("Baseline confusion matrix:\n" + format_confusion_matrix(baseline_metrics["confusion_matrix"], classes))

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        if epoch == FREEZE_BACKBONE_EPOCHS:
            print("Unfreezing full model.")
            unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
            )

        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=MIXUP_ALPHA)

            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"{batch_idx}/{total_batches} batches")

        epoch_loss = running_loss / len(train_loader.dataset)
        metrics = evaluate_model(model, val_loader, criterion, device, classes)
        scheduler.step(metrics["val_loss"])

        print(
            f"train_loss={epoch_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_acc={metrics['val_acc']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"cancer_recall={metrics['cancer_recall']:.4f}"
        )

        print("Confusion matrix:")
        print(metrics["confusion_matrix"])

        log_info(
            f"Epoch {epoch + 1}/{EPOCHS} | train_loss={epoch_loss:.6f} | "
            f"val_loss={metrics['val_loss']:.6f} | "
            f"val_acc={metrics['val_acc']:.6f} | "
            f"precision={metrics['precision']:.6f} | "
            f"recall={metrics['recall']:.6f} | "
            f"f1={metrics['f1']:.6f} | "
            f"cancer_recall={metrics['cancer_recall']:.6f}"
        )
        log_info("Confusion matrix:\n" + format_confusion_matrix(metrics["confusion_matrix"], classes))

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model")
            log_info(f"Saved best model at epoch {epoch + 1} with val_loss={metrics['val_loss']:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                log_info("Early stopping triggered")
                break

    log_info("=== TRAINING RUN ENDED ===\n")


if __name__ == "__main__":
    main()
