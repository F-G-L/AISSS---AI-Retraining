from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
)

from dataset import (
    MALIGNANT_CLASSES,
    SkinLesionDataset,
)

from model_Simple import get_model


SEED = 42

CLASSES = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]

BINARY_CLASSES = [
    "BENIGN",
    "MALIGNANT_OR_CONCERNING",
]

BATCH_SIZE = 16
NUM_WORKERS = 0

# Must match the Control training budgets.
BINARY_STEPS = 1000
FINAL_SEVEN_CLASS_STEPS = 1000

EVALUATE_EVERY_STEPS = 100

# Maximum number of images from each binary group.
BINARY_TRAIN_MAX_PER_GROUP = 600
BINARY_VALIDATION_MAX_PER_GROUP = 200

BACKBONE_LR = 3e-5
HEAD_LR = 3e-4

WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.30


def find_project_folder() -> Path:
    current = (
        Path(__file__)
        .resolve()
        .parent
    )

    while True:
        if current.name == "Cnn_Project_Simple":
            return current

        if current.parent == current:
            raise FileNotFoundError(
                "Could not find "
                "Cnn_Project_Simple."
            )

        current = current.parent


PROJECT = find_project_folder()
PROJECTS_FOLDER = PROJECT.parent

IMAGE_FOLDER = (
    PROJECT
    / "isic2018"
    / "ISIC2018_Task3_Training_Input"
)

GROUND_TRUTH_PATH = (
    PROJECT
    / "isic2018"
    / "ISIC2018_Task3_Training_GroundTruth.csv"
)

SPLIT_PATH = (
    PROJECTS_FOLDER
    / "experiment_split.csv"
)

START_MODEL_PATH = (
    PROJECT
    / "best_model_Simple.pth"
)

BINARY_MODEL_PATH = (
    PROJECT
    / "simple_binary_stage.pth"
)

FINAL_MODEL_PATH = (
    PROJECT
    / "simple_final_7class.pth"
)

BINARY_SELECTION_PATH = (
    PROJECT
    / "simple_binary_selection.csv"
)

LOG_PATH = (
    PROJECT
    / "simple_training_log.txt"
)

METRICS_PATH = (
    PROJECT
    / "simple_validation_metrics.json"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log(message: str) -> None:
    print(message)

    with LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            message + "\n"
        )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        checkpoint = checkpoint[
            "model_state_dict"
        ]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint format:\n"
            f"{checkpoint_path}"
        )

    if (
        checkpoint
        and all(
            key.startswith("module.")
            for key in checkpoint
        )
    ):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value
            in checkpoint.items()
        }

    return checkpoint


def configure_trainable_layers(
    model: nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def set_training_mode(
    model: nn.Module,
) -> None:
    model.train()

    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()


def build_optimizer(
    model: nn.Module,
):
    return optim.AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": BACKBONE_LR,
            },
            {
                "params": model.fc.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


def make_transforms():
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(
            224,
            scale=(0.75, 1.0),
        ),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
        T.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
        T.RandomErasing(
            p=0.20,
            scale=(0.02, 0.10),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])

    validation_transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    return (
        train_transform,
        validation_transform,
    )


def evenly_sample_classes(
    dataframe: pd.DataFrame,
    class_names: list[str],
    total_images: int,
    seed: int,
):
    """
    Distribute samples as evenly as possible
    across the original seven-class labels.
    """

    pools: dict[str, list[str]] = {}

    for offset, class_name in enumerate(
        class_names
    ):
        image_names = (
            dataframe.loc[
                dataframe["label"] == class_name,
                "image",
            ]
            .drop_duplicates()
            .tolist()
        )

        random.Random(
            seed + offset
        ).shuffle(
            image_names
        )

        pools[class_name] = image_names

    allocations = {
        class_name: 0
        for class_name in class_names
    }

    selected_count = 0

    while selected_count < total_images:
        made_progress = False

        for class_name in class_names:
            if selected_count >= total_images:
                break

            available = len(
                pools[class_name]
            )

            if (
                allocations[class_name]
                < available
            ):
                allocations[class_name] += 1
                selected_count += 1
                made_progress = True

        if not made_progress:
            break

    selected_images: list[str] = []

    for class_name in class_names:
        number_to_take = (
            allocations[class_name]
        )

        selected_images.extend(
            pools[class_name][
                :number_to_take
            ]
        )

    return (
        selected_images,
        allocations,
    )


def select_binary_images(
    split_name: str,
    maximum_per_binary_group: int,
    seed: int,
):
    split_dataframe = pd.read_csv(
        SPLIT_PATH
    )

    split_dataframe = split_dataframe.loc[
        split_dataframe["split"]
        == split_name
    ].copy()

    malignant_classes = [
        class_name
        for class_name in CLASSES
        if class_name in MALIGNANT_CLASSES
    ]

    benign_classes = [
        class_name
        for class_name in CLASSES
        if class_name not in MALIGNANT_CLASSES
    ]

    malignant_available = int(
        split_dataframe["label"]
        .isin(malignant_classes)
        .sum()
    )

    benign_available = int(
        split_dataframe["label"]
        .isin(benign_classes)
        .sum()
    )

    images_per_binary_group = min(
        maximum_per_binary_group,
        malignant_available,
        benign_available,
    )

    if images_per_binary_group <= 0:
        raise ValueError(
            f"Could not construct binary "
            f"subset for split={split_name}."
        )

    (
        malignant_images,
        malignant_allocations,
    ) = evenly_sample_classes(
        dataframe=split_dataframe,
        class_names=malignant_classes,
        total_images=images_per_binary_group,
        seed=seed,
    )

    (
        benign_images,
        benign_allocations,
    ) = evenly_sample_classes(
        dataframe=split_dataframe,
        class_names=benign_classes,
        total_images=images_per_binary_group,
        seed=seed + 1000,
    )

    selected_images = (
        malignant_images
        + benign_images
    )

    details = {
        "split": split_name,
        "images_per_binary_group": (
            images_per_binary_group
        ),
        "malignant_allocations": (
            malignant_allocations
        ),
        "benign_allocations": (
            benign_allocations
        ),
    }

    return (
        selected_images,
        details,
    )


def make_binary_loaders():
    (
        train_transform,
        validation_transform,
    ) = make_transforms()

    (
        train_images,
        train_details,
    ) = select_binary_images(
        split_name="train",
        maximum_per_binary_group=(
            BINARY_TRAIN_MAX_PER_GROUP
        ),
        seed=SEED,
    )

    (
        validation_images,
        validation_details,
    ) = select_binary_images(
        split_name="val",
        maximum_per_binary_group=(
            BINARY_VALIDATION_MAX_PER_GROUP
        ),
        seed=SEED + 1,
    )

    log(
        "Binary training selection: "
        + json.dumps(
            train_details,
            sort_keys=True,
        )
    )

    log(
        "Binary validation selection: "
        + json.dumps(
            validation_details,
            sort_keys=True,
        )
    )

    split_dataframe = (
        pd.read_csv(SPLIT_PATH)
        .set_index("image")
    )

    selection_rows = []

    for split_name, image_names in [
        ("train", train_images),
        ("val", validation_images),
    ]:
        for image_name in image_names:
            original_label = (
                split_dataframe.loc[
                    image_name,
                    "label",
                ]
            )

            if original_label in MALIGNANT_CLASSES:
                binary_label = (
                    "MALIGNANT_OR_CONCERNING"
                )
            else:
                binary_label = "BENIGN"

            selection_rows.append({
                "image": image_name,
                "split": split_name,
                "original_label": (
                    original_label
                ),
                "binary_label": (
                    binary_label
                ),
            })

    pd.DataFrame(
        selection_rows
    ).to_csv(
        BINARY_SELECTION_PATH,
        index=False,
    )

    train_dataset = SkinLesionDataset(
        img_dir=str(IMAGE_FOLDER),
        gt_csv=str(GROUND_TRUTH_PATH),
        split_csv=str(SPLIT_PATH),
        split="train",
        classes=CLASSES,
        transform=train_transform,
        binary=True,
        selected_images=train_images,
    )

    validation_dataset = SkinLesionDataset(
        img_dir=str(IMAGE_FOLDER),
        gt_csv=str(GROUND_TRUTH_PATH),
        split_csv=str(SPLIT_PATH),
        split="val",
        classes=CLASSES,
        transform=validation_transform,
        binary=True,
        selected_images=validation_images,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        validation_loader,
    )


def make_seven_class_loaders():
    (
        train_transform,
        validation_transform,
    ) = make_transforms()

    train_dataset = SkinLesionDataset(
        img_dir=str(IMAGE_FOLDER),
        gt_csv=str(GROUND_TRUTH_PATH),
        split_csv=str(SPLIT_PATH),
        split="train",
        classes=CLASSES,
        transform=train_transform,
        binary=False,
    )

    validation_dataset = SkinLesionDataset(
        img_dir=str(IMAGE_FOLDER),
        gt_csv=str(GROUND_TRUTH_PATH),
        split_csv=str(SPLIT_PATH),
        split="val",
        classes=CLASSES,
        transform=validation_transform,
        binary=False,
    )

    class_counts = np.bincount(
        train_dataset.labels,
        minlength=len(CLASSES),
    ).astype(np.float32)

    class_weights = (
        1.0
        / np.sqrt(
            np.maximum(
                class_counts,
                1.0,
            )
        )
    )

    class_weights = (
        class_weights
        / class_weights.mean()
    )

    sample_weights = (
        class_weights[
            train_dataset.labels
        ]
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        validation_loader,
        torch.tensor(
            class_weights,
            dtype=torch.float32,
        ),
    )


def mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
):
    if MIXUP_ALPHA <= 0:
        return (
            images,
            labels,
            labels,
            1.0,
        )

    lam = float(
        np.random.beta(
            MIXUP_ALPHA,
            MIXUP_ALPHA,
        )
    )

    permutation = torch.randperm(
        images.size(0),
        device=images.device,
    )

    mixed_images = (
        lam * images
        + (1.0 - lam)
        * images[permutation]
    )

    return (
        mixed_images,
        labels,
        labels[permutation],
        lam,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    number_of_classes: int,
):
    model.eval()

    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    labels_array = np.asarray(
        all_labels
    )

    predictions_array = np.asarray(
        all_predictions
    )

    return {
        "loss": (
            total_loss
            / len(loader.dataset)
        ),
        "accuracy": accuracy_score(
            labels_array,
            predictions_array,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                labels_array,
                predictions_array,
            )
        ),
        "macro_precision": precision_score(
            labels_array,
            predictions_array,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            labels_array,
            predictions_array,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            labels_array,
            predictions_array,
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": (
            confusion_matrix(
                labels_array,
                predictions_array,
                labels=list(
                    range(number_of_classes)
                ),
            )
            .tolist()
        ),
    }


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    number_of_classes: int,
    number_of_steps: int,
    output_path: Path,
    stage_name: str,
):
    configure_trainable_layers(
        model
    )

    optimizer = build_optimizer(
        model
    )

    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                number_of_steps,
                1,
            ),
        )
    )

    training_iterator = iter(
        train_loader
    )

    best_balanced_accuracy = float(
        "-inf"
    )

    best_metrics = None

    for step in range(
        1,
        number_of_steps + 1,
    ):
        try:
            images, labels = next(
                training_iterator
            )
        except StopIteration:
            training_iterator = iter(
                train_loader
            )

            images, labels = next(
                training_iterator
            )

        set_training_mode(
            model
        )

        images = images.to(device)
        labels = labels.to(device)

        (
            mixed_images,
            labels_a,
            labels_b,
            lam,
        ) = mixup_data(
            images,
            labels,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            mixed_images
        )

        loss = (
            lam
            * criterion(
                outputs,
                labels_a,
            )
            + (1.0 - lam)
            * criterion(
                outputs,
                labels_b,
            )
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        should_evaluate = (
            step % EVALUATE_EVERY_STEPS == 0
            or step == number_of_steps
        )

        if should_evaluate:
            metrics = evaluate_model(
                model,
                validation_loader,
                criterion,
                device,
                number_of_classes,
            )

            log(
                f"{stage_name} "
                f"| step={step}/{number_of_steps} "
                f"| train_loss={loss.item():.6f} "
                f"| val_loss={metrics['loss']:.6f} "
                f"| val_accuracy="
                f"{metrics['accuracy']:.6f} "
                f"| balanced_accuracy="
                f"{metrics['balanced_accuracy']:.6f} "
                f"| macro_f1="
                f"{metrics['macro_f1']:.6f}"
            )

            log(
                "Confusion matrix:\n"
                + np.array2string(
                    np.asarray(
                        metrics[
                            "confusion_matrix"
                        ]
                    )
                )
            )

            if (
                metrics[
                    "balanced_accuracy"
                ]
                > best_balanced_accuracy
            ):
                best_balanced_accuracy = (
                    metrics[
                        "balanced_accuracy"
                    ]
                )

                best_metrics = metrics

                torch.save(
                    model.state_dict(),
                    output_path,
                )

                log(
                    f"Saved best "
                    f"{stage_name} model: "
                    f"{output_path.name}"
                )

    if best_metrics is None:
        raise RuntimeError(
            f"No model was saved during "
            f"{stage_name}."
        )

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Simple output files.",
    )

    args = parser.parse_args()

    set_seed(SEED)

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Shared split not found:\n"
            f"{SPLIT_PATH}\n\n"
            f"Run create_experiment_split.py first."
        )

    if not START_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Starting model not found:\n"
            f"{START_MODEL_PATH}"
        )

    existing_outputs = [
        path
        for path in [
            BINARY_MODEL_PATH,
            FINAL_MODEL_PATH,
            BINARY_SELECTION_PATH,
        ]
        if path.exists()
    ]

    if existing_outputs and not args.overwrite:
        print(
            "Simple output files already exist:"
        )

        for path in existing_outputs:
            print(path)

        print()
        print(
            "Nothing was changed. Run with "
            "--overwrite to replace them."
        )
        return

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log(
        "=== SIMPLE TRAINING START ==="
    )

    log(
        f"Device: {device}"
    )

    # Stage 1: binary training.
    (
        binary_train_loader,
        binary_validation_loader,
    ) = make_binary_loaders()

    binary_model = get_model(
        num_classes=len(CLASSES),
        pretrained=False,
    ).to(device)

    binary_model.load_state_dict(
        load_checkpoint(
            START_MODEL_PATH,
            device,
        ),
        strict=True,
    )

    # Remove the old seven-class head.
    binary_model.fc = nn.Linear(
        binary_model.fc.in_features,
        2,
    ).to(device)

    binary_criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )

    binary_metrics = train_stage(
        model=binary_model,
        train_loader=binary_train_loader,
        validation_loader=(
            binary_validation_loader
        ),
        criterion=binary_criterion,
        device=device,
        number_of_classes=2,
        number_of_steps=BINARY_STEPS,
        output_path=BINARY_MODEL_PATH,
        stage_name="simple_binary_stage",
    )

    # Stage 2: return to seven classes.
    (
        seven_class_train_loader,
        seven_class_validation_loader,
        class_weights,
    ) = make_seven_class_loaders()

    final_model = get_model(
        num_classes=2,
        pretrained=False,
    ).to(device)

    final_model.load_state_dict(
        load_checkpoint(
            BINARY_MODEL_PATH,
            device,
        ),
        strict=True,
    )

    # Remove the binary head and create
    # a new seven-class head.
    final_model.fc = nn.Linear(
        final_model.fc.in_features,
        len(CLASSES),
    ).to(device)

    class_weights = class_weights.to(
        device
    )

    seven_class_criterion = (
        nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=(
                LABEL_SMOOTHING
            ),
        )
    )

    final_metrics = train_stage(
        model=final_model,
        train_loader=(
            seven_class_train_loader
        ),
        validation_loader=(
            seven_class_validation_loader
        ),
        criterion=seven_class_criterion,
        device=device,
        number_of_classes=len(CLASSES),
        number_of_steps=(
            FINAL_SEVEN_CLASS_STEPS
        ),
        output_path=FINAL_MODEL_PATH,
        stage_name=(
            "simple_final_7class_stage"
        ),
    )

    results = {
        "binary_stage": binary_metrics,
        "final_seven_class_stage": (
            final_metrics
        ),
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    log(
        "Final Simple checkpoint:"
    )

    log(
        str(FINAL_MODEL_PATH)
    )

    log(
        "=== SIMPLE TRAINING END ==="
    )


if __name__ == "__main__":
    main()
