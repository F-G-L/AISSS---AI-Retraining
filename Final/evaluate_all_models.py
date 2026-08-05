from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
)


CLASSES = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]

BATCH_SIZE = 16
NUM_WORKERS = 0


class TestDataset(Dataset):
    def __init__(
        self,
        image_folder: Path,
        ground_truth_path: Path,
        split_path: Path,
        transform,
    ):
        ground_truth = pd.read_csv(
            ground_truth_path
        )

        ground_truth["label_name"] = (
            ground_truth[CLASSES]
            .idxmax(axis=1)
        )

        class_to_index = {
            class_name: index
            for index, class_name
            in enumerate(CLASSES)
        }

        ground_truth["target"] = (
            ground_truth["label_name"]
            .map(class_to_index)
        )

        split_dataframe = pd.read_csv(
            split_path
        )

        dataframe = ground_truth.merge(
            split_dataframe[
                [
                    "image",
                    "split",
                ]
            ],
            on="image",
            how="inner",
            validate="one_to_one",
        )

        dataframe = dataframe.loc[
            dataframe["split"] == "test"
        ].copy()

        if dataframe.empty:
            raise ValueError(
                "No test images were found."
            )

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.image_folder = (
            image_folder
        )

        self.transform = transform

    def __len__(self):
        return len(
            self.dataframe
        )

    def __getitem__(self, index):
        record = (
            self.dataframe.iloc[index]
        )

        image_path = (
            self.image_folder
            / (
                record["image"]
                + ".jpg"
            )
        )

        image = (
            Image
            .open(image_path)
            .convert("RGB")
        )

        image = self.transform(
            image
        )

        return (
            image,
            int(record["target"]),
        )


def create_model():
    try:
        model = models.resnet50(
            weights=None
        )
    except TypeError:
        model = models.resnet50(
            pretrained=False
        )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASSES),
    )

    return model


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
            f"Unsupported checkpoint:\n"
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


def evaluate_model(
    model,
    loader,
    device,
):
    model.eval()

    criterion = (
        nn.CrossEntropyLoss()
    )

    total_loss = 0.0

    all_labels = []
    all_predictions = []

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

    (
        per_class_precision,
        per_class_recall,
        per_class_f1,
        per_class_support,
    ) = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        labels=list(
            range(len(CLASSES))
        ),
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels_array,
        predictions_array,
        labels=list(
            range(len(CLASSES))
        ),
    )

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_totals != 0,
    )

    per_class_results = {}

    for index, class_name in enumerate(
        CLASSES
    ):
        per_class_results[
            class_name
        ] = {
            "precision": float(
                per_class_precision[index]
            ),
            "recall": float(
                per_class_recall[index]
            ),
            "f1": float(
                per_class_f1[index]
            ),
            "support": int(
                per_class_support[index]
            ),
        }

    return {
        "test_loss": (
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
            matrix.tolist()
        ),
        "normalized_confusion_matrix": (
            normalized_matrix.tolist()
        ),
        "per_class": (
            per_class_results
        ),
    }


def main() -> None:
    projects_folder = (
        Path(__file__)
        .resolve()
        .parent
    )

    original_project = (
        projects_folder
        / "Cnn_Project"
    )

    dataset_folder = (
        original_project
        / "isic2018"
    )

    image_folder = (
        dataset_folder
        / "ISIC2018_Task3_Training_Input"
    )

    ground_truth_path = (
        dataset_folder
        / "ISIC2018_Task3_Training_GroundTruth.csv"
    )

    split_path = (
        projects_folder
        / "experiment_split.csv"
    )

    checkpoints = {
        "baseline": (
            original_project
            / "best_model.pth"
        ),
        "control": (
            projects_folder
            / "Cnn_Project_Control"
            / "control_final_7class.pth"
        ),
        "simple": (
            projects_folder
            / "Cnn_Project_Simple"
            / "simple_final_7class.pth"
        ),
    }

    if not split_path.exists():
        raise FileNotFoundError(
            f"Shared split not found:\n"
            f"{split_path}"
        )

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

    test_dataset = TestDataset(
        image_folder=image_folder,
        ground_truth_path=(
            ground_truth_path
        ),
        split_path=split_path,
        transform=validation_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_folder = (
        projects_folder
        / "experiment_results"
    )

    output_folder.mkdir(
        exist_ok=True
    )

    all_results = {}
    summary_rows = []

    for (
        model_name,
        checkpoint_path,
    ) in checkpoints.items():

        if not checkpoint_path.exists():
            print()
            print(
                f"Skipping {model_name}. "
                f"Checkpoint not found:"
            )
            print(
                checkpoint_path
            )
            continue

        print()
        print(
            f"Evaluating {model_name}..."
        )

        model = create_model().to(
            device
        )

        model.load_state_dict(
            load_checkpoint(
                checkpoint_path,
                device,
            ),
            strict=True,
        )

        metrics = evaluate_model(
            model,
            test_loader,
            device,
        )

        all_results[
            model_name
        ] = metrics

        summary_rows.append({
            "model": model_name,
            "test_loss": (
                metrics["test_loss"]
            ),
            "accuracy": (
                metrics["accuracy"]
            ),
            "balanced_accuracy": (
                metrics[
                    "balanced_accuracy"
                ]
            ),
            "macro_precision": (
                metrics[
                    "macro_precision"
                ]
            ),
            "macro_recall": (
                metrics[
                    "macro_recall"
                ]
            ),
            "macro_f1": (
                metrics["macro_f1"]
            ),
        })

        confusion_dataframe = (
            pd.DataFrame(
                metrics[
                    "confusion_matrix"
                ],
                index=CLASSES,
                columns=CLASSES,
            )
        )

        confusion_dataframe.index.name = (
            "true"
        )

        confusion_dataframe.columns.name = (
            "predicted"
        )

        confusion_dataframe.to_csv(
            output_folder
            / (
                f"{model_name}"
                f"_confusion_matrix.csv"
            )
        )

        normalized_dataframe = (
            pd.DataFrame(
                metrics[
                    "normalized_confusion_matrix"
                ],
                index=CLASSES,
                columns=CLASSES,
            )
        )

        normalized_dataframe.index.name = (
            "true"
        )

        normalized_dataframe.columns.name = (
            "predicted"
        )

        normalized_dataframe.to_csv(
            output_folder
            / (
                f"{model_name}"
                f"_normalized_confusion_matrix.csv"
            )
        )

        per_class_dataframe = (
            pd.DataFrame(
                metrics["per_class"]
            )
            .transpose()
        )

        per_class_dataframe.index.name = (
            "class"
        )

        per_class_dataframe.to_csv(
            output_folder
            / (
                f"{model_name}"
                f"_per_class_metrics.csv"
            )
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )

    if not all_results:
        raise RuntimeError(
            "No checkpoints were available "
            "for evaluation."
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        output_folder
        / "summary.csv",
        index=False,
    )

    with (
        output_folder
        / "all_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            indent=2,
        )

    print()
    print(
        "Results saved inside:"
    )

    print(
        output_folder
    )


if __name__ == "__main__":
    main()
