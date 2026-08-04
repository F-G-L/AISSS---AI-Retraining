from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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


def can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts()

    return (
        len(counts) > 1
        and int(counts.min()) >= 2
    )


def split_dataframe(
    dataframe: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    stratify_labels = None

    if can_stratify(dataframe["label"]):
        stratify_labels = dataframe["label"]

    first, second = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    return first.copy(), second.copy()


def majority_label(labels: pd.Series) -> str:
    modes = labels.mode()

    if len(modes) == 0:
        return str(labels.iloc[0])

    return str(modes.iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing experiment_split.csv file.",
    )

    args = parser.parse_args()

    projects_folder = Path(__file__).resolve().parent

    dataset_folder = (
        projects_folder
        / "Cnn_Project"
        / "isic2018"
    )

    ground_truth_path = (
        dataset_folder
        / "ISIC2018_Task3_Training_GroundTruth.csv"
    )

    grouping_path = (
        dataset_folder
        / "ISIC2018_Task3_Training_LesionGroupings.csv"
    )

    output_path = (
        projects_folder
        / "experiment_split.csv"
    )

    if output_path.exists() and not args.overwrite:
        print("The shared split already exists:")
        print(output_path)
        print()
        print(
            "Nothing was changed. Use --overwrite only "
            "if you intentionally want a new split."
        )
        return

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth CSV was not found:\n"
            f"{ground_truth_path}"
        )

    if not grouping_path.exists():
        raise FileNotFoundError(
            f"Lesion-grouping CSV was not found:\n"
            f"{grouping_path}"
        )

    ground_truth = pd.read_csv(
        ground_truth_path
    )

    grouping = pd.read_csv(
        grouping_path
    )

    missing_classes = [
        class_name
        for class_name in CLASSES
        if class_name not in ground_truth.columns
    ]

    if missing_classes:
        raise ValueError(
            "Ground-truth CSV is missing these columns: "
            + ", ".join(missing_classes)
        )

    ground_truth["label"] = (
        ground_truth[CLASSES]
        .idxmax(axis=1)
    )

    merged = ground_truth[
        ["image", "label"]
    ].merge(
        grouping[
            ["image", "lesion_id"]
        ],
        on="image",
        how="left",
        validate="one_to_one",
    )

    # An image without a lesion ID becomes its own group.
    merged["lesion_id"] = (
        merged["lesion_id"]
        .fillna(merged["image"])
    )

    # Produce one row per physical lesion.
    lesions = (
        merged
        .groupby(
            "lesion_id",
            as_index=False,
        )
        .agg(
            label=(
                "label",
                majority_label,
            )
        )
    )

    # 70% training, 30% temporary.
    train_lesions, temporary_lesions = (
        split_dataframe(
            lesions,
            test_size=0.30,
            seed=SEED,
        )
    )

    # Split temporary data equally:
    # 15% validation and 15% test.
    validation_lesions, test_lesions = (
        split_dataframe(
            temporary_lesions,
            test_size=0.50,
            seed=SEED + 1,
        )
    )

    split_by_lesion: dict[str, str] = {}

    for lesion_id in train_lesions["lesion_id"]:
        split_by_lesion[lesion_id] = "train"

    for lesion_id in validation_lesions["lesion_id"]:
        split_by_lesion[lesion_id] = "val"

    for lesion_id in test_lesions["lesion_id"]:
        split_by_lesion[lesion_id] = "test"

    merged["split"] = (
        merged["lesion_id"]
        .map(split_by_lesion)
    )

    if merged["split"].isna().any():
        raise RuntimeError(
            "Some images did not receive a split."
        )

    output = merged[
        [
            "image",
            "lesion_id",
            "label",
            "split",
        ]
    ].copy()

    # Confirm that no lesion occurs in multiple splits.
    maximum_split_count = (
        output
        .groupby("lesion_id")["split"]
        .nunique()
        .max()
    )

    if int(maximum_split_count) != 1:
        raise RuntimeError(
            "Lesion leakage was detected."
        )

    output.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Shared experiment split created:")
    print(output_path)
    print()
    print("Image counts by split and class:")
    print(
        output
        .groupby(
            ["split", "label"]
        )
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=CLASSES,
            fill_value=0,
        )
    )
    print()
    print(
        "Verified: every lesion appears "
        "in exactly one split."
    )


if __name__ == "__main__":
    main()

