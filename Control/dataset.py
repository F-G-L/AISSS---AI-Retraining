from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_CLASSES = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]

MALIGNANT_CLASSES = {
    "MEL",
    "BCC",
    "AKIEC",
}


class SkinLesionDataset(Dataset):
    """
    ISIC 2018 dataset loader using the shared experiment split.

    binary=False:
        Returns the original seven-class target.

    binary=True:
        Returns:
            0 = benign
            1 = malignant or concerning
    """

    def __init__(
        self,
        img_dir: str,
        gt_csv: str,
        split_csv: str,
        split: str,
        classes: list[str] | None = None,
        transform=None,
        binary: bool = False,
        selected_images: Iterable[str] | None = None,
    ):
        self.img_dir = img_dir
        self.classes = (
            classes
            if classes is not None
            else DEFAULT_CLASSES
        )

        self.class_to_index = {
            class_name: index
            for index, class_name
            in enumerate(self.classes)
        }

        self.transform = transform
        self.binary = binary

        ground_truth = pd.read_csv(
            gt_csv
        )

        missing_classes = [
            class_name
            for class_name in self.classes
            if class_name not in ground_truth.columns
        ]

        if missing_classes:
            raise ValueError(
                "Ground-truth CSV is missing: "
                + ", ".join(missing_classes)
            )

        ground_truth["label_name"] = (
            ground_truth[self.classes]
            .idxmax(axis=1)
        )

        split_dataframe = pd.read_csv(
            split_csv
        )

        required_columns = {
            "image",
            "split",
        }

        missing_columns = (
            required_columns
            - set(split_dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Split CSV is missing: "
                + ", ".join(
                    sorted(missing_columns)
                )
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
            dataframe["split"] == split
        ].copy()

        if selected_images is not None:
            selected_images = set(
                selected_images
            )

            dataframe = dataframe.loc[
                dataframe["image"].isin(
                    selected_images
                )
            ].copy()

        if dataframe.empty:
            raise ValueError(
                f"No images were found for "
                f"split={split!r}."
            )

        if binary:
            dataframe["target"] = (
                dataframe["label_name"]
                .isin(MALIGNANT_CLASSES)
                .astype(int)
            )
        else:
            dataframe["target"] = (
                dataframe["label_name"]
                .map(self.class_to_index)
            )

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        record = self.df.iloc[index]

        image_path = os.path.join(
            self.img_dir,
            record["image"] + ".jpg",
        )

        try:
            image = (
                Image
                .open(image_path)
                .convert("RGB")
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load image:\n"
                f"{image_path}"
            ) from error

        if self.transform is not None:
            image = self.transform(
                image
            )

        label = int(
            record["target"]
        )

        return image, label

    @property
    def labels(self):
        return self.df[
            "target"
        ].to_numpy()

    @property
    def image_names(self) -> list[str]:
        return self.df[
            "image"
        ].tolist()
