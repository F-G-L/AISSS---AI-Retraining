import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class SkinLesionDataset(Dataset):
    def __init__(self, img_dir, gt_csv, grouping_csv=None, classes=None, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(gt_csv)

        if grouping_csv is not None:
            df_grp = pd.read_csv(grouping_csv).set_index("image")
            self.df["lesion_id"] = self.df["image"].map(df_grp["lesion_id"])

        if classes is None:
            classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

        self.classes = classes
        self.class2idx = {c: i for i, c in enumerate(classes)}
        self.df["label_idx"] = self.df[classes].idxmax(axis=1).map(self.class2idx)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        img_name = record["image"] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error: skipping a bad image: {img_path} ({e})")
            return None, None

        if self.transform:
            image = self.transform(image)

        label = int(record["label_idx"])
        return image, label
