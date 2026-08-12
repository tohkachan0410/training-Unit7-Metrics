from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path("/home/u5450760/train_6")


def resolve_image_path(path_str):
    """
    依序嘗試：
    1. CSV 本身是絕對路徑
    2. /home/u5450760/train_6/<CSV路徑>
    3. /home/u5450760/train_6/data/raw/<檔名>
    """
    path = Path(str(path_str))

    if path.is_absolute() and path.exists():
        return path

    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate

    candidate = PROJECT_ROOT / "data" / "raw" / path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"找不到圖片：{path_str}\n"
        f"也找不到：{PROJECT_ROOT / path}\n"
        f"也找不到：{PROJECT_ROOT / 'data' / 'raw' / path.name}"
    )


def get_train_transform(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.85, 1.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
        ),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class RankingDataset(Dataset):
    def __init__(
        self,
        csv_path,
        transform=None,
        target_column="quality_score",
        require_target=True,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        self.target_column = target_column
        self.require_target = require_target

        required = {"image_id", "image_path"}

        if require_target:
            required.add(target_column)

        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(
                f"{csv_path} 缺少欄位：{sorted(missing)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = resolve_image_path(row["image_path"])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        result = {
            "image": image,
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
        }

        if self.require_target:
            # 將 0~100 正規化成 0~1 訓練
            quality = float(row[self.target_column]) / 100.0
            result["target"] = quality

        return result
