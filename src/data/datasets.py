from collections.abc import Callable
from pathlib import Path
from typing import Self

import pandas as pd
import torch
from pandas import read_csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src import config


class DocumentImageSet(Dataset):
    COLUMN_FILENAME = "ID"
    COLUMN_CLASS = "target"
    COLUMN_CLASS_NAME = "class_name"

    def __init__(
        self, meta_csv_filepath: Path, img_directory: Path, class_meta_csv_filepath: Path, img_transform: Callable
    ):
        super().__init__()

        self.meta_csv_filepath = meta_csv_filepath
        self.img_directory = img_directory
        self.img_transform = img_transform
        self.class_meta_csv_filepath = class_meta_csv_filepath

        self.meta_df = pd.read_csv(self.meta_csv_filepath)
        self.class_meta_df = pd.read_csv(self.class_meta_csv_filepath).set_index(self.COLUMN_CLASS)

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        meta_row = self.meta_df.iloc[idx]
        filename: str = str(meta_row[self.COLUMN_FILENAME])
        target: int = int(meta_row[self.COLUMN_CLASS])

        img_path = self.img_directory / filename
        img = Image.open(img_path).convert("RGB")
        item_image: torch.Tensor = self.img_transform(img)
        return item_image, target

    @property
    def num_classes(self) -> int:
        return self.meta_df[self.COLUMN_CLASS].nunique()

    @property
    def class_names(self) -> dict:
        return self.class_meta_df[self.COLUMN_CLASS_NAME].to_dict()

    @classmethod
    def create_train_dataset(cls, img_transform: Callable) -> Self:
        from src import config

        return cls(
            meta_csv_filepath=config.TRAIN_META_CSV_PATH,
            img_directory=config.TRAIN_IMG_DIR,
            class_meta_csv_filepath=config.CLASS_META_CSV_PATH,
            img_transform=img_transform if img_transform else transforms.ToTensor(),
        )

    @classmethod
    def create_augmented_train_dataset(cls, img_transform: Callable) -> Self:
        return cls(
            meta_csv_filepath=config.RAW_DATA_DIR / "augment.csv",
            img_directory=config.RAW_DATA_DIR / "augment",
            class_meta_csv_filepath=config.CLASS_META_CSV_PATH,
            img_transform=img_transform if img_transform else transforms.ToTensor(),
        )

    @classmethod
    def create_test_dataset(cls, img_transform: Callable) -> Self:
        return cls(
            meta_csv_filepath=config.TEST_META_CSV_PATH,
            img_directory=config.TEST_IMG_DIR,
            class_meta_csv_filepath=config.CLASS_META_CSV_PATH,
            img_transform=img_transform if img_transform else transforms.ToTensor(),
        )

    @classmethod
    def calculate_metadata_classes(cls) -> int:
        meta_df = read_csv(config.CLASS_META_CSV_PATH)
        return meta_df[cls.COLUMN_CLASS].nunique()
