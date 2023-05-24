"""Dataset module."""
from typing import Callable, Optional

import torch
from owlml import generate_records
from PIL import Image
from torchvision.models import ResNet18_Weights

from .config import Config


class Holdouts:
    """Holdouts class for holdout evaluators used to split the dataset."""

    train: Callable[[int], bool] = lambda x: x % 100 < 80
    validation: Callable[[int], bool] = lambda x: 80 <= x % 100 < 90
    test: Callable[[int], bool] = lambda x: 90 <= x % 100


class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, holdout: Optional[Callable[[int], bool]] = None):
        """Initialize the dataset."""
        self.records = generate_records(
            Config.data_directory, Config.dataset_version, holdout_evaluator=holdout
        )
        self.transforms = ResNet18_Weights.DEFAULT.transforms()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the item at the given index."""
        record = self.records[index]
        image = Image.open(record["image_path"])
        assert len(record["label_ids"]) == 1
        label_id = record["label_ids"][0]
        return dict(images=self.transforms(image), labels=torch.tensor(label_id))
