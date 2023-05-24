"""Configuration for the project."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Configuration for the project."""

    data_directory: Path = Path("/code/data")
    dataset_version: str = "version1"

    batch_size: int = 8
    num_workers: int = 4
    max_epochs: int = 5
    labels: tuple[str, str] = ("alive", "dead")
