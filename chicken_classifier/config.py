from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for the project."""

    data_directory: Path = Path("/code/data")
    dataset_version: str = "version1"
