"""Model for the chickens classifier."""
import torch
from torchvision.models import ResNet18_Weights, resnet18

from .config import Config


class Model(torch.nn.Module):
    """Classifier model."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        default_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(
            *list(default_model.children())[:-1],
            torch.nn.Flatten(),
            torch.nn.Linear(default_model.fc.in_features, len(Config.labels)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        logits = self.model(images)
        if self.training:
            return logits
        else:
            return torch.nn.functional.softmax(logits, dim=-1)
