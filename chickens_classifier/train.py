"""Train the chickens classifier."""
import torch
from owlml import generate_mlflow_url
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)

from .config import Config
from .dataset import Dataset, Holdouts
from .model import Model


class Lightning(LightningModule):
    """Lightning module for the chickens classifier."""

    def __init__(self):
        """Initialize the lightning module."""
        super().__init__()
        self.model = Model()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Return the train dataloader."""
        return torch.utils.data.DataLoader(
            Dataset(Holdouts.train),
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Return the validation dataloader."""
        return torch.utils.data.DataLoader(
            Dataset(Holdouts.validation),
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Return the test dataloader."""
        return torch.utils.data.DataLoader(
            Dataset(Holdouts.test),
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
        )

    def training_step(
        self, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> STEP_OUTPUT:
        """Execute the training step."""
        logits = self.model(batch["images"])
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], *args, **kwargs) -> None:
        """Execute the validation step."""
        scores = self.model(batch["images"])
        accuracy = (scores.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the optimizer."""
        return torch.optim.Adam(self.model.parameters())


def train_model() -> None:
    """Train the model."""
    logger = MLFlowLogger(tracking_uri=generate_mlflow_url())
    model = Lightning()
    trainer = Trainer(log_every_n_steps=1, max_epochs=Config.max_epochs, logger=logger)
    trainer.fit(model)
