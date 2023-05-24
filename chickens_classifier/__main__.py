import fire

from .train import train_model


def main() -> None:
    """Call CLI commands."""
    fire.Fire({"train-model": train_model})
