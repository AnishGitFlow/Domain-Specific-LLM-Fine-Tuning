"""Fine-tune the base model with QLoRA."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.qlora import train_qlora
from src.utils.config import load_yaml
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

def main() -> None:
    model_config = load_yaml("configs/model_config.yaml")
    train_config = load_yaml("configs/train_config.yaml")
    adapter_path = train_qlora(model_config, train_config)
    LOGGER.info("Saved LoRA adapter to %s", adapter_path)


if __name__ == "__main__":
    main()
