"""Prepare PubMedQA data for instruction fine-tuning."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline import prepare_dataset, save_dataset
from src.utils.config import load_yaml
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

def main() -> None:
    config = load_yaml("configs/data_config.yaml")
    dataset = prepare_dataset(config)
    paths = save_dataset(dataset)

    LOGGER.info("Prepared dataset")
    LOGGER.info("Train rows: %s", len(dataset["train"]))
    LOGGER.info("Validation rows: %s", len(dataset["validation"]))
    LOGGER.info("Test rows: %s", len(dataset["test"]))
    LOGGER.info("Saved train split to %s", paths.train)
    LOGGER.info("Saved validation split to %s", paths.validation)
    LOGGER.info("Saved test split to %s", paths.test)


if __name__ == "__main__":
    main()
