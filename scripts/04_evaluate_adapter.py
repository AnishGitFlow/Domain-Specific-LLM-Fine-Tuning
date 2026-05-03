"""Evaluate the fine-tuned LoRA adapter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.runner import run_generation_benchmark
from src.utils.config import load_yaml

def main() -> None:
    model_config = load_yaml("configs/model_config.yaml")
    train_config = load_yaml("configs/train_config.yaml")
    eval_config = load_yaml("configs/eval_config.yaml")

    run_generation_benchmark(
        model_config=model_config,
        eval_config=eval_config,
        dataset_path="data/processed/test.jsonl",
        output_prefix="finetuned",
        adapter_path=train_config["output_dir"],
    )


if __name__ == "__main__":
    main()
