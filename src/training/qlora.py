"""QLoRA fine-tuning pipeline."""

from __future__ import annotations

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training

# Windows may default to cp1252, while TRL ships UTF-8 Jinja templates.
# Patch Path.read_text before importing TRL so its lazy imports decode cleanly.
from pathlib import Path


_ORIGINAL_READ_TEXT = Path.read_text


def _read_text_utf8(self, encoding=None, errors=None):
    return _ORIGINAL_READ_TEXT(self, encoding=encoding or "utf-8", errors=errors)


Path.read_text = _read_text_utf8

from trl import SFTConfig, SFTTrainer

from src.inference.model_loader import load_causal_lm
from src.utils.config import ensure_dir


def build_lora_config(train_config: dict) -> LoraConfig:
    """Create PEFT LoRA config."""
    lora = train_config["lora"]
    return LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
    )


def train_qlora(model_config: dict, train_config: dict) -> str:
    """Fine-tune a quantized model with QLoRA."""
    train_dataset = load_dataset("json", data_files="data/processed/train.jsonl", split="train")
    eval_dataset = load_dataset("json", data_files="data/processed/validation.jsonl", split="train")

    if "completion" not in train_dataset.column_names:
        train_dataset = train_dataset.map(lambda row: {"completion": row["output"]})
    if "completion" not in eval_dataset.column_names:
        eval_dataset = eval_dataset.map(lambda row: {"completion": row["output"]})

    train_sample_limit = train_config.get("train_sample_limit")
    eval_sample_limit = train_config.get("eval_sample_limit")
    if train_sample_limit:
        train_dataset = train_dataset.select(range(min(int(train_sample_limit), len(train_dataset))))
    if eval_sample_limit:
        eval_dataset = eval_dataset.select(range(min(int(eval_sample_limit), len(eval_dataset))))

    model, tokenizer = load_causal_lm(model_config)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    output_dir = ensure_dir(train_config["output_dir"])
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_config["num_train_epochs"]),
        max_steps=int(train_config.get("max_steps", -1)),
        per_device_train_batch_size=int(train_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        warmup_ratio=float(train_config["warmup_ratio"]),
        lr_scheduler_type=train_config["lr_scheduler_type"],
        logging_steps=int(train_config["logging_steps"]),
        eval_steps=int(train_config["eval_steps"]),
        save_steps=int(train_config["save_steps"]),
        save_total_limit=int(train_config["save_total_limit"]),
        gradient_checkpointing=bool(train_config["gradient_checkpointing"]),
        optim=train_config["optim"],
        bf16=bool(train_config["bf16"]) and torch.cuda.is_available(),
        fp16=bool(train_config["fp16"]) and torch.cuda.is_available(),
        seed=int(train_config["seed"]),
        report_to=[],
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        dataset_text_field="text",
        max_length=int(model_config["max_seq_length"]),
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(train_config),
        processing_class=tokenizer,
    )

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("pubmedqa-qlora")
    with mlflow.start_run(run_name="qwen2.5-7b-pubmedqa-qlora"):
        mlflow.log_params(
            {
                "base_model": model_config["base_model"],
                "max_seq_length": model_config["max_seq_length"],
                "lora_r": train_config["lora"]["r"],
                "lora_alpha": train_config["lora"]["alpha"],
                "learning_rate": train_config["learning_rate"],
                "gradient_accumulation_steps": train_config["gradient_accumulation_steps"],
            }
        )
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)
