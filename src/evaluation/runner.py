"""Reusable evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.evaluation.metrics import compute_latency_metrics, compute_perplexity, compute_text_metrics
from src.inference.generate import generate_response
from src.inference.model_loader import load_causal_lm
from src.utils.config import ensure_dir
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def run_generation_benchmark(
    model_config: dict,
    eval_config: dict,
    dataset_path: str | Path,
    output_prefix: str,
    adapter_path: str | None = None,
) -> dict[str, float]:
    """Run generation, text metrics, latency, memory, and perplexity evaluation."""
    model, tokenizer = load_causal_lm(model_config, adapter_path=adapter_path)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    num_samples = min(int(eval_config["benchmark"].get("num_samples", 100)), len(dataset))
    dataset = dataset.select(range(num_samples))

    predictions: list[str] = []
    references: list[str] = []
    latencies: list[float] = []
    memory_values: list[float] = []
    rows: list[dict] = []

    for row in tqdm(dataset, desc=f"Evaluating {output_prefix}"):
        result = generate_response(model, tokenizer, row["prompt"], eval_config["generation"])
        predictions.append(result.text)
        references.append(row["output"])
        latencies.append(result.latency_seconds)
        memory_values.append(result.peak_memory_mb)
        rows.append(
            {
                "question": row.get("question", row["instruction"]),
                "context": row.get("context", row["input"]),
                "reference": row["output"],
                "prediction": result.text,
                "latency_seconds": result.latency_seconds,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "peak_memory_mb": result.peak_memory_mb,
            }
        )

    metrics = compute_text_metrics(predictions, references)
    metrics.update(compute_latency_metrics(latencies))
    metrics["peak_memory_mean_mb"] = float(sum(memory_values) / len(memory_values))
    metrics["peak_memory_max_mb"] = float(max(memory_values))
    metrics["perplexity"] = compute_perplexity(
        model,
        tokenizer,
        [row["text"] for row in dataset],
        max_length=int(model_config.get("max_seq_length", 1024)),
    )
    metrics["num_samples"] = float(num_samples)

    metrics_dir = ensure_dir("results/metrics")
    reports_dir = ensure_dir("results/reports")

    metrics_path = metrics_dir / f"{output_prefix}_metrics.json"
    predictions_path = reports_dir / f"{output_prefix}_predictions.jsonl"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved predictions to %s", predictions_path)
    return metrics
