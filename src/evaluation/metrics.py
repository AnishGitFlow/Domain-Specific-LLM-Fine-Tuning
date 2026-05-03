"""Evaluation metrics for generated answers."""

from __future__ import annotations

import math
from typing import Iterable

import evaluate
import numpy as np
import torch


def compute_text_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE and BLEU scores."""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

    return {
        "rouge1": float(rouge_scores["rouge1"]),
        "rouge2": float(rouge_scores["rouge2"]),
        "rougeL": float(rouge_scores["rougeL"]),
        "bleu": float(bleu_score["score"]),
    }


def compute_latency_metrics(latencies: Iterable[float]) -> dict[str, float]:
    """Summarize latency measurements."""
    values = np.array(list(latencies), dtype=float)
    return {
        "latency_mean_seconds": float(values.mean()),
        "latency_p50_seconds": float(np.percentile(values, 50)),
        "latency_p95_seconds": float(np.percentile(values, 95)),
    }


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 1024) -> float:
    """Compute simple average perplexity over reference texts."""
    losses = []
    for text in texts:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.inference_mode():
            outputs = model(**encodings, labels=encodings["input_ids"])
        losses.append(float(outputs.loss.detach().cpu()))

    return float(math.exp(np.mean(losses))) if losses else float("nan")
