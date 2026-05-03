"""Text generation helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    peak_memory_mb: float


def generate_response(model, tokenizer, prompt: str, generation_config: dict) -> GenerationResult:
    """Generate one response and collect latency/memory metadata."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(generation_config.get("max_new_tokens", 256)),
            temperature=float(generation_config.get("temperature", 0.2)),
            top_p=float(generation_config.get("top_p", 0.9)),
            do_sample=bool(generation_config.get("do_sample", True)),
            pad_token_id=tokenizer.eos_token_id,
        )

    latency = time.perf_counter() - start
    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    return GenerationResult(
        text=text,
        latency_seconds=latency,
        input_tokens=int(inputs["input_ids"].shape[-1]),
        output_tokens=int(generated_ids.shape[-1]),
        peak_memory_mb=peak_memory_mb,
    )
