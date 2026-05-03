"""Optional FastAPI inference service."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data.formatting import build_prompt
from src.inference.generate import generate_response
from src.inference.model_loader import load_causal_lm
from src.utils.config import load_yaml

app = FastAPI(title="PubMedQA QLoRA Assistant API")

MODEL = None
TOKENIZER = None
MODEL_CONFIG = None


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = ""
    adapter_path: str | None = "models/adapters/qwen2_5_1_5b_pubmedqa_qlora"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    answer: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    peak_memory_mb: float


def get_model(adapter_path: str | None):
    global MODEL, TOKENIZER, MODEL_CONFIG
    if MODEL is None:
        MODEL_CONFIG = load_yaml("configs/model_config.yaml")
        MODEL, TOKENIZER = load_causal_lm(MODEL_CONFIG, adapter_path=adapter_path)
    return MODEL, TOKENIZER


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    model, tokenizer = get_model(request.adapter_path)
    prompt = build_prompt(request.question, request.context)
    result = generate_response(
        model,
        tokenizer,
        prompt,
        {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
        },
    )
    return GenerateResponse(
        answer=result.text,
        latency_seconds=result.latency_seconds,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        peak_memory_mb=result.peak_memory_mb,
    )
