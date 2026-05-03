"""Model and tokenizer loading helpers."""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def dtype_from_name(name: str) -> torch.dtype:
    """Convert config dtype names to torch dtypes."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(name).lower(), torch.bfloat16)


def build_quantization_config(config: dict) -> BitsAndBytesConfig | None:
    """Build a bitsandbytes quantization config from model settings."""
    if not config.get("load_in_4bit", True):
        return None

    compute_dtype = dtype_from_name(config.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
    )


def load_tokenizer(model_name: str, trust_remote_code: bool = False):
    """Load tokenizer and guarantee a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(config: dict, adapter_path: str | None = None):
    """Load a quantized causal LM and optional LoRA adapter."""
    model_name = config["base_model"]
    trust_remote_code = bool(config.get("trust_remote_code", False))
    quantization_config = build_quantization_config(config)

    tokenizer = load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=dtype_from_name(config.get("torch_dtype", "bfloat16")),
        trust_remote_code=trust_remote_code,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer
