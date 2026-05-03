"""Prompt formatting for instruction fine-tuning and evaluation."""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Answer using only the provided context. "
    "Be concise, evidence-based, and avoid giving clinical diagnosis or treatment advice."
)


def build_prompt(question: str, context: str | None = None) -> str:
    """Build a chat-style prompt for Qwen instruction models."""
    context = (context or "").strip()
    user_content = f"Question:\n{question.strip()}"
    if context:
        user_content = f"{user_content}\n\nContext:\n{context}"

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_training_text(question: str, context: str | None, output: str) -> str:
    """Build a full supervised fine-tuning sample."""
    return f"{build_prompt(question, context)}{output.strip()}<|im_end|>"
