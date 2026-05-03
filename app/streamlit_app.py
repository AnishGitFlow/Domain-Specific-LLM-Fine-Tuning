"""Streamlit demo application for the fine-tuned biomedical QA model."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.data.formatting import build_prompt
from src.inference.generate import generate_response
from src.inference.model_loader import load_causal_lm
from src.utils.config import load_yaml


@st.cache_resource(show_spinner="Loading model. This can take a minute on first run.")
def get_model(adapter_path: str | None):
    model_config = load_yaml("configs/model_config.yaml")
    model, tokenizer = load_causal_lm(model_config, adapter_path=adapter_path or None)
    return model, tokenizer, model_config


def main() -> None:
    st.set_page_config(page_title="PubMedQA QLoRA Assistant", page_icon=":material/biotech:")
    st.title("PubMedQA QLoRA Assistant")
    st.caption("Biomedical literature QA demo using a QLoRA fine-tuned open-source LLM.")

    deploy_config = load_yaml("configs/deploy_config.yaml")
    default_adapter = "models/adapters/qwen2_5_1_5b_pubmedqa_qlora"

    with st.sidebar:
        st.header("Model")
        use_adapter = st.checkbox("Use fine-tuned LoRA adapter", value=True)
        adapter_path = st.text_input("Adapter path", value=default_adapter)
        max_new_tokens = st.slider(
            "Max new tokens",
            min_value=64,
            max_value=512,
            value=int(deploy_config["max_new_tokens"]),
            step=32,
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(deploy_config["temperature"]),
            step=0.05,
        )

    question = st.text_area(
        "Question",
        value="Answer the biomedical research question based on the context.",
        height=90,
    )
    context = st.text_area(
        "PubMed abstract or evidence context",
        value="Paste a PubMed abstract or biomedical evidence passage here.",
        height=220,
    )

    if st.button("Generate answer", type="primary"):
        if not context.strip():
            st.warning("Please paste a PubMed abstract or evidence context before generating an answer.")
            st.stop()

        selected_adapter = adapter_path if use_adapter else None
        model, tokenizer, _ = get_model(selected_adapter)
        prompt = build_prompt(question, context)
        result = generate_response(
            model,
            tokenizer,
            prompt,
            {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": float(deploy_config["top_p"]),
                "do_sample": temperature > 0,
            },
        )

        st.subheader("Answer")
        st.write(result.text)
        st.subheader("Runtime")
        st.json(
            {
                "latency_seconds": round(result.latency_seconds, 3),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "peak_memory_mb": round(result.peak_memory_mb, 2),
            }
        )


if __name__ == "__main__":
    main()
