# Phase 1 Project Plan

## Domain

Biomedical research question answering using PubMed-style abstracts.

## Model

Primary model: Qwen/Qwen2.5-7B-Instruct.

Fallback model: microsoft/Phi-3-mini-4k-instruct for low-memory smoke tests.

## Dataset

Primary MVP dataset: llamafactory/PubMedQA.

Research-grade extension: bigbio/pubmed_qa.

## MVP Scope

- Dataset preparation
- Base model evaluation
- QLoRA fine-tuning
- Fine-tuned model evaluation
- MLflow experiment tracking
- Streamlit deployment
- GGUF conversion notes and benchmark

## Phase Commands

| Phase | Command |
|---|---|
| Phase 1: Project structure | Already scaffolded in this repository |
| Phase 2: Environment setup | `python scripts/00_verify_gpu.py` |
| Phase 3: Dataset preprocessing | `python scripts/01_prepare_dataset.py` |
| Phase 4: Baseline evaluation | `python scripts/02_baseline_eval.py` |
| Phase 5: QLoRA training | `python scripts/03_train_qlora.py` |
| Phase 6: Benchmark comparison | `python scripts/04_evaluate_adapter.py` then `python scripts/06_compare_results.py` |
| Phase 7: Streamlit deployment | `streamlit run app/streamlit_app.py` |
| Demo examples | `python scripts/07_create_demo_examples.py` |

## Timeline

| Week | Goal |
|---|---|
| 1 | Environment, repo setup, dataset preparation |
| 2 | Baseline inference and evaluation |
| 3 | QLoRA fine-tuning |
| 4 | Benchmarking and ablation study |
| 5 | Deployment and optimization |
| 6 | Documentation, report, final polish |

## Training Profiles

The default config is a fast local MVP profile:

- 1,000 training rows
- 100 validation rows
- 250 optimizer steps
- sequence length 768
- LoRA rank 16

For final report-quality training, increase:

- `configs/model_config.yaml`: `max_seq_length: 1024`
- `configs/train_config.yaml`: `train_sample_limit: 4000` or remove the limit
- `configs/train_config.yaml`: `max_steps: 500` to `1000`
