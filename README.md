# PubMedQA QLoRA Assistant

Production-style pipeline for adapting an open-source instruction-tuned language model to biomedical question answering with QLoRA. The project covers dataset preparation, parameter-efficient fine-tuning, benchmark evaluation, experiment tracking, model export notes, and lightweight deployment through Streamlit or FastAPI.

> This project is intended for biomedical literature question answering and experimentation. It is not a medical diagnosis or treatment system.

## Highlights

- Fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` on instruction-formatted PubMedQA data.
- Uses 4-bit NF4 quantization, PEFT LoRA adapters, TRL `SFTTrainer`, and bitsandbytes for consumer-GPU training.
- Benchmarks base and fine-tuned models with ROUGE, BLEU, perplexity, latency, and GPU memory metrics.
- Tracks training runs locally with MLflow under `mlruns/`.
- Ships reusable source modules plus script entry points for each pipeline stage.
- Includes Streamlit and FastAPI inference surfaces for local demos.
- Provides Docker and GGUF export guidance for deployment and inference optimization.

## Repository Structure

```text
app/                  Streamlit and FastAPI demo applications
configs/              YAML configs for data, model, training, evaluation, deployment
data/                 Raw, interim, processed, and sample datasets
docs/                 Architecture, setup, deployment, reports, and demo examples
mlruns/               Local MLflow experiment tracking data
models/               LoRA adapters, merged-model placeholders, and GGUF placeholders
notebooks/            Notebook workspace for exploration and analysis
results/              Metrics, prediction reports, benchmark tables, and plots
scripts/              Pipeline entry points from setup through evaluation and reporting
src/                  Reusable Python package for data, training, evaluation, inference
tests/                Test workspace
```

## System Overview

The pipeline prepares PubMedQA instruction data, evaluates the base model, trains a QLoRA adapter, evaluates the adapted model on the same held-out test split, and generates comparison artifacts.

```text
PubMedQA -> preprocessing -> JSONL splits -> base evaluation
                                      |
                                      -> QLoRA fine-tuning -> LoRA adapter
                                                               |
                                                               -> adapter evaluation
                                                               -> Streamlit / FastAPI demo
```

Core implementation modules:

- `src/data/pipeline.py`: loads, normalizes, filters, deduplicates, splits, and saves PubMedQA data.
- `src/data/formatting.py`: builds the Qwen chat-style prompt and supervised fine-tuning text.
- `src/inference/model_loader.py`: loads the quantized base model and optional PEFT adapter.
- `src/inference/generate.py`: generates answers and records latency, token counts, and peak CUDA memory.
- `src/training/qlora.py`: configures LoRA and runs TRL supervised fine-tuning with MLflow logging.
- `src/evaluation/runner.py`: runs generation benchmarks and writes metrics and predictions.
- `src/evaluation/metrics.py`: computes ROUGE, BLEU, perplexity, latency summaries, and memory metrics.

## Requirements

- Python 3.10 or newer
- NVIDIA GPU recommended for training and evaluation
- CUDA-compatible PyTorch build
- Hugging Face access for downloading the base model and dataset

The included setup guide targets an NVIDIA RTX 5070 with 12 GB VRAM and CUDA 12.8 PyTorch wheels. Smaller GPUs may require reducing `max_seq_length`, batch size, or sample limits.

## Quickstart

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
$env:PYTHONUTF8="1"
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Verify GPU access:

```bash
python scripts/00_verify_gpu.py
```

Run the full pipeline:

```bash
python scripts/01_prepare_dataset.py
python scripts/02_baseline_eval.py
python scripts/03_train_qlora.py
python scripts/04_evaluate_adapter.py
python scripts/06_compare_results.py
```

You can also use the Makefile shortcuts:

```bash
make setup
make verify
make prepare
make baseline
make train
make eval
make compare
```

## Configuration

The project is driven by YAML files in `configs/`.

| File | Purpose |
|---|---|
| `configs/data_config.yaml` | Hugging Face dataset name, source columns, split ratios, seed, and sample caps |
| `configs/model_config.yaml` | Base model, dtype, 4-bit quantization settings, and max sequence length |
| `configs/train_config.yaml` | Output adapter path, QLoRA hyperparameters, optimizer, steps, batch sizes, and LoRA target modules |
| `configs/eval_config.yaml` | Generation settings and benchmark sample count |
| `configs/deploy_config.yaml` | Streamlit/FastAPI defaults for host, port, tokens, temperature, and top-p |

Current default model settings:

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
max_seq_length: 1024
```

Current adapter output:

```text
models/adapters/qwen2_5_1_5b_pubmedqa_qlora
```

## Pipeline Outputs

Running the scripts produces these primary artifacts:

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
data/samples/sample.jsonl
models/adapters/qwen2_5_1_5b_pubmedqa_qlora/
results/metrics/baseline_metrics.json
results/metrics/finetuned_metrics.json
results/reports/baseline_predictions.jsonl
results/reports/finetuned_predictions.jsonl
results/reports/benchmark_comparison.csv
results/plots/benchmark_comparison.png
```

MLflow metadata is written to `mlruns/` with the experiment name `pubmedqa-qlora`.

## Benchmark Results

The included benchmark compares the base model and fine-tuned LoRA adapter on 100 held-out PubMedQA test examples.

| Metric | Base Model | Fine-Tuned Model | Change |
|---|---:|---:|---:|
| BLEU | 3.266 | 4.837 | +1.571 |
| ROUGE-1 | 0.273 | 0.302 | +0.029 |
| ROUGE-2 | 0.072 | 0.108 | +0.036 |
| ROUGE-L | 0.167 | 0.216 | +0.049 |
| Perplexity | 10.342 | 9.340 | -1.003 |
| Mean latency | 4.206s | 2.155s | -2.052s |
| p50 latency | 4.146s | 2.120s | -2.026s |
| p95 latency | 7.147s | 3.250s | -3.897s |
| Peak memory max | 1236.874 MB | 1307.312 MB | +70.438 MB |

The fine-tuned adapter improves lexical overlap metrics and reduces perplexity, indicating better alignment with biomedical reference answers. The adapter adds a small amount of peak memory usage, which is expected when loading LoRA weights on top of the quantized base model.

## Running the Demo

Start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

The app lets you:

- Toggle the fine-tuned LoRA adapter on or off.
- Edit the adapter path.
- Set max new tokens and temperature.
- Submit a biomedical question and PubMed abstract or evidence context.
- View generated answer, latency, token counts, and peak CUDA memory.

The UI requires a non-empty context before generation to reduce unsupported biomedical hallucination.

## FastAPI Service

Start the API:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Generate an answer:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the study conclude?",
    "context": "Paste a PubMed abstract or biomedical evidence passage here.",
    "adapter_path": "models/adapters/qwen2_5_1_5b_pubmedqa_qlora",
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.9
  }'
```

Response shape:

```json
{
  "answer": "...",
  "latency_seconds": 2.15,
  "input_tokens": 512,
  "output_tokens": 96,
  "peak_memory_mb": 1307.31
}
```

## Docker

Build the image:

```bash
docker build -t pubmedqa-qlora .
```

Run the Streamlit demo with GPU access:

```bash
docker run --gpus all -p 7860:7860 pubmedqa-qlora
```

Then open:

```text
http://localhost:7860
```

## GGUF Export

The repository includes `scripts/05_export_gguf.py`, which writes a GGUF conversion guide to `docs/gguf_export.md`. The intended workflow is:

1. Merge the LoRA adapter into the base Hugging Face model with PEFT.
2. Convert the merged model with `llama.cpp`.
3. Quantize to a local inference format such as `Q4_K_M`.
4. Run with `llama-cli` or another GGUF-compatible runtime.

Generate the guide:

```bash
python scripts/05_export_gguf.py
```

## Development Notes

- Use `PYTHONUTF8=1` on Windows if TRL or Jinja template loading raises encoding errors.
- Install PyTorch with the CUDA index appropriate for your GPU and driver.
- If training runs out of memory, reduce `per_device_train_batch_size`, `max_seq_length`, or LoRA rank in the config files.
- The repository currently contains processed data, model adapter artifacts, and MLflow outputs for reproducibility and demonstration.
- `tests/` is present as a workspace but does not currently contain executable tests.

## Additional Documentation

- `docs/setup.md`: detailed environment setup and troubleshooting.
- `docs/architecture.md`: Mermaid architecture diagrams for the pipeline and deployment path.
- `docs/deployment.md`: Streamlit, FastAPI, Docker, and hosted demo notes.
- `docs/results_summary.md`: benchmark interpretation and report-ready conclusions.
- `docs/demo_examples.md`: sample inputs for manual demo testing.
- `docs/final_technical_report.md`: broader project report.

## License

No license file is currently included. Add an explicit license before distributing, publishing, or reusing the project outside its current context.
