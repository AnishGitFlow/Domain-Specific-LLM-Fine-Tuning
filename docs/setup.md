# Environment Setup

This project targets Windows or Linux with an NVIDIA RTX 5070, 12 GB VRAM, and 32 GB RAM.

## 1. Create Environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

On Windows PowerShell, enable UTF-8 mode for the current terminal session:

```powershell
$env:PYTHONUTF8="1"
```

On Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install PyTorch for RTX 5070

RTX 50-series GPUs require CUDA 12.8+ compatible PyTorch wheels.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 3. Install Project Dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Verify GPU

```bash
python scripts/00_verify_gpu.py
```

You should see:

```text
CUDA available: True
GPU name: NVIDIA GeForce RTX 5070
PyTorch CUDA: 12.8
```

## 5. Hugging Face Login

Qwen2.5 does not require a paid API. Login is still useful for model caching and uploads.

```bash
huggingface-cli login
```

Create `.env` from `.env.example` if needed:

```bash
copy .env.example .env
```

## 6. Run Pipeline

```bash
python scripts/01_prepare_dataset.py
python scripts/02_baseline_eval.py
python scripts/03_train_qlora.py
python scripts/04_evaluate_adapter.py
python scripts/06_compare_results.py
```

## 7. Run Demo

```bash
streamlit run app/streamlit_app.py
```

## Troubleshooting

If you see `no kernel image is available for execution on the device`, reinstall PyTorch using the CUDA 12.8 index above.

If TRL fails with a `UnicodeDecodeError` while reading a `.jinja` template on Windows, run the command with UTF-8 mode:

```powershell
$env:PYTHONUTF8="1"
python scripts/03_train_qlora.py
```

If you hit CUDA out-of-memory, reduce these values in `configs/model_config.yaml` and `configs/train_config.yaml`:

- `max_seq_length: 768`
- `gradient_accumulation_steps: 16`
- keep `per_device_train_batch_size: 1`
