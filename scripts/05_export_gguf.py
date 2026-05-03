"""Export or document GGUF conversion for llama.cpp."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import ensure_dir


GUIDE = """# GGUF Export Guide

The recommended workflow is:

1. Merge the LoRA adapter into the base model with PEFT.
2. Clone llama.cpp.
3. Convert the merged Hugging Face model to GGUF.
4. Quantize to Q4_K_M for local CPU/GPU inference.

Commands:

```bash
git clone https://github.com/ggerganov/llama.cpp external/llama.cpp
cd external/llama.cpp
python -m pip install -r requirements.txt
python convert_hf_to_gguf.py ../../models/merged/qwen2_5_7b_pubmedqa --outfile ../../models/gguf/qwen2_5_7b_pubmedqa-f16.gguf
./llama-quantize ../../models/gguf/qwen2_5_7b_pubmedqa-f16.gguf ../../models/gguf/qwen2_5_7b_pubmedqa-q4_k_m.gguf Q4_K_M
./llama-cli -m ../../models/gguf/qwen2_5_7b_pubmedqa-q4_k_m.gguf -p "Answer the biomedical question..."
```
"""


def main() -> None:
    output_dir = ensure_dir("docs")
    path = Path(output_dir) / "gguf_export.md"
    path.write_text(GUIDE, encoding="utf-8")
    print(f"Wrote GGUF export guide to {path}")


if __name__ == "__main__":
    main()
