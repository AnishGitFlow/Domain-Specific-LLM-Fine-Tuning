"""Verify Python, PyTorch, CUDA, and GPU availability."""

from __future__ import annotations

import platform

import torch


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA arch list: {torch.cuda.get_arch_list()}")
        print(f"VRAM GB: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")


if __name__ == "__main__":
    main()
