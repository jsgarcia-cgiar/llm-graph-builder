## Installation Guide — llm-graph-builder with PyTorch (CPU or GPU)

This guide shows how to install `llm-graph-builder` with either CPU-only PyTorch or a GPU-enabled build.

Recommended prerequisites:
- Python 3.10+
- pip >= 22, virtual environment (venv) or conda

Useful links:
- [PyTorch Get Started](https://pytorch.org/get-started/locally) — choose the right PyTorch wheel for your OS/CUDA/ROCm.

---

### Quick start

- CPU-only (any OS):
```bash
python -m venv .venv && source .venv/bin/activate  # PowerShell: .venv\\Scripts\\Activate.ps1
pip install --upgrade pip

# Install llm-graph-builder letting deps resolve from PyPI,
# with PyTorch CPU wheels available as a fallback index
pip install llm-graph-builder --extra-index-url https://download.pytorch.org/whl/cpu
```

- GPU (NVIDIA CUDA example):
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# Pick the CUDA tag matching your driver/runtime (e.g. cu124, cu121, etc.)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Then install llm-graph-builder (from PyPI)
pip install llm-graph-builder
```

> Tip: If you see conflicting Torch wheels, uninstall any pre-existing Torch first:
```bash
pip uninstall -y torch torchvision torchaudio
```

---

### Install from source (editable)

Clone the repository and install in editable mode. You can enforce CPU-only or GPU Torch by controlling the index at install time.

- CPU-only from source:
```bash
git clone https://github.com/your-org/llm-graph-builder.git
cd llm-graph-builder
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# Single command: PyPI primary; PyTorch CPU wheels as extra index
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

- GPU (CUDA example) from source:
```bash
git clone https://github.com/your-org/llm-graph-builder.git
cd llm-graph-builder
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# Choose the CUDA tag that matches your system (e.g., cu124)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install -e .
```

> macOS (Apple Silicon): Use the default PyTorch wheels from PyPI (they include MPS acceleration). For CPU-only, you can skip Torch GPU wheels and just `pip install torch` from PyPI.

> AMD (ROCm): Follow the [PyTorch Get Started](https://pytorch.org/get-started/locally) page to choose the appropriate ROCm wheels and use the provided index URL.

---

### Verify your installation

Run a quick check to ensure Torch is installed and that CUDA is detected when expected:
```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

Expected:
- CPU-only: `cuda available: False`
- GPU: `cuda available: True` and device is `cuda`

---

### Notes

- This project does not pin a specific `torch` wheel in `pyproject.toml`. Dependencies like `sentence-transformers` may bring in `torch`. Using `--extra-index-url https://download.pytorch.org/whl/cpu` lets pip resolve most packages from PyPI while making PyTorch CPU wheels available to avoid compatibility issues.
- If you already installed the package and need to switch Torch variants, uninstall Torch-related packages and reinstall using the correct index URL.
- If you manage installs via CI or scripts, prefer passing `--extra-index-url https://download.pytorch.org/whl/cpu` (for CPU) or using the CUDA/ROCm index (for GPU) consistently to avoid mismatches.


