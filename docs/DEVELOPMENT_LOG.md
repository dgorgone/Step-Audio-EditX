# Development Log - Step-Audio-EditX Apple Silicon Port

## Technical State & Architecture Overview
- **Repository Fork**: [https://github.com/dgorgone/Step-Audio-EditX.git](https://github.com/dgorgone/Step-Audio-EditX.git)
- **Target Platform**: macOS (Apple Silicon Mac Mini M4)
- **Acceleration Backend**: PyTorch MPS (`device="mps"`), ONNX CoreML/CPU Provider
- **Virtual Environment**: `uv` with Python 3.12 (`.venv`)

---

## Progress Log

### Session: July 28, 2026
- **Technical Viability Analysis**: Saved full report to `docs/MACOS_M4_VIABILITY_ANALYSIS.md`.
- **Git Remotes**: Updated `origin` to `https://github.com/dgorgone/Step-Audio-EditX.git` and `upstream` to `https://github.com/stepfun-ai/Step-Audio-EditX.git`. Commited and pushed initial documentation to `origin/main`.
- **Clean Environment Setup**: Configured `pyproject.toml` and built a clean `.venv` using `uv`. Verified PyTorch 2.13.0 MPS backend (`MPS Available: True`).
- **ONNX Hardware Acceleration**: Updated `tokenizer.py` and `stepvocoder/cosyvoice2/cli/frontend.py` to support `CoreMLExecutionProvider` and `CPUExecutionProvider` on macOS.
- **CosyVoice Vocoder MPS Device**: Updated `stepvocoder/cosyvoice2/cli/cosyvoice.py` to prioritize `torch.device("mps")` when available. Disabled CUDA graphs on non-CUDA platforms.
- **PyTorch MPS LLM Engine**: Added `PyTorchCausalLMEngine` fallback in `model_loader.py` using `AutoModelForCausalLM` on `device="mps"` with `torch.bfloat16`. Made `vllm` imports optional.
- **Environment Variable Guards**: Guarded `VLLM_ATTENTION_BACKEND` in `app.py` and `tts_infer.py` to prevent environment conflicts on macOS.
