# Technical Viability Analysis: Running Step-Audio-EditX on Apple Silicon (Mac Mini M4)

## Executive Summary & Verdict

| Dimension | Assessment |
| :--- | :--- |
| **Viability** | **100% Possible.** All components of Step-Audio-EditX (FunASR, Whisper, ONNX Tokenizer, 3B LLM, CosyVoice Vocoder) can run on Apple Silicon hardware (Metal / MPS / CoreML / ANE / MLX). |
| **Level of Effort (LOE)** | **MODERATE.** Estimated **1 to 2 developer days** for a PyTorch MPS implementation, or **3 to 4 days** for an optimized Apple MLX implementation. |
| **Mac Mini M4 Hardware Suitability** | **Excellent.** A standard Mac Mini M4 (16GB, 24GB, or 32GB Unified Memory) can easily hold the entire pipeline (~3.4 GB to 7.4 GB RAM required) in Unified Memory and execute it with low latency. |

---

## 1. Project Architecture & Component Breakdown

The `Step-Audio-EditX` repository relies on a 3-stage inference pipeline:

```
[ Input Audio ] ──> 1. StepAudioTokenizer (FunASR + Whisper + ONNX)
                          │
                          ▼
                    2. Step-Audio-EditX (3B LLM)
                          │
                          ▼
                    3. CosyVoice-300M Vocoder (Flow Matching + BigVGAN) ──> [ Output Wav ]
```

### Component Compatibility Matrix

| Pipeline Component | Key Files | Original Implementation | Apple Silicon Target Hardware | Effort Level |
| :--- | :--- | :--- | :--- | :--- |
| **1. Audio Tokenizer** | `tokenizer.py`<br>`funasr_detach/` | FunASR Paraformer (PyTorch) + Whisper Mel Spectrogram + ONNX (`speech_tokenizer_v1.onnx` with CUDA provider) | **ONNX CoreML / CPU Provider** + PyTorch MPS | **Low** (1-line code change) |
| **2. 3B Audio LLM** | `model_loader.py`<br>`tts.py`<br>`src/model/step_audio.py` | **vLLM** (`vllm.LLM`), hardcoded `VLLM_ATTENTION_BACKEND = "TRITON_ATTN"`, `onnxruntime-gpu`, `deepspeed`, `bitsandbytes` | **PyTorch MPS (`device="mps"`)** or **Apple MLX (`mlx-lm`)** | **Moderate** (Replace vLLM engine with PyTorch/MLX generator) |
| **3. Vocoder (CosyVoice 300M)** | `stepvocoder/cosyvoice2/cli/cosyvoice.py`<br>`stepvocoder/cosyvoice2/bigvgan/bigvgan.py` | Hardcoded `device="cuda"`, CUDA Graphs enabled by default | **PyTorch MPS (`device="mps"`)** (Snake activations fall back to pure PyTorch) | **Low** (Update device selection & disable CUDA graphs) |

---

## 2. Detailed Breakdown of Blockers & Technical Fixes

### A. Primary Blocker: `vLLM` Dependency
* **The Problem:** The repo uses `vLLM` in `model_loader.py` as its primary LLM engine. `vLLM` relies on CUDA C++ extensions, Triton kernels, and Linux shared memory which **do not run on macOS**.
* **The Solution:** 
  * `Step-Audio-EditX` is fundamentally a standard 3B Causal Language Model (`Step1Model` in `src/model/step_audio.py`).
  * Replace the `vLLM` call in `model_loader.py` with standard HuggingFace `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16).to("mps")` or convert the weights to Apple's **MLX** (`mlx-lm`) framework.

### B. ONNX Execution Provider
* **The Problem:** `tokenizer.py` explicitly passes `providers = ["CUDAExecutionProvider"]`.
* **The Solution:** Update the ONNX session options to use Apple Silicon hardware acceleration:
  ```python
  providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
  ```

### C. Hardcoded CUDA Device Strings & CUDA Graphs
* **The Problem:**
  * `app.py` sets `os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"`.
  * `stepvocoder/cosyvoice2/cli/cosyvoice.py` sets `self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
  * `tts.py` defaults `cosyvoice_cuda_graph=True`.
* **The Solution:**
  * Remove `TRITON_ATTN` env var.
  * Update device detection to check `torch.backends.mps.is_available()`.
  * Pass `--no-cosyvoice-cuda-graph` or default `cosyvoice_cuda_graph=False` on macOS.

### D. Custom CUDA Activation Kernels in BigVGAN
* **The Problem:** BigVGAN in `stepvocoder/cosyvoice2/bigvgan/alias_free_activation/cuda/load.py` has optional CUDA C++ anti-aliased Snake activations (`anti_alias_activation_cuda.cu`).
* **The Solution:** BigVGAN already includes a pure PyTorch fallback (`stepvocoder/cosyvoice2/bigvgan/alias_free_activation/torch/act.py`) which runs out-of-the-box on PyTorch MPS without compiling any custom C++ kernels.

---

## 3. Hardware Resource Estimation for Mac Mini M4

* **Unified Memory Footprint**:
  * **Step-Audio-Tokenizer (FunASR + Whisper)**: ~0.8 GB
  * **Step-Audio-EditX 3B LLM** (bfloat16 / float16): ~6.0 GB *(or ~2.0 GB if quantized to 4-bit)*
  * **CosyVoice 300M Vocoder**: ~0.6 GB
  * **Total Memory Footprint**: **~7.4 GB** (FP16/BF16) or **~3.4 GB** (INT4/MLX).
* **Mac Mini M4 Performance**:
  * Baseline M4 (16GB, 24GB, or 32GB RAM) easily handles the full pipeline in Unified Memory.
  * M4 Unified Memory bandwidth (120 GB/s to 273 GB/s) enables near real-time or faster-than-real-time audio generation.
