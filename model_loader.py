"""
Unified model loading utility supporting vLLM and PyTorch MPS/CPU engines.
Supports ModelScope, HuggingFace and local path loading.
"""
import os
import logging
import threading
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from funasr_detach import AutoModel

# Optional vLLM import
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    LLM = None
    SamplingParams = None

# Global cache for downloaded models to avoid repeated downloads
_model_download_cache = {}
_download_cache_lock = threading.Lock()


class ModelSource:
    """Model source enumeration"""
    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    AUTO = "auto"


class PyTorchCausalLMEngine:
    """Fallback engine using standard PyTorch / HuggingFace AutoModelForCausalLM for MPS/CPU"""
    def __init__(self, model, device, torch_dtype):
        self.model = model
        self.device = device
        self.torch_dtype = torch_dtype

    def generate(self, prompts, sampling_params=None, use_tqdm=False, **kwargs):
        results = []
        for prompt_dict in prompts:
            token_ids = prompt_dict["prompt_token_ids"]
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            max_tokens = getattr(sampling_params, "max_tokens", 4096) if sampling_params else 4096
            temperature = getattr(sampling_params, "temperature", 0.7) if sampling_params else 0.7

            with torch.no_grad():
                gen_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": max_tokens,
                    "do_sample": (temperature > 0),
                    "eos_token_id": 3,  # <|EOT|>
                }
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature

                outputs = self.model.generate(**gen_kwargs)

            gen_ids = outputs[0][len(token_ids):].cpu().tolist()

            class FakeOutput:
                def __init__(self, token_ids):
                    self.token_ids = token_ids

            class FakeRequestOutput:
                def __init__(self, token_ids):
                    self.outputs = [FakeOutput(token_ids)]

            results.append(FakeRequestOutput(gen_ids))
        return results


class UnifiedModelLoader:
    """Unified model loader supporting vLLM and PyTorch MPS/CPU"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _cached_snapshot_download(self, model_path: str, source: str, **kwargs) -> str:
        """Cached version of snapshot_download to avoid repeated downloads"""
        cache_key = (model_path, source, str(sorted(kwargs.items())))

        with _download_cache_lock:
            if cache_key in _model_download_cache:
                cached_path = _model_download_cache[cache_key]
                self.logger.info(f"Using cached download for {model_path} from {source}: {cached_path}")
                return cached_path

        if source == ModelSource.MODELSCOPE:
            from modelscope.hub.snapshot_download import snapshot_download
            local_path = snapshot_download(model_path, **kwargs)
        elif source == ModelSource.HUGGINGFACE:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported source for cached download: {source}")

        with _download_cache_lock:
            _model_download_cache[cache_key] = local_path

        self.logger.info(f"Downloaded and cached {model_path} from {source}: {local_path}")
        return local_path

    def detect_model_source(self, model_path: str) -> str:
        """Automatically detect model source"""
        if os.path.exists(model_path) or os.path.isabs(model_path):
            return ModelSource.LOCAL

        if "/" in model_path and not model_path.startswith("http"):
            if "modelscope" in model_path.lower():
                return ModelSource.MODELSCOPE
            else:
                return ModelSource.HUGGINGFACE

        return ModelSource.LOCAL

    def load_model(
        self,
        model_path: str,
        source: str = ModelSource.AUTO,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        max_model_len: Optional[int] = None,
        enforce_eager: bool = False,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        kv_cache_dtype: Optional[str] = None,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        **kwargs
    ) -> tuple:
        """
        Load model using vLLM if available, or PyTorch MPS/CPU as fallback.
        """
        if source == ModelSource.AUTO:
            source = self.detect_model_source(model_path)

        self.logger.info(f"🚀 Loading model from {source}: {model_path}")
        if quantization:
            self.logger.info(f"🔧 Quantization: {quantization}")

        try:
            # Resolve model path based on source
            resolved_path = model_path
            if source == ModelSource.MODELSCOPE:
                resolved_path = self._cached_snapshot_download(model_path, ModelSource.MODELSCOPE)
            elif source == ModelSource.HUGGINGFACE:
                resolved_path = self._cached_snapshot_download(model_path, ModelSource.HUGGINGFACE)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                trust_remote_code=True
            )

            # Check for vLLM availability (CUDA only)
            if HAS_VLLM and torch.cuda.is_available():
                self.logger.info("⚡ Using vLLM inference engine (CUDA)")
                llm_kwargs = {
                    "model": resolved_path,
                    "trust_remote_code": trust_remote_code,
                    "tensor_parallel_size": tensor_parallel_size,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "dtype": dtype,
                    "enforce_eager": enforce_eager,
                }
                if quantization:
                    llm_kwargs["quantization"] = quantization
                if max_model_len is not None:
                    llm_kwargs["max_model_len"] = max_model_len
                if kv_cache_dtype is not None:
                    llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
                if max_num_seqs is not None:
                    llm_kwargs["max_num_seqs"] = max_num_seqs
                if max_num_batched_tokens is not None:
                    llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
                llm_kwargs.update(kwargs)

                llm = LLM(**llm_kwargs)
                self.logger.info("✅ Successfully loaded vLLM model")
            else:
                # PyTorch MPS / CPU fallback engine
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    self.logger.info("🍎 Using PyTorch MPS engine (Apple Silicon GPU)")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                    self.logger.info("⚡ Using PyTorch CUDA engine")
                else:
                    device = torch.device("cpu")
                    self.logger.info("💻 Using PyTorch CPU engine")

                torch_dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

                hf_model = AutoModelForCausalLM.from_pretrained(
                    resolved_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                ).to(device)

                llm = PyTorchCausalLMEngine(hf_model, device=device, torch_dtype=torch_dtype)
                self.logger.info("✅ Successfully loaded PyTorch CausalLM model")

            return llm, tokenizer, resolved_path

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise

    def load_funasr_model(
        self,
        repo_path: str,
        model_path: str,
        source: str = ModelSource.AUTO,
        **kwargs
    ) -> AutoModel:
        """Load FunASR model (for StepAudioTokenizer)"""
        if source == ModelSource.AUTO:
            source = self.detect_model_source(model_path)

        self.logger.info(f"Loading FunASR model from {source}: {model_path}")

        try:
            model_revision = kwargs.pop("model_revision", "main")

            if source == ModelSource.LOCAL:
                model_hub = "local"
            elif source == ModelSource.MODELSCOPE:
                model_hub = "ms"
            elif source == ModelSource.HUGGINGFACE:
                model_hub = "hf"
            else:
                raise ValueError(f"Unsupported model source: {source}")

            model = AutoModel(
                repo_path=repo_path,
                model=model_path,
                model_hub=model_hub,
                model_revision=model_revision,
                **kwargs
            )

            self.logger.info(f"✅ Successfully loaded FunASR model")
            return model

        except Exception as e:
            self.logger.error(f"❌ Failed to load FunASR model: {e}")
            raise


# Global instance
model_loader = UnifiedModelLoader()
