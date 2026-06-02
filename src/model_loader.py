import os
import torch
from qwen_tts import Qwen3TTSModel

class TTSModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        # 内存紧张时优先加载 0.6B 小模型，可通过环境变量切换
        local_primary = os.environ.get("QWEN_TTS_MODEL_PATH", "./models/qwen-0.6b")
        hf_primary = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        local_fallback = "./models/qwen-1.7b"
        hf_fallback = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        use_gpu = torch.cuda.is_available()
        device = "cuda:0" if use_gpu else "cpu"

        # 检查 GPU 是否支持 bfloat16
        supports_bf16 = False
        if use_gpu:
            try:
                caps = torch.cuda.get_device_capability()
                supports_bf16 = caps[0] >= 8  # Ampere (SM 8.0+) 及以上才支持 bfloat16
                print(f"[TTS] GPU: {torch.cuda.get_device_name(0)}, 计算能力: {caps[0]}.{caps[1]}, bf16支持: {supports_bf16}")
            except Exception:
                pass

        compute_dtype = torch.bfloat16 if (use_gpu and supports_bf16) else torch.float32
        print(f"[TTS] 使用精度: {compute_dtype}")

        # 优先尝试 0.6B 小模型（内存友好）
        print(f"[TTS] 优先加载 0.6B 模型: {local_primary} (不存在则回退 HuggingFace: {hf_primary}) on {device}")

        try:
            load_kwargs = {
                "device_map": device,
                "dtype": compute_dtype,
                "attn_implementation": "sdpa",
                "low_cpu_mem_usage": True,
            }
            if os.path.exists(local_primary):
                print(f"[TTS] 开始加载 0.6B 模型（约需 20-40 秒）...")
                self.model = Qwen3TTSModel.from_pretrained(
                    local_primary,
                    **load_kwargs
                )
                print(f"[TTS] 0.6B 模型加载成功: {local_primary}")
            else:
                print(f"[TTS] 本地 0.6B 不存在，尝试 HuggingFace: {hf_primary}")
                self.model = Qwen3TTSModel.from_pretrained(
                    hf_primary,
                    **load_kwargs
                )
                print(f"[TTS] HuggingFace 0.6B 加载成功: {hf_primary}")

            print(f"Model loaded successfully on device: {self.model.device}")
            speakers = self.model.get_supported_speakers()
            languages = self.model.get_supported_languages()
            print(f"Supported speakers ({len(speakers)}): {speakers}")
            print(f"Supported languages ({len(languages)}): {languages}")

        except Exception as e:
            print(f"[TTS] 0.6B 加载失败: {e}")
            # 降级方案：尝试加载 1.7B 模型
            print(f"[TTS] 尝试降级加载 1.7B: {local_fallback} 或 {hf_fallback}")
            try:
                print(f"[TTS] 开始加载 1.7B 模型（约需 30-60 秒）...")
                fallback_kwargs = {
                    "device_map": device,
                    "dtype": compute_dtype,
                    "attn_implementation": "sdpa",
                    "low_cpu_mem_usage": True,
                }
                if os.path.exists(local_fallback):
                    self.model = Qwen3TTSModel.from_pretrained(
                        local_fallback,
                        **fallback_kwargs
                    )
                    print(f"[TTS] 1.7B 降级模型加载成功: {local_fallback}")
                else:
                    self.model = Qwen3TTSModel.from_pretrained(
                        hf_fallback,
                        **fallback_kwargs
                    )
                    print(f"[TTS] HuggingFace 1.7B 加载成功: {hf_fallback}")

                print(f"Model loaded successfully on device: {self.model.device}")
            except Exception as e2:
                raise RuntimeError(f"All model loading attempts failed: {e2}")

    def generate(self, text: str, voice: str = "Ryan", language: str = "English",
                 instructions: str = "", streaming: bool = False):
        """
        根据官方文档调用 generate_custom_voice
        """
        try:
            # 官方 API: generate_custom_voice(text, language, speaker, instruct)
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
                instruct=instructions if instructions else None,
                non_streaming_mode=not streaming # 控制是否流式
            )
            return wavs, sr
        except Exception as e:
            raise RuntimeError(f"TTS generation failed: {e}")