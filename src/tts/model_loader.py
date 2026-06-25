import os
import re
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from src.common.logging import get_logger
from src.tts.config import TTSConfig

_logger = get_logger("TTS")

class TTSModelManager:
    _instance = None

    def __new__(cls, config: TTSConfig | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = config or TTSConfig()
            cls._instance._load_model()
        return cls._instance

    @property
    def config(self) -> TTSConfig:
        return self._config

    def _load_model(self):
        cfg = self._config

        # ── TF32 加速 ──
        if cfg.tf32_enabled:
            torch.set_float32_matmul_precision('high')
            _logger.info("TF32 矩阵加速已启用")

        local_primary = os.environ.get("QWEN_TTS_MODEL_PATH") or cfg.model_path
        hf_primary = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        local_fallback = cfg.fallback_model_path
        hf_fallback = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        use_gpu = torch.cuda.is_available()
        device = "cuda:0" if use_gpu else "cpu"

        # 检查 GPU 是否支持 bfloat16
        supports_bf16 = False
        if use_gpu:
            try:
                caps = torch.cuda.get_device_capability()
                supports_bf16 = caps[0] >= 8
                _logger.info(f"GPU: {torch.cuda.get_device_name(0)}, CC={caps[0]}.{caps[1]}, bf16={supports_bf16}")
            except Exception:
                pass

        if cfg.dtype == "bfloat16" and use_gpu and supports_bf16:
            compute_dtype = torch.bfloat16
        elif cfg.dtype == "bfloat16" and not supports_bf16:
            _logger.warning("GPU 不支持 bfloat16，回退到 float32")
            compute_dtype = torch.float32
        else:
            compute_dtype = torch.float32
        _logger.info(f"使用精度: {compute_dtype}")

        _logger.info(f"加载模型: {local_primary} on {device}")

        try:
            load_kwargs = {
                "device_map": device,
                "dtype": compute_dtype,
                "attn_implementation": cfg.engine,
            }
            if os.path.exists(local_primary):
                _logger.info("开始加载模型...")
                self.model = Qwen3TTSModel.from_pretrained(local_primary, **load_kwargs)
                _logger.success(f"模型加载成功: {local_primary}")
            else:
                _logger.info(f"本地模型不存在，从 HuggingFace 加载: {hf_primary}")
                self.model = Qwen3TTSModel.from_pretrained(hf_primary, **load_kwargs)

            _logger.success(f"模型就绪: device={self.model.device}")
            speakers = self.model.get_supported_speakers()
            languages = self.model.get_supported_languages()
            _logger.info(f"支持音色({len(speakers)}): {speakers}")
            _logger.info(f"支持语言({len(languages)}): {languages}")

            # ── torch.compile 优化 ──
            if cfg.compile_enabled:
                _logger.info(f"torch.compile 优化中 (mode={cfg.compile_mode}, 首次较慢)...")
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                try:
                    self.model.model = torch.compile(
                        self.model.model,
                        mode=cfg.compile_mode,
                    )
                    # 预热一次让 CUDA 完成编译
                    self.model.generate_custom_voice(
                        text="Test.", language="English", speaker="Ono_Anna",
                        instruct=None, non_streaming_mode=True
                    )
                    _logger.success("torch.compile 优化完成")
                except Exception as ce:
                    _logger.warning(f"torch.compile 失败: {ce}，回退到 eager 模式")

        except Exception as e:
            _logger.error(f"模型加载失败: {e}")
            _logger.warning(f"降级加载: {local_fallback} 或 {hf_fallback}")
            try:
                _logger.info("开始加载降级模型...")
                fallback_kwargs = {
                    "device_map": device,
                    "dtype": compute_dtype,
                    "attn_implementation": cfg.engine,
                }
                if os.path.exists(local_fallback):
                    self.model = Qwen3TTSModel.from_pretrained(local_fallback, **fallback_kwargs)
                    _logger.success(f"降级加载成功: {local_fallback}")
                else:
                    self.model = Qwen3TTSModel.from_pretrained(hf_fallback, **fallback_kwargs)
                    _logger.success(f"HuggingFace 降级加载成功: {hf_fallback}")

                _logger.success(f"模型就绪: device={self.model.device}")
                speakers = self.model.get_supported_speakers()
                languages = self.model.get_supported_languages()
                _logger.info(f"支持音色({len(speakers)}): {speakers}")
                _logger.info(f"支持语言({len(languages)}): {languages}")
            except Exception as e2:
                raise RuntimeError(f"All model loading attempts failed: {e2}")

    def generate(self, text: str, voice: str = "", language: str = "",
                 instructions: str = "", streaming: bool = False):
        """
        文本转语音，自动处理文本质量问题和分句。

        Qwen3-TTS 的特性:
        - 单次生成建议 20-300 字符。过短无韵律上下文，过长 token 溢出。
        - 极短文本（如单个单词）需要标点或载体句提供韵律信息。
        - 本方法自动：短词加标点、长文本分句合并。
        """
        cfg = self._config
        voice = voice or cfg.default_voice
        language = language or cfg.default_language

        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        instructions = instructions.strip() if instructions else None

        # ── 极短文本修复 ──
        if len(text) < cfg.short_text_threshold and not re.search(r'[.!?]$', text):
            text = text[0].upper() + text[1:] if text else text
            text = text.rstrip(',;:') + '.'
            _logger.debug(f"短文本自动补标点: {text[:60]}")

        # ── 短文本直接生成 ──
        if len(text) <= cfg.max_chunk_chars:
            try:
                wavs, sr = self.model.generate_custom_voice(
                    text=text, language=language, speaker=voice,
                    instruct=instructions,
                    non_streaming_mode=not streaming
                )
                return wavs, sr
            except Exception as e:
                raise RuntimeError(f"TTS generation failed: {e}")

        # ── 长文本分句合并 ──
        _logger.info(f"长文本 ({len(text)} 字符)，分句生成...")
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        buf = ""
        for s in sentences:
            if buf and len(buf) + len(s) > cfg.max_chunk_chars:
                chunks.append(buf.strip())
                buf = s
            else:
                buf = buf + " " + s if buf else s
        if buf.strip():
            chunks.append(buf.strip())

        _logger.info(f"分为 {len(chunks)} 个块 (原 {len(sentences)} 句)")

        all_wavs = []
        sr = None
        for i, chunk in enumerate(chunks):
            _logger.info(f"生成块 {i+1}/{len(chunks)} ({len(chunk)} 字符)...")
            try:
                wavs, sr = self.model.generate_custom_voice(
                    text=chunk, language=language, speaker=voice,
                    instruct=instructions if i == 0 else None,
                    non_streaming_mode=True
                )
                all_wavs.append(wavs[0])
            except Exception as e:
                _logger.error(f"块 {i+1} 生成失败: {e}")
                raise RuntimeError(f"TTS failed at chunk {i+1}: {e}")

        if not all_wavs:
            raise RuntimeError("No audio generated")

        pause_ms = cfg.sentence_pause_ms / 1000.0
        pause = np.zeros(int(sr * pause_ms), dtype=all_wavs[0].dtype)
        combined = all_wavs[0]
        for wav in all_wavs[1:]:
            combined = np.concatenate([combined, pause, wav])

        _logger.success(f"长文本完成: {len(chunks)} 块, {len(combined)/sr:.1f}s")
        return [combined], sr