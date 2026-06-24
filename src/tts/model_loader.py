import os
import re
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from src.common.logging import get_logger

_logger = get_logger("TTS")

class TTSModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        # 优先加载 1.7B 模型（速度更快，token 效率更高）
        # 内存紧张时可通过环境变量切换到 0.6B
        local_primary = os.environ.get("QWEN_TTS_MODEL_PATH", "./models/qwen-1.7b")
        hf_primary = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        local_fallback = "./models/qwen-0.6b"
        hf_fallback = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        use_gpu = torch.cuda.is_available()
        device = "cuda:0" if use_gpu else "cpu"

        # 检查 GPU 是否支持 bfloat16
        supports_bf16 = False
        if use_gpu:
            try:
                caps = torch.cuda.get_device_capability()
                supports_bf16 = caps[0] >= 8  # Ampere (SM 8.0+) 及以上才支持 bfloat16
                _logger.info(f"GPU: {torch.cuda.get_device_name(0)}, CC={caps[0]}.{caps[1]}, bf16={supports_bf16}")
            except Exception:
                pass

        compute_dtype = torch.bfloat16 if (use_gpu and supports_bf16) else torch.float32
        _logger.info(f"使用精度: {compute_dtype}")

        # 优先尝试 0.6B 小模型（内存友好）
        _logger.info(f"加载 1.7B 模型: {local_primary} (回退: {hf_primary}) on {device}")

        try:
            load_kwargs = {
                "device_map": device,
                "dtype": compute_dtype,
                "attn_implementation": "sdpa",
            }
            if os.path.exists(local_primary):
                _logger.info("开始加载 1.7B 模型...")
                self.model = Qwen3TTSModel.from_pretrained(
                    local_primary,
                    **load_kwargs
                )
                _logger.success(f"1.7B 加载成功: {local_primary}")
            else:
                _logger.info(f"本地 1.7B 不存在，从 HuggingFace 加载: {hf_primary}")
                self.model = Qwen3TTSModel.from_pretrained(
                    hf_primary,
                    **load_kwargs
                )
                _logger.success(f"HuggingFace 1.7B 加载成功: {hf_primary}")

            _logger.success(f"模型就绪: device={self.model.device}")
            speakers = self.model.get_supported_speakers()
            languages = self.model.get_supported_languages()
            _logger.info(f"支持音色({len(speakers)}): {speakers}")
            _logger.info(f"支持语言({len(languages)}): {languages}")

        except Exception as e:
            _logger.error(f"1.7B 加载失败: {e}")
            _logger.warning(f"降级加载 0.6B: {local_fallback} 或 {hf_fallback}")
            try:
                _logger.info("开始加载 0.6B 模型...")
                fallback_kwargs = {
                    "device_map": device,
                    "dtype": compute_dtype,
                    "attn_implementation": "sdpa",
                }
                if os.path.exists(local_fallback):
                    self.model = Qwen3TTSModel.from_pretrained(
                        local_fallback,
                        **fallback_kwargs
                    )
                    _logger.success(f"0.6B 降级加载成功: {local_fallback}")
                else:
                    self.model = Qwen3TTSModel.from_pretrained(
                        hf_fallback,
                        **fallback_kwargs
                    )
                    _logger.success(f"HuggingFace 0.6B 加载成功: {hf_fallback}")

                _logger.success(f"模型就绪: device={self.model.device}")
                speakers = self.model.get_supported_speakers()
                languages = self.model.get_supported_languages()
                _logger.info(f"支持音色({len(speakers)}): {speakers}")
                _logger.info(f"支持语言({len(languages)}): {languages}")
            except Exception as e2:
                raise RuntimeError(f"All model loading attempts failed: {e2}")

    def generate(self, text: str, voice: str = "Ryan", language: str = "English",
                 instructions: str = "", streaming: bool = False, speed_priority: bool = False):
        """
        文本转语音，自动处理文本质量问题和分句。

        Qwen3-TTS 的特性:
        - 单次生成建议 20-300 字符。过短无韵律上下文，过长 token 溢出。
        - 极短文本（如单个单词）需要标点或载体句提供韵律信息。
        - 本方法自动：短词加标点、长文本分句合并、可选提速。

        Args:
            speed_priority: True 时将短句合并为更大块以减少调用次数
        """
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        instructions = instructions.strip() if instructions else None

        # ── 极短文本修复: 加标点 + 首字母大写 ──
        if len(text) < 10 and not re.search(r'[.!?]$', text):
            text = text[0].upper() + text[1:] if text else text
            text = text.rstrip(',;:') + '.'
            _logger.debug(f"短文本自动补标点: {text[:60]}")

        # ── 短文本直接生成 ──
        if len(text) <= 250:
            try:
                wavs, sr = self.model.generate_custom_voice(
                    text=text, language=language, speaker=voice,
                    instruct=instructions,
                    non_streaming_mode=not streaming
                )
                return wavs, sr
            except Exception as e:
                raise RuntimeError(f"TTS generation failed: {e}")

        # ── 长文本分句 ──
        _logger.info(f"长文本 ({len(text)} 字符)，分句生成...")
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # 合并块: 每个块尽量接近 200 字符以减少调用次数
        chunks = []
        buf = ""
        for s in sentences:
            if buf and len(buf) + len(s) > 250:
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

        pause = np.zeros(int(sr * 0.18), dtype=all_wavs[0].dtype)
        combined = all_wavs[0]
        for wav in all_wavs[1:]:
            combined = np.concatenate([combined, pause, wav])

        _logger.success(f"长文本完成: {len(chunks)} 块, {len(combined)/sr:.1f}s")
        return [combined], sr