"""
TTS 配置管理 — 模型选择、性能优化、文本处理参数
继承自 src.common.base_config.BaseConfig
"""
from pathlib import Path
from typing import Optional, Literal
from src.common.base_config import BaseConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "tts.yaml"

TtsEngine = Literal["sdpa", "flash_attention_2"]
TtsDtype = Literal["bfloat16", "float32"]
CompileMode = Literal["reduce-overhead", "max-autotune", "default"]


class TTSConfig(BaseConfig):
    """TTS 配置管理器"""
    env_prefix = "TTSCONF_"
    _default_config_path = DEFAULT_CONFIG_PATH

    # ── 模型 ──

    @property
    def engine(self) -> TtsEngine:
        return self._env("engine") or self._get("tts.engine", "sdpa")

    @property
    def dtype(self) -> TtsDtype:
        return self._env("dtype") or self._get("tts.dtype", "bfloat16")

    @property
    def model_path(self) -> str:
        return self._env("model_path") or self._get("tts.model_path", "./models/qwen-1.7b")

    @property
    def fallback_model_path(self) -> str:
        return self._get("tts.fallback_model_path", "./models/qwen-0.6b")

    # ── 性能提示 ──
    # 对于 8GB VRAM 的 GPU (如 RTX 4060)：
    #   - 1.7B 模型 (bfloat16) 约需 3.4GB VRAM，生成较慢
    #   - 0.6B 模型 (bfloat16) 约需 1.2GB VRAM，生成速度快 2-3 倍
    #   - 对话场景建议使用 0.6B 模型（质量差异在短句中不明显）
    #   - 在 config/tts.yaml 中设置 tts.model_path: ./models/qwen-0.6b

    # ── 默认参数 ──

    @property
    def default_voice(self) -> str:
        return self._get("tts.default_voice", "Ono_Anna")

    @property
    def default_language(self) -> str:
        return self._get("tts.default_language", "English")

    # ── 性能优化 ──

    @property
    def compile_enabled(self) -> bool:
        val = self._env("compile") or self._get("tts.performance.compile", False)
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    @property
    def tf32_enabled(self) -> bool:
        val = self._env("tf32_matmul") or self._get("tts.performance.tf32_matmul", True)
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    @property
    def compile_mode(self) -> CompileMode:
        return self._env("compile_mode") or self._get("tts.performance.compile_mode", "reduce-overhead")

    # ── 文本处理 ──

    @property
    def max_chunk_chars(self) -> int:
        return int(self._get("tts.text.max_chunk_chars", 250))

    @property
    def short_text_threshold(self) -> int:
        return int(self._get("tts.text.short_text_threshold", 80))

    @property
    def sentence_pause_ms(self) -> int:
        return int(self._get("tts.text.sentence_pause_ms", 180))

    # ── 导出 ──

    def summary(self) -> dict:
        return {
            "model_path": self.model_path,
            "engine": self.engine, "dtype": self.dtype,
            "compile": self.compile_enabled,
            "compile_mode": self.compile_mode,
            "tf32_matmul": self.tf32_enabled,
            "max_chunk_chars": self.max_chunk_chars,
            "short_text_threshold": self.short_text_threshold,
        }

    def print_summary(self):
        s = self.summary()
        print("=" * 50)
        print("  TTS 配置")
        print("=" * 50)
        for l, v in [
            ("模型", s["model_path"]),
            ("引擎/精度", f"{s['engine']}/{s['dtype']}"),
            ("compile优化", f"{s['compile']} (mode={s['compile_mode']})"),
            ("TF32加速", s["tf32_matmul"]),
            ("文本分块", f"{s['max_chunk_chars']}字符, 短文本阈值{s['short_text_threshold']}"),
        ]:
            print(f"  {l}: {v}")
        print("=" * 50)
