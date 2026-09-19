import os
import re
import sys
import numpy as np
import torch
import torch._dynamo
from qwen_tts import Qwen3TTSModel
from src.common.logging import get_logger
from src.tts.config import TTSConfig

# ── faster 后端（CUDA Graph）本地 backport 的导入路径 ──
# .backport/faster_qwen3_tts 是打了 transformers 4.x 兼容补丁的副本（MIT，
# 见 .backport/README.md）。它**未安装**成包，靠 sys.path 注入，
# 这样主环境依赖保持零变更（transformers 仍是 4.57.3）。
_BACKPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".backport",
)


def _import_faster_cls():
    """导入 backport 版 FasterQwen3TTS；不可用时返回 None（由调用方降级）。"""
    if _BACKPORT_DIR not in sys.path:
        sys.path.insert(0, _BACKPORT_DIR)
    try:
        from faster_qwen3_tts import FasterQwen3TTS  # type: ignore
        return FasterQwen3TTS
    except Exception as e:  # noqa: BLE001
        _logger.warning(f"faster 后端不可用（{type(e).__name__}: {e}），将使用 upstream 后端")
        return None


# ── 全局 TF32 加速 (Ampere+ GPU, CC≥8.0) ──
# 在模块导入时设置一次即可
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

_logger = get_logger("TTS")

# 语言简写/别名 → 完整名称映射（模型支持的完整名称见 get_supported_languages()）。
# 用于兼容客户端传入的 ISO 语言代码（如 en / zh / ja），不区分大小写。
_LANGUAGE_ALIASES = {
    "en": "english", "eng": "english",
    "zh": "chinese", "zh-cn": "chinese", "zh-hans": "chinese", "zh-hant": "chinese",
    "ch": "chinese", "cn": "chinese", "chs": "chinese",
    "fr": "french", "fra": "french", "fre": "french",
    "de": "german", "deu": "german", "ger": "german",
    "it": "italian", "ita": "italian",
    "ja": "japanese", "jp": "japanese", "jpn": "japanese",
    "ko": "korean", "kr": "korean", "kor": "korean",
    "pt": "portuguese", "por": "portuguese",
    "ru": "russian", "rus": "russian",
    "es": "spanish", "spa": "spanish",
}

# 语言 → 母语音色映射（官方 README 建议使用音色母语以获得最佳质量；
# 客户端未指定音色时按语言自动匹配，避免跨语言生成劣化）
_LANG_DEFAULT_VOICE = {
    "english": "aiden",
    "chinese": "vivian",
    "japanese": "ono_anna",
    "korean": "sohee",
    "french": "aiden", "german": "aiden", "italian": "aiden",
    "portuguese": "aiden", "russian": "aiden", "spanish": "aiden",
}


def _normalize_language(language):
    """将语言简写/别名规范化为完整名称（不区分大小写），未命中时原样返回。"""
    if not language:
        return language
    key = str(language).strip().lower()
    return _LANGUAGE_ALIASES.get(key, str(language).strip())


def _pick_default_voice(language, fallback: str) -> str:
    """按语言返回母语音色；非母语映射未覆盖时回退默认音色。"""
    lang = _normalize_language(language)
    if lang:
        return _LANG_DEFAULT_VOICE.get(lang.strip().lower(), fallback)
    return fallback


def _derive_seed(text: str, voice: str, language: str,
                 instructions: str | None) -> int:
    """由生成输入派生一个稳定的 seed（确定性采样用）。

    为什么需要：对话朗读是"点击才生成"，同一条回复可能被点多次。
    若每次随机采样，用户会听到"同一句每次念得不一样"，且重复点击会导致
    听感不一致。用文本派生的固定 seed 后，同一 (文本,音色,语言,指令)
    每次生成**完全相同**的音频——从而既不需要缓存、也不浪费 GPU。

    注意：不同文本几乎必然得到不同 seed（SHA-256 取模），
    所以音频的多样性不受影响（该有的语调变化仍在）。
    """
    import hashlib
    key = "\u0001".join([
        (text or "").strip(),
        str(voice or "").strip().lower(),
        str(language or "").strip().lower(),
        (instructions or "").strip(),
    ])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # 取前 4 字节 → 31 位正整数，避免超过部分后端 seed 的 int32 范围
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# 尾部标点（服务端短文本修复可能补句号，判定时剥离）
_TRAILING_PUNCT = ".,;:!?'\"。，；：！？’\""


def is_single_word(text: str, max_chars: int = 40) -> bool:
    """
    判定是否为"单词级"文本（听写场景）。

    规则:
      - 剥离尾部标点（含补的句号）后非空
      - 无内部空白（英文多个词、短句 → False）
      - 英文/数字/连字符/撇号，或纯中文串（无空格视为单个词）
      - 长度不超过 max_chars 兜底

    示例:
      'ahead' / 'Ahead.' / 'well' / '你好' / "don't" / 'a-head' → True
      'in the morning' / 'Hello. This is a speed benchmark test.' → False
    """
    t = (text or "").strip().rstrip(_TRAILING_PUNCT)
    if not t or len(t) > max_chars:
        return False
    if re.search(r"\s", t):
        return False
    return bool(re.fullmatch(r"[a-zA-Z0-9'’-]+", t) or
                re.fullmatch(r"[\u4e00-\u9fff]+", t))


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
            if cfg.backend == "faster":
                self.model = self._load_faster(local_primary, hf_primary, device, compute_dtype, cfg)
            elif os.path.exists(local_primary):
                _logger.info("开始加载模型...")
                self.model = Qwen3TTSModel.from_pretrained(local_primary, **load_kwargs)
                _logger.success(f"模型加载成功: {local_primary}")
            else:
                _logger.info(f"本地模型不存在，从 HuggingFace 加载: {hf_primary}")
                self.model = Qwen3TTSModel.from_pretrained(hf_primary, **load_kwargs)

            _logger.success(f"模型就绪: device={self.model.device}")
            speakers = self.get_supported_speakers()
            languages = self.get_supported_languages()
            _logger.info(f"支持音色({len(speakers)}): {speakers}")
            _logger.info(f"支持语言({len(languages)}): {languages}")

            # ── torch.compile 优化（仅 upstream 后端需要；faster 后端自带 CUDA Graph）──
            if cfg.compile_enabled and cfg.backend != "faster":
                _logger.info(f"torch.compile 优化中 (mode={cfg.compile_mode}, 首次较慢)...")
                torch._dynamo.config.suppress_errors = True
                try:
                    self.model.model = torch.compile(
                        self.model.model,
                        mode=cfg.compile_mode,
                    )
                    # 预热一次让 CUDA 完成编译
                    self.model.generate_custom_voice(
                        text="Test.", language="English", speaker="Aiden",
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

    def _load_faster(self, local_primary, hf_primary, device, compute_dtype, cfg):
        """加载 backport 版 FasterQwen3TTS（CUDA Graph + 静态 KV cache）。

        实测收益（RTX 4060 Laptop 8G，见 docs/FASTER_TTS_VALIDATION.md）：
        短句 RTF_wall 从 2.25x 降到 0.58x（约 4-5 倍），TTFA ~0.33s。

        与 upstream 的差异：
        - 参数名不同：device / dtype / max_seq_len（不是 device_map）
        - `warmup()` 必须调用一次以捕获 CUDA Graph（约 1.5-1.8s，很便宜）
        - 模型对象是 wrapper，内层真模型在 `.model`（上游 Qwen3TTSModel）
        """
        FasterCls = _import_faster_cls()
        if FasterCls is None:
            _logger.warning("faster 后端不可用，回退 upstream 后端")
            self._backend_actual = "upstream"
            if os.path.exists(local_primary):
                return Qwen3TTSModel.from_pretrained(
                    local_primary, device_map=device, dtype=compute_dtype,
                    attn_implementation=cfg.engine)
            return Qwen3TTSModel.from_pretrained(
                hf_primary, device_map=device, dtype=compute_dtype,
                attn_implementation=cfg.engine)

        path = local_primary if os.path.exists(local_primary) else hf_primary
        _logger.info(f"[faster] 加载 CUDA Graph 后端: {path}")
        m = FasterCls.from_pretrained(
            path,
            device="cuda" if str(device).startswith("cuda") else "cpu",
            dtype=compute_dtype,
            attn_implementation=cfg.engine,
            max_seq_len=cfg.stream_max_seq_len,
        )
        # 捕获 CUDA Graph（幂等；首次约 1.5-1.8s）
        import time as _t
        t0 = _t.perf_counter()
        try:
            m.warmup(prefill_len=120)
            _logger.success(f"[faster] CUDA Graph 捕获完成 ({_t.perf_counter()-t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            _logger.warning(f"[faster] warmup 失败（首次生成会较慢）: {e}")
        self._backend_actual = "faster"
        return m

    @property
    def backend(self) -> str:
        """实际生效的后端：faster（CUDA Graph）或 upstream。"""
        return getattr(self, "_backend_actual", "upstream")

    def get_supported_speakers(self):
        """支持音色列表。

        faster 后端的 wrapper 没有这个方法，委托给它内部的上游模型对象。
        """
        m = self.model
        fn = getattr(m, "get_supported_speakers", None)
        if callable(fn):
            return fn()
        return getattr(m, "model", None) and m.model.get_supported_speakers()

    def get_supported_languages(self):
        """支持语言列表（同样需要委托到内层模型）。"""
        m = self.model
        fn = getattr(m, "get_supported_languages", None)
        if callable(fn):
            return fn()
        return getattr(m, "model", None) and m.model.get_supported_languages()

    def generate(self, text: str, voice: str = "", language: str = "",
                 instructions: str = "", streaming: bool = False,
                 verify: bool | None = None, seed: int | None = None):
        """
        文本转语音，自动处理文本质量问题和分句。

        Qwen3-TTS 的特性:
        - 单次生成建议 20-300 字符。过短无韵律上下文，过长 token 溢出。
        - 极短文本（如单个单词）需要标点或载体句提供韵律信息。
        - 本方法自动：短词加标点、长文本分句合并。

        短/长文本分治策略:
        - 单词语音（is_single_word 判定，如 "ahead" / "well" / "你好"，
          听写/单词场景）:
          温和确定性解码（低温度 + 固定 seed，重试时 seed+i）+ ASR 回读校验闭环，
          保证输出可被识别为目标单词（宽松匹配 + 置信度门槛）。
        - 长文本与短句（如 "How are you?"、对话段落）:
          保持随机采样更自然，仅做轻量时长校验，失败重试 long_max_retries 次。
          （短句生成表现好，无需严格 ASR 校验。）

        Args:
            verify: None=自动（单词语音校验/其余仅轻量），True=强制 ASR 校验（仅对单词语音生效），
                    False=跳过 ASR 校验（仍做时长检查）。
            seed: 基础随机种子（重试时 seed+i）。None 时单词语音用配置默认 seed。

        Returns:
            (wavs, sr, meta)
            meta: dict {verified, attempts, asr_text, confidence, duration, strategy, seed, details}
        """
        cfg = self._config
        # 兼容客户端传入的简写语言代码（如 "en" → "english"）
        language = _normalize_language(language) or cfg.default_language
        voice = voice or _pick_default_voice(language, cfg.default_voice)

        # ── 文本规范化（与 generate_stream 共用同一口径）──
        text = self._prepare_text(text, cfg)
        instructions = instructions.strip() if instructions else None

        is_short = is_single_word(text, cfg.verify_text_threshold)
        if not is_short:
            # ── 长文本/短句：分句生成 + 轻量时长校验 ──
            return self._generate_long(
                text, voice, language, instructions, streaming, cfg)
        # ── 单词语音：确定性解码 + ASR 校验闭环 ──
        return self._generate_short(
            text, voice, language, instructions, streaming,
            verify=verify, seed=seed, cfg=cfg)

    # ────────────────────────────────────────────────
    # 流式生成（按帧产出，供 /synthesize-stream 边生成边下发）
    # ────────────────────────────────────────────────
    def _prepare_text(self, text: str, cfg) -> str:
        """生成前的文本规范化（generate 与 generate_stream 共用，避免两套口径）。"""
        t = (text or "").strip()
        t = re.sub(r"\s+", " ", t)
        if not t:
            raise ValueError("text is required")
        # 极短文本补标点，提升韵律稳定性
        if len(t) < cfg.short_text_threshold and not re.search(r"[.!?]$", t):
            t = t[0].upper() + t[1:]
            t = t.rstrip(",;:") + "."
        return t

    def generate_stream(self, text: str, voice: str = "", language: str = "",
                        instructions: str = "", chunk_steps: int | None = None,
                        max_new_tokens: int | None = None, seed: int | None = None):
        """流式生成，逐片 yield (wav_float32, sample_rate)。

        与 generate() 的区别：**不等整段生成完**，首片约 330ms 即可拿到
        （实测与文本长度无关，见 docs/STREAMING_ARCHITECTURE_ANALYSIS.md §2.5）。

        **确定性（seed）**：对话朗读采用"点击才生成"的语义，用户可能对同一条回复
        多次点击。因此默认用**由文本派生的固定 seed**，保证同一 (文本,音色,语言,指令)
        每次生成出**完全相同**的音频——用户不会听到"同一句每次念得不一样"，
        也不需要为此落盘缓存。传 `seed=None` 之外的显式值可覆盖。

        注意：
        - 本方法**不做** ASR 校验（流式实时性优先，与 /stream 的既有约定一致）；
        - 调用方**必须**持有 app.state.model_lock（约束 P1）；
        - 生成是阻塞的，需在独立线程中消费（见 router 的实现）；
        - `max_new_tokens` 必须 <= 模型静态缓存的 max_seq_len，
          否则底层**静默截断**（见 docs/CONV_SPEED_ASSESSMENT.md §4.2）。
        """
        cfg = self._config
        language = _normalize_language(language) or cfg.default_language
        voice = voice or _pick_default_voice(language, cfg.default_voice)
        t = self._prepare_text(text, cfg)
        ins = instructions.strip() if instructions else None

        chunk_steps = chunk_steps or cfg.stream_chunk_steps
        max_new = max_new_tokens or cfg.stream_max_new_tokens

        # 防止超长文本被静默截断：静态缓存容量是硬上限
        max_seq = getattr(self.model, "max_seq_len", None)
        if max_seq and max_new > max_seq:
            _logger.warning(
                f"max_new_tokens({max_new}) > 模型 max_seq_len({max_seq})，"
                f"已收敛为 {max_seq}（超出会静默截断音频）"
            )
            max_new = max_seq

        # ── 确定性地播种 ──
        # 底层流式 API 不支持 seed 参数。这里**不能**只用 torch.manual_seed：
        # CUDA Graph 重放会捕获/影响全局 CUDA RNG 状态，实测同一文本两次结果不同
        # （见 bench/verify_stream_determinism.py）。因此额外构造一个**独立的
        # torch.Generator** 传给采样，使随机源与全局状态解耦，才能真正复现。
        #
        # ⚠️ generator 的设备必须与 logits 一致：cuda 张量只接受 cuda generator
        # （否则 RuntimeError: Expected a 'cuda' device type for generator）。
        if seed is None:
            seed = _derive_seed(t, voice, language, ins)
        torch.manual_seed(seed)
        try:
            dev = next(self.model.parameters()).device
        except Exception:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        gen = torch.Generator(device=dev.type)
        gen.manual_seed(seed)

        _logger.info(
            f"流式生成开始: len={len(t)} voice={voice} lang={language} "
            f"chunk_steps={chunk_steps} max_new={max_new} seed={seed} "
            f"gen_device={dev.type} backend={self.backend}"
        )

        # 结构说明（实测确认）：
        #   faster 后端: self.model 是 FasterQwen3TTS wrapper
        #                → 流式 API 在 **wrapper 上**（不是 .model 上）
        #                → self.model.model 才是上游 Qwen3TTSModel
        #   upstream  : self.model 本身就是 Qwen3TTSModel，**没有流式 API**
        stream_fn = getattr(self.model, "generate_custom_voice_streaming", None)
        if callable(stream_fn):
            # faster 后端：真正逐片产出（首片 ~330ms）
            for wav, sr, _timing in stream_fn(
                text=t, speaker=voice, language=language, instruct=ins,
                chunk_size=chunk_steps, max_new_tokens=max_new,
                generator=gen,
            ):
                yield wav, sr
        else:
            # upstream 后端：没有流式 API —— 整段生成后作为**单片**返回。
            # 这仍然有用（HTTP 层可以早发头、且与 faster 后端接口一致），
            # 但**首片延迟等于整段生成时间**。日志明确提示，避免误判。
            _logger.warning(
                "当前后端不支持真正的流式生成（upstream），整段生成后单片返回；"
                "如需低首片延迟请设置 tts.backend=faster"
            )
            inner = getattr(self.model, "model", self.model)
            wavs, sr = inner.generate_custom_voice(
                text=t, language=language, speaker=voice, instruct=ins,
                non_streaming_mode=True,
                max_new_tokens=min(max_new, 2048),
            )
            yield np.asarray(wavs[0], dtype=np.float32), sr

    # ────────────────────────────────────────────────
    # 短文本（单词/听写）：确定性解码 + ASR 校验闭环
    # ────────────────────────────────────────────────
    def _generate_short(self, text, voice, language, instructions, streaming,
                        verify, seed, cfg):
        decode = cfg.short_decode
        # None → 短文本默认校验；True → 强制校验；False → 跳过 ASR（仍查时长）
        do_verify = verify is not False
        base_seed = seed if seed is not None else decode.get("seed")
        attempts = cfg.verify_max_retries if do_verify else 1

        meta = {
            "verified": False, "attempts": 0,
            "asr_text": None, "confidence": None,
            "duration": 0.0, "strategy": "short",
            "seed": base_seed, "details": [],
        }
        last = None

        for i in range(attempts):
            cur_seed = (base_seed + i) if base_seed is not None else None
            if cur_seed is not None:
                torch.manual_seed(cur_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(cur_seed)
            try:
                wavs, sr = self.model.generate_custom_voice(
                    text=text, language=language, speaker=voice,
                    instruct=instructions,
                    non_streaming_mode=not streaming,
                    temperature=decode["temperature"],
                    top_k=decode["top_k"],
                    top_p=decode["top_p"],
                    repetition_penalty=decode["repetition_penalty"],
                    max_new_tokens=decode["max_new_tokens"],
                )
            except Exception as e:
                if i == attempts - 1:
                    raise RuntimeError(f"TTS generation failed: {e}")
                _logger.warning(f"短文本生成第 {i+1} 次失败: {e}，换 seed 重试")
                continue

            wav = wavs[0]
            duration = len(wav) / sr
            meta["duration"] = duration
            meta["seed"] = cur_seed
            last = (wavs, sr)

            if streaming:
                # 流式路径不做校验（WebSocket 实时性优先），直接返回
                meta["verified"] = True
                meta["attempts"] = i + 1
                return wavs, sr, meta

            # 时长检查：单词音频超出合理范围 → 疑似填充码循环，判失败重试
            if not (0.05 <= duration <= cfg.word_max_duration_s):
                meta["details"].append({
                    "attempt": i + 1, "seed": cur_seed,
                    "duration": round(duration, 3), "asr_text": None,
                    "verified": False, "reason": "duration_out_of_range",
                })
                _logger.warning(
                    f"短文本时长异常 {duration:.2f}s (seed={cur_seed})，重试"
                )
                continue

            if do_verify:
                ok, asr_text, conf = self._verify_audio(
                    wav, sr, text, language, cfg)
                meta["asr_text"] = asr_text
                meta["confidence"] = conf
                meta["details"].append({
                    "attempt": i + 1, "seed": cur_seed,
                    "duration": round(duration, 3), "asr_text": asr_text,
                    "verified": ok,
                    "reason": None if ok else "asr_mismatch",
                })
                if ok:
                    meta["verified"] = True
                    meta["attempts"] = i + 1
                    _logger.success(
                        f"短文本校验通过: {text!r} asr={asr_text!r} "
                        f"attempt={i+1} seed={cur_seed}"
                    )
                    return wavs, sr, meta
                _logger.warning(
                    f"短文本 ASR 校验失败: target={text!r} asr={asr_text!r} "
                    f"attempt={i+1}/{attempts}"
                )
            else:
                meta["verified"] = True
                meta["attempts"] = i + 1
                return wavs, sr, meta

        # 全部失败：宽松降级，返回最后一次生成的音频并标记 verified=false
        meta["attempts"] = attempts
        if last is not None:
            _logger.warning(
                f"短文本 {text!r} 重试 {attempts} 次仍未通过校验，降级返回 "
                f"(verified=false)"
            )
            return last[0], last[1], meta
        raise RuntimeError("TTS generation failed: no audio produced")

    # ────────────────────────────────────────────────
    # 长文本（AI 对话）：分句生成 + 轻量时长校验
    # ────────────────────────────────────────────────
    def _generate_long(self, text, voice, language, instructions, streaming, cfg):
        decode = cfg.long_decode
        max_retries = cfg.long_max_retries

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
            generated = False
            for attempt in range(max_retries):
                try:
                    wavs, sr = self.model.generate_custom_voice(
                        text=chunk, language=language, speaker=voice,
                        instruct=instructions if i == 0 else None,
                        non_streaming_mode=True,
                        temperature=decode["temperature"],
                        top_k=decode["top_k"],
                        top_p=decode["top_p"],
                        repetition_penalty=decode["repetition_penalty"],
                        max_new_tokens=decode["max_new_tokens"],
                    )
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"TTS failed at chunk {i+1}: {e}")
                    _logger.warning(f"块 {i+1} 生成失败: {e}，重试")
                    continue
                wav = wavs[0]
                duration = len(wav) / sr
                max_dur = len(chunk) * cfg.long_duration_factor + cfg.long_duration_bias
                if 0.3 <= duration <= max_dur or attempt == max_retries - 1:
                    # 时长合理，或已到最后一次重试（降级接受）
                    all_wavs.append(wav)
                    generated = True
                    break
                _logger.warning(
                    f"块 {i+1} 时长异常 {duration:.2f}s "
                    f"(合理区间 0.3-{max_dur:.1f}s)，重试"
                )
            if not generated:
                raise RuntimeError(f"TTS failed at chunk {i+1}: no audio produced")

        if not all_wavs:
            raise RuntimeError("No audio generated")

        pause_ms = cfg.sentence_pause_ms / 1000.0
        pause = np.zeros(int(sr * pause_ms), dtype=all_wavs[0].dtype)
        combined = all_wavs[0]
        for wav in all_wavs[1:]:
            combined = np.concatenate([combined, pause, wav])

        meta = {
            "verified": True, "attempts": len(chunks),
            "asr_text": None, "confidence": None,
            "duration": len(combined) / sr, "strategy": "long",
            "seed": None, "details": [],
        }
        _logger.success(f"长文本完成: {len(chunks)} 块, {meta['duration']:.1f}s")
        return [combined], sr, meta

    # ────────────────────────────────────────────────
    # ASR 回读校验（宽松匹配 + 置信度门槛）
    # ────────────────────────────────────────────────
    def _verify_audio(self, wav, sr, target_text, language, cfg):
        from src.tts.verifier import verify_audio
        from src.asr.model_loader import ASRModelManager
        asr = ASRModelManager()
        return verify_audio(
            wav=wav, sr=sr, target_text=target_text, language=language,
            asr_manager=asr,
            confidence_threshold=cfg.asr_confidence_threshold,
            beam_size=5,
        )