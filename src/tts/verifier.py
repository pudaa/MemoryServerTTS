"""
TTS 生成结果校验模块 — ASR 回读比对（听写/单词场景的治本闭环）

判定策略（宽松 + 置信度门槛）:
  1. 归一化：小写、去除标点/空白（含中文保留）
  2. 宽松匹配：归一化后完全相等 / 互为子串 / 编辑距离 ≤ 1
     —— 容忍 Whisper 对极短音频的常见差异：ahead ↔ "a head"、
        "Mm-hmm ahead" 前缀、标点差异等
  3. 置信度门槛：ASR 平均对数概率低于阈值判为失败（低置信度识别不可信）
"""
import re
import tempfile

from src.common.logging import get_logger

_logger = get_logger("TTS-VERIFY")

# TTS 语言全名 → Whisper ISO 语言代码（ASR 使用）
_LANG_TO_ISO = {
    "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
    "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "russian": "ru", "spanish": "es",
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+", re.IGNORECASE)


def normalize_nospace(text: str) -> str:
    """归一化为不含空白的小写字母/数字/汉字串，用于子串与编辑距离比对"""
    return _NON_ALNUM_RE.sub("", (text or "").lower())


def _levenshtein(a: str, b: str) -> int:
    """编辑距离（DP），仅用于短文本比对（≤40 字符，开销可忽略）"""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def matches(target: str, hypothesis: str) -> bool:
    """
    宽松匹配：归一化后完全相等 / 互为子串 / 编辑距离 ≤ 1。
    空目标视为无需校验（直接通过）。
    """
    t = normalize_nospace(target)
    h = normalize_nospace(hypothesis)
    if not t:
        return True
    if not h:
        return False
    if t == h:
        return True
    if t in h or h in t:
        return True
    return _levenshtein(t, h) <= 1


def verify_audio(wav, sr, target_text: str, language: str,
                 asr_manager, confidence_threshold: float = -1.0,
                 beam_size: int = 5):
    """
    对生成音频做 ASR 回读校验。

    Args:
        wav: numpy 波形（一维 float）
        sr: 采样率
        target_text: 目标文本（原始单词，如 "ahead"）
        language: TTS 语言全名（如 "English"），用于映射 Whisper 语言代码
        asr_manager: ASRModelManager 实例
        confidence_threshold: ASR 平均对数概率门槛
        beam_size: Whisper 束搜索大小

    Returns:
        (verified: bool, asr_text: str, avg_logprob: float|None)
    """
    iso = _LANG_TO_ISO.get((language or "").strip().lower())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
        tmp = tf.name
    try:
        import soundfile as sf
        sf.write(tmp, wav, sr)
        result = asr_manager.transcribe(
            audio_path=tmp, language=iso, task="transcribe", beam_size=beam_size
        )
        asr_text = (result.get("text") or "").strip()
        conf = result.get("avg_logprob")
        if not asr_text:
            return False, "", conf
        ok = matches(target_text, asr_text)
        if ok and conf is not None and conf < confidence_threshold:
            _logger.debug(
                f"ASR 匹配但置信度不足: target={target_text!r} asr={asr_text!r} "
                f"avg_logprob={conf:.3f} < {confidence_threshold}"
            )
            return False, asr_text, conf
        if not ok:
            _logger.debug(f"ASR 不匹配: target={target_text!r} asr={asr_text!r}")
        return ok, asr_text, conf
    except Exception as e:
        _logger.warning(f"ASR 校验失败（视为不通过）: {e}")
        return False, "", None
    finally:
        try:
            import os
            os.remove(tmp)
        except OSError:
            pass
