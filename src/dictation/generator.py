"""
词库预生成器 — best-of-N 择优 + 评分

评分逻辑（quality_score，0-1）:
  - 置信度分（0.7 权重）：ASR avg_logprob 从 -1.5 映射到 0 → 0..1
  - 时长分（0.3 权重）：单词理想时长 0.3-1.5s 得满分，越短/越长递减
  - 只有通过 ASR 校验（内容可识别 + 置信度达标）的候选才参与择优；
    全部失败则返回失败明细，由调用方决定（不缓存 / 重试 / 报告）。

依赖注入设计：synth_fn / verify_fn 由调用方提供——
  - CLI 预生成：注入 1.7B 模型生成 + 真实 ASR 校验
  - 服务端在线回填：注入已加载模型生成 + 真实 ASR 校验
  - 单元测试：注入假实现，纯测择优/评分逻辑
"""
from src.common.logging import get_logger

_logger = get_logger("DICT-GEN")

# 评分参数（单词音频理想时长窗口）
_IDEAL_DUR_MIN = 0.3
_IDEAL_DUR_MAX = 1.5
_DUR_DECAY_RANGE = 2.5          # 超过理想窗口后，再延长多少秒衰减到 0
_CONF_MAP_LO = -1.5             # avg_logprob 该值 → 置信度分 0
_CONF_MAP_HI = 0.0              # avg_logprob 该值 → 置信度分 1

# 单词音频绝对合理区间（超出直接判失败，不入候选）
_DUR_OK_MIN = 0.05
_DUR_OK_MAX = 5.0


def score_candidate(avg_logprob: float | None, duration: float) -> float:
    """
    候选质量分 0-1：0.7 * 置信度分 + 0.3 * 时长分。
    avg_logprob 为 None（未知）时置信度分取 0。
    """
    conf_score = 0.0
    if avg_logprob is not None:
        conf_score = min(max((avg_logprob - _CONF_MAP_LO) / (_CONF_MAP_HI - _CONF_MAP_LO), 0.0), 1.0)

    if _IDEAL_DUR_MIN <= duration <= _IDEAL_DUR_MAX:
        dur_score = 1.0
    elif duration < _IDEAL_DUR_MIN:
        dur_score = max(0.0, duration / _IDEAL_DUR_MIN)
    else:
        dur_score = max(0.0, 1.0 - (duration - _IDEAL_DUR_MAX) / _DUR_DECAY_RANGE)

    return round(0.7 * conf_score + 0.3 * dur_score, 3)


def make_synth_fn(model, decode: dict):
    """
    构造生成函数：设置 seed 后调用 Qwen3TTSModel.generate_custom_voice。
    model 为已加载的 Qwen3TTSModel（词库预生成固定 1.7B）。
    """
    import torch

    def synth(text, voice, language, instruct, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        wavs, sr = model.generate_custom_voice(
            text=text, language=language, speaker=voice, instruct=instruct,
            non_streaming_mode=True,
            temperature=decode["temperature"],
            top_k=decode["top_k"],
            top_p=decode["top_p"],
            repetition_penalty=decode["repetition_penalty"],
            max_new_tokens=decode["max_new_tokens"],
        )
        return wavs[0], sr

    return synth


def make_verify_fn(asr_manager, conf_threshold: float):
    """构造 ASR 校验函数（复用 src.tts.verifier 的宽松匹配 + 置信度门槛）"""
    from src.tts.verifier import verify_audio

    def verify(wav, sr, word, language):
        return verify_audio(
            wav=wav, sr=sr, target_text=word, language=language,
            asr_manager=asr_manager, confidence_threshold=conf_threshold,
        )

    return verify


def generate_best(word, voice, language, instruct, synth_fn, verify_fn,
                  best_of: int = 3, seed_base: int = 1000,
                  conf_threshold: float = -1.0,
                  dur_ok=(_DUR_OK_MIN, _DUR_OK_MAX)):
    """
    生成 best-of-N 候选并择优。

    Args:
        synth_fn(text, voice, language, instruct, seed) -> (wav, sr)
        verify_fn(wav, sr, word, language) -> (verified, asr_text, avg_logprob)

    Returns:
        (best|None, failures)
        best: dict {wav, sr, seed, asr_text, avg_logprob, duration, score}
        failures: list[{seed, reason, ...}]
    """
    candidates = []
    failures = []
    for i in range(max(1, best_of)):
        seed = seed_base + i
        try:
            wav, sr = synth_fn(
                text=word, voice=voice, language=language,
                instruct=instruct, seed=seed,
            )
        except Exception as e:
            failures.append({"seed": seed, "reason": f"synth_error: {e}"})
            _logger.warning(f"[{word}] seed={seed} 生成异常: {e}")
            continue

        duration = len(wav) / sr
        if not (dur_ok[0] <= duration <= dur_ok[1]):
            failures.append({
                "seed": seed, "reason": "duration_out_of_range",
                "duration": round(duration, 3),
            })
            _logger.warning(f"[{word}] seed={seed} 时长异常 {duration:.2f}s")
            continue

        verified, asr_text, conf = verify_fn(wav, sr, word, language)
        if not verified:
            failures.append({
                "seed": seed, "reason": "asr_mismatch",
                "asr_text": asr_text, "avg_logprob": conf,
                "duration": round(duration, 3),
            })
            _logger.warning(
                f"[{word}] seed={seed} ASR 校验失败: {asr_text!r} conf={conf}"
            )
            continue
        if conf is not None and conf < conf_threshold:
            failures.append({
                "seed": seed, "reason": "low_confidence",
                "asr_text": asr_text, "avg_logprob": conf,
                "duration": round(duration, 3),
            })
            _logger.warning(f"[{word}] seed={seed} 置信度不足: {conf}")
            continue

        candidates.append({
            "wav": wav, "sr": sr, "seed": seed,
            "asr_text": asr_text, "avg_logprob": conf,
            "duration": duration,
            "score": score_candidate(conf, duration),
        })

    if not candidates:
        return None, failures

    best = max(candidates, key=lambda c: c["score"])
    _logger.success(
        f"[{word}] 择优完成: seed={best['seed']} score={best['score']} "
        f"conf={best['avg_logprob']} dur={best['duration']:.2f}s "
        f"(candidates={len(candidates)}, failures={len(failures)})"
    )
    return best, failures
