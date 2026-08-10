"""词库词条规格归一化（供 router 与 CLI 复用）"""
from src.tts.model_loader import _normalize_language, _pick_default_voice


def normalize_spec(cfg, word, voice, language, instruct):
    """
    归一化词条规格：语言别名（en→English）、按语言自动匹配母语音色、
    instruct 空串归一为 None。
    返回 (word, voice, language, instruct)。
    """
    language = _normalize_language(language) or cfg.default_language
    voice = voice or _pick_default_voice(language, cfg.default_voice)
    instruct = (instruct or "").strip() or None
    return word, voice, language, instruct
