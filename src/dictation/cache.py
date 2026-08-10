"""
词库音频缓存 — 存储、查找、迭代（听写场景，Phase 2）

设计要点:
  - 缓存条目 = <key>.wav + <key>.json（元数据），同一目录原子写入。
  - cache key = hash(word | voice | language | instruct | gen_config_version)
    —— instruct 由业务侧（听写模块）传入，不同指令 = 不同风格音频，必须区分；
    —— gen_config_version 对"生成配方"（模型路径 + 短文本解码参数 + 校验参数 +
       缓存布局版本）做自动哈希，任何生成配置变更都会让旧条目自然 miss，
       实现"升级即换代"，无需手动清缓存。
  - 元数据携带 quality_score / bad_flags / served_count 等，支撑管理员的
    试听、标记 bad、重新生成等迭代操作。
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

# 缓存布局版本：修改本条目的存储结构/语义时 +1，强制整库失效
CACHE_LAYOUT_VERSION = 1

_lock = threading.Lock()

# 参与 gen_config_version 哈希的校验参数
_VERIFY_PARAM_KEYS = (
    "asr_confidence_threshold", "word_max_duration_s",
    "verify_max_retries", "dictation_conf_threshold",
)


def gen_config_version(cfg) -> str:
    """生成配方版本：模型路径 + 短文本解码参数 + 校验参数 + 布局版本 的稳定哈希"""
    payload = {
        "layout": CACHE_LAYOUT_VERSION,
        "model": getattr(cfg, "model_path", ""),
        "decode": getattr(cfg, "short_decode", {}),
        "verify": {k: getattr(cfg, k, None) for k in _VERIFY_PARAM_KEYS},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def cache_key(word: str, voice: str, language: str, instruct: str | None,
              cfg) -> str:
    """生成缓存 key（不含音色默认值解析，调用方需先归一化 voice/language）"""
    norm_word = re.sub(r"\s+", " ", (word or "").strip()).lower()
    payload = "|".join([
        norm_word,
        (voice or "").strip(),
        (language or "").strip(),
        (instruct or "").strip(),
        gen_config_version(cfg),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def cache_dir(cfg) -> Path:
    return Path(getattr(cfg, "dictation_cache_dir", "word-cache"))


def ensure_dir(cfg) -> Path:
    d = cache_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _entry_path(cfg, key: str) -> Path:
    return ensure_dir(cfg) / f"{key}.json"


def _wav_path(cfg, key: str) -> Path:
    return ensure_dir(cfg) / f"{key}.wav"


def lookup(cfg, key: str) -> dict | None:
    """按 key 读取条目；json 或 wav 缺失返回 None"""
    ep = _entry_path(cfg, key)
    if not ep.exists():
        return None
    try:
        entry = json.loads(ep.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not _wav_path(cfg, key).exists():
        return None
    return entry


def save(cfg, key: str, wav, sr: int, meta: dict) -> dict:
    """
    原子写入：先写 .tmp 再 os.replace。
    meta 需包含 word/voice/language/instruct/seed/verified/asr_text/avg_logprob/
    quality_score/duration 等；generated_at/served_count/bad_flags 自动补全。
    """
    d = ensure_dir(cfg)
    entry = dict(meta)
    entry.update({
        "key": key,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "served_count": 0,
        "bad_flags": [],
        "bad_reason": None,
        "bad_at": None,
    })
    tmp_wav = d / f"{key}.wav.tmp"
    tmp_json = d / f"{key}.json.tmp"
    import soundfile as sf
    sf.write(str(tmp_wav), wav, sr, format="WAV")
    tmp_json.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp_wav, _wav_path(cfg, key))
    os.replace(tmp_json, _entry_path(cfg, key))
    return entry


def mark_bad(cfg, key: str, reason: str) -> dict | None:
    """管理员标记坏条目；返回更新后的条目（不存在则 None）"""
    entry = lookup(cfg, key)
    if entry is None:
        return None
    with _lock:
        flags = entry.get("bad_flags") or []
        if reason and reason not in flags:
            flags.append(reason)
        entry["bad_flags"] = flags
        entry["bad_reason"] = reason
        entry["bad_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _entry_path(cfg, key).write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return entry


def bump_served(cfg, key: str):
    """命中次数 +1（尽力而为，失败不影响主流程）"""
    entry = lookup(cfg, key)
    if entry is None:
        return
    with _lock:
        entry["served_count"] = int(entry.get("served_count", 0)) + 1
        try:
            _entry_path(cfg, key).write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass


def list_entries(cfg) -> list[dict]:
    """列出全部缓存条目（按 word 排序）"""
    d = cache_dir(cfg)
    if not d.exists():
        return []
    entries = []
    for jf in sorted(d.glob("*.json")):
        try:
            entry = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (d / f"{jf.stem}.wav").exists():
            entries.append(entry)
    return sorted(entries, key=lambda e: str(e.get("word", "")))


def delete(cfg, key: str) -> bool:
    """删除条目（wav + json），管理员整词清除用"""
    removed = False
    for p in (_wav_path(cfg, key), _entry_path(cfg, key)):
        try:
            if p.exists():
                p.unlink()
                removed = True
        except OSError:
            pass
    return removed


def summary(cfg) -> dict:
    entries = list_entries(cfg)
    return {
        "total": len(entries),
        "verified": sum(1 for e in entries if e.get("verified")),
        "bad": sum(1 for e in entries if e.get("bad_flags")),
        "unverified": sum(1 for e in entries if not e.get("verified")),
    }
