"""
词库 API（听写场景，Phase 2）

接口:
  GET  /api/v1/dictation/audio            听写模块取音频：缓存命中直接返回；
                                           miss 实时生成（best-of-N + ASR 校验）并回填缓存
  GET  /api/v1/dictation/words            管理员：缓存条目列表 + 统计
  POST /api/v1/dictation/words/{key}/regenerate  管理员：强制重新生成该词条
  POST /api/v1/dictation/feedback         管理员：标记 bad（触发后台重生成）
  POST /api/v1/dictation/pregenerate      管理员：批量预生成（后台任务，轮询进度）
  GET  /api/v1/dictation/tasks/{task_id}  任务进度轮询

原则:
  - instruct 完全由业务侧（听写模块）传入，服务端不注入任何情绪指令；
  - 缓存 key 含 gen_config_version（生成配方自动哈希），配置变更即自动换代；
  - 生成失败（所有候选未通过校验）时明确报错且不入缓存，宁缺毋滥。
"""
import asyncio
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.dictation import cache, generator
from src.dictation.spec import normalize_spec

router = APIRouter(prefix="/api/v1/dictation", tags=["Dictation"])

# 后台任务进度（内存态；服务重启丢失，可接受）
_TASKS: dict[str, dict] = {}

_WARNING_0B6 = "instruct is ignored on the 0.6B model; dictation pre-generation should use the 1.7B model"


class RegenerateRequest(BaseModel):
    best_of: int | None = None


class FeedbackRequest(BaseModel):
    key: str
    reason: str = "quality"   # emotion / noise / wrong_pronunciation / unclear / ...


class PregenItem(BaseModel):
    word: str
    voice: str = ""
    language: str = "English"
    instruct: str | None = None


class PregenRequest(BaseModel):
    words: list[PregenItem]
    best_of: int | None = None


def _entry_public(entry: dict) -> dict:
    """条目对外字段（不含 wav 数据）"""
    keys = ("key", "word", "voice", "language", "instruct", "gen_config_version",
            "seed", "attempts", "verified", "asr_text", "avg_logprob",
            "quality_score", "duration", "generated_at", "served_count",
            "bad_flags", "bad_reason", "bad_at")
    return {k: entry.get(k) for k in keys}


def _serve_wav_response(request: Request, cfg, key: str, entry: dict,
                        source: str, include_meta: bool):
    cache.bump_served(cfg, key)
    wav_path = cache._wav_path(cfg, key)
    if include_meta:
        # JSON 模式：audioUrl 指向本接口自身（命中路径），客户端可直接播放
        entry_q = entry
        url = (f"/api/v1/dictation/audio?word={entry_q.get('word')}"
               f"&voice={entry_q.get('voice')}&language={entry_q.get('language')}"
               f"&instruct={entry_q.get('instruct') or ''}")
        return JSONResponse({
            "audioUrl": url,
            "key": key,
            "source": source,
            "verified": entry.get("verified", False),
            "score": entry.get("quality_score"),
            "duration": entry.get("duration"),
            "bad": bool(entry.get("bad_flags")),
            "asrText": entry.get("asr_text"),
        })
    headers = {
        "X-Dict-Key": key,
        "X-Dict-Source": source,
        "X-Dict-Verified": str(entry.get("verified", False)).lower(),
        "X-Dict-Score": str(entry.get("quality_score", "")),
        "X-Dict-Duration": f'{entry.get("duration", 0.0):.3f}',
        "X-Dict-Bad": str(bool(entry.get("bad_flags"))).lower(),
        "X-Dict-Warning": str(request.state.dict_warning or ""),
    }
    return FileResponse(wav_path, media_type="audio/wav", headers=headers)


# ────────────────────────────────────────────────
# 听写模块取音频（缓存感知）
# ────────────────────────────────────────────────
@router.get("/audio")
async def get_word_audio(request: Request, word: str, voice: str = "",
                         language: str = "English", instruct: str | None = None,
                         include_meta: bool = False,
                         best_of: int | None = None):
    cfg = request.app.state.tts_config
    word = (word or "").strip()
    if not word:
        raise HTTPException(status_code=400, detail="word is required")

    word, voice, language, instruct = normalize_spec(cfg, word, voice, language, instruct)
    key = cache.cache_key(word, voice, language, instruct, cfg)

    # 命中缓存（且未被标记 bad）→ 直接返回
    entry = cache.lookup(cfg, key)
    if entry is not None and not entry.get("bad_flags"):
        request.state.dict_warning = None
        return _serve_wav_response(request, cfg, key, entry, "cache", include_meta)

    # 未命中 / 条目被标记 bad → 实时生成（best-of-N + 校验）并回填
    model = request.app.state.model
    async with request.app.state.model_lock:
        best, failures = generator.generate_best(
            word=word, voice=voice, language=language, instruct=instruct,
            synth_fn=generator.make_synth_fn(model.model, cfg.short_decode),
            verify_fn=generator.make_verify_fn(
                request.app.state.asr_model, cfg.dictation_conf_threshold),
            best_of=best_of or cfg.dictation_best_of,
            seed_base=cfg.dictation_seed_base,
            conf_threshold=cfg.dictation_conf_threshold,
        )

    if best is None:
        raise HTTPException(status_code=502, detail={
            "message": "generation failed: no candidate passed verification",
            "word": word, "failures": failures,
        })

    entry = cache.save(cfg, key, best["wav"], best["sr"], {
        "word": word, "voice": voice, "language": language, "instruct": instruct,
        "gen_config_version": cache.gen_config_version(cfg),
        "seed": best["seed"], "attempts": len(failures) + 1,
        "verified": True, "asr_text": best["asr_text"],
        "avg_logprob": best["avg_logprob"],
        "quality_score": best["score"], "duration": best["duration"],
    })
    # 0.6B 模型会静默丢弃 instruct，向调用方提示
    request.state.dict_warning = (
        _WARNING_0B6 if getattr(model.model, "tts_model_size", "") == "0b6" and instruct else None
    )
    return _serve_wav_response(request, cfg, key, entry, "generated", include_meta)


# ────────────────────────────────────────────────
# 管理端接口
# ────────────────────────────────────────────────
@router.get("/words")
async def list_words(request: Request):
    cfg = request.app.state.tts_config
    return {
        "summary": cache.summary(cfg),
        "entries": [_entry_public(e) for e in cache.list_entries(cfg)],
    }


@router.post("/words/{key}/regenerate")
async def regenerate_word(request: Request, key: str, req: RegenerateRequest):
    cfg = request.app.state.tts_config
    old = cache.lookup(cfg, key)
    if old is None:
        raise HTTPException(status_code=404, detail="cache entry not found")

    model = request.app.state.model
    async with request.app.state.model_lock:
        best, failures = generator.generate_best(
            word=old["word"], voice=old["voice"], language=old["language"],
            instruct=old.get("instruct"),
            synth_fn=generator.make_synth_fn(model.model, cfg.short_decode),
            verify_fn=generator.make_verify_fn(
                request.app.state.asr_model, cfg.dictation_conf_threshold),
            best_of=req.best_of or cfg.dictation_best_of,
            seed_base=cfg.dictation_seed_base,
            conf_threshold=cfg.dictation_conf_threshold,
        )
    if best is None:
        raise HTTPException(status_code=502, detail={
            "message": "regeneration failed: no candidate passed verification",
            "key": key, "failures": failures,
        })
    entry = cache.save(cfg, key, best["wav"], best["sr"], {
        "word": old["word"], "voice": old["voice"], "language": old["language"],
        "instruct": old.get("instruct"),
        "gen_config_version": cache.gen_config_version(cfg),
        "seed": best["seed"], "attempts": len(failures) + 1,
        "verified": True, "asr_text": best["asr_text"],
        "avg_logprob": best["avg_logprob"],
        "quality_score": best["score"], "duration": best["duration"],
    })
    return _entry_public(entry)


@router.post("/feedback")
async def feedback(request: Request, req: FeedbackRequest):
    """管理员标记坏条目 → 立即后台重生成（择优替换，自动清除 bad 标记）"""
    cfg = request.app.state.tts_config
    entry = cache.mark_bad(cfg, req.key, req.reason)
    if entry is None:
        raise HTTPException(status_code=404, detail="cache entry not found")

    task_id = _start_task(_regenerate_background, request.app, req.key)
    return {"ok": True, "key": req.key, "bad_flags": entry["bad_flags"],
            "taskId": task_id}


@router.post("/pregenerate")
async def pregenerate(request: Request, req: PregenRequest):
    """批量预生成（后台任务）。建议大批量走 CLI：python -m src.dictation.pregenerate"""
    if not req.words:
        raise HTTPException(status_code=400, detail="words is required")
    specs = [normalize_spec(request.app.state.tts_config,
                             w.word, w.voice, w.language, w.instruct)
             for w in req.words]
    task_id = _start_task(_pregenerate_background, request.app, specs,
                          req.best_of)
    return {"taskId": task_id, "total": len(specs)}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = _TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


# ────────────────────────────────────────────────
# 后台任务
# ────────────────────────────────────────────────
def _start_task(coro_factory, app, *args) -> str:
    task_id = uuid.uuid4().hex[:12]
    _TASKS[task_id] = {"status": "running", "total": 0, "done": 0,
                       "current": "", "results": {}}
    asyncio.create_task(coro_factory(task_id, app, *args))
    return task_id


async def _regenerate_background(task_id, app, key: str):
    try:
        cfg = app.state.tts_config
        old = cache.lookup(cfg, key)
        model = app.state.model
        _TASKS[task_id].update({"total": 1, "current": old["word"] if old else key})
        async with app.state.model_lock:
            best, failures = generator.generate_best(
                word=old["word"], voice=old["voice"], language=old["language"],
                instruct=old.get("instruct"),
                synth_fn=generator.make_synth_fn(model.model, cfg.short_decode),
                verify_fn=generator.make_verify_fn(
                    app.state.asr_model, cfg.dictation_conf_threshold),
                best_of=cfg.dictation_best_of,
                seed_base=cfg.dictation_seed_base,
                conf_threshold=cfg.dictation_conf_threshold,
            )
        if best is not None:
            cache.save(cfg, key, best["wav"], best["sr"], {
                "word": old["word"], "voice": old["voice"], "language": old["language"],
                "instruct": old.get("instruct"),
                "gen_config_version": cache.gen_config_version(cfg),
                "seed": best["seed"], "attempts": len(failures) + 1,
                "verified": True, "asr_text": best["asr_text"],
                "avg_logprob": best["avg_logprob"],
                "quality_score": best["score"], "duration": best["duration"],
            })
            _TASKS[task_id].update({"done": 1, "status": "done",
                                    "current": old["word"]})
        else:
            _TASKS[task_id].update({"done": 1, "status": "failed",
                                    "current": old["word"],
                                    "failures": failures})
    except Exception as e:
        _TASKS[task_id].update({"status": "error", "error": str(e)})


async def _pregenerate_background(task_id, app, specs, best_of):
    cfg = app.state.tts_config
    model = app.state.model
    total = len(specs)
    _TASKS[task_id]["total"] = total
    results = _TASKS[task_id]["results"]
    for i, (word, voice, language, instruct) in enumerate(specs):
        _TASKS[task_id]["current"] = word
        key = cache.cache_key(word, voice, language, instruct, cfg)
        try:
            async with app.state.model_lock:
                best, failures = generator.generate_best(
                    word=word, voice=voice, language=language, instruct=instruct,
                    synth_fn=generator.make_synth_fn(model.model, cfg.short_decode),
                    verify_fn=generator.make_verify_fn(
                        app.state.asr_model, cfg.dictation_conf_threshold),
                    best_of=best_of or cfg.dictation_best_of,
                    seed_base=cfg.dictation_seed_base,
                    conf_threshold=cfg.dictation_conf_threshold,
                )
            if best is not None:
                cache.save(cfg, key, best["wav"], best["sr"], {
                    "word": word, "voice": voice, "language": language,
                    "instruct": instruct,
                    "gen_config_version": cache.gen_config_version(cfg),
                    "seed": best["seed"], "attempts": len(failures) + 1,
                    "verified": True, "asr_text": best["asr_text"],
                    "avg_logprob": best["avg_logprob"],
                    "quality_score": best["score"], "duration": best["duration"],
                })
                results[word] = {"status": "ok", "score": best["score"]}
            else:
                results[word] = {"status": "failed",
                                 "failures": failures[:3]}
        except Exception as e:
            results[word] = {"status": "error", "error": str(e)}
        _TASKS[task_id]["done"] = i + 1
        if i % 5 == 4 or i == total - 1:
            await asyncio.sleep(0)   # 让出事件循环
    _TASKS[task_id]["status"] = "done"
