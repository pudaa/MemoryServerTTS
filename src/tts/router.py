"""TTS 模块路由 —— 文本转语音 API (WebSocket 由 server.py 管理)"""
import asyncio, os, re, threading, time, uuid, tempfile, json, urllib.parse
from pathlib import Path
import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

from src.common.logging import get_logger

_logger = get_logger("TTS")

router = APIRouter(prefix="/api/v1/tts", tags=["TTS"])

class TTSRequest(BaseModel):
    text: str
    voice: str = ""
    language: str = "English"
    instructions: str | None = None
    output_format: str = "wav"
    # ── 校验闭环参数 ──
    # verify: None=自动（短文本走 ASR 校验闭环，长文本仅轻量时长校验）
    #         True=强制 ASR 校验（仅对短文本生效），False=跳过 ASR 校验
    verify: bool | None = None
    seed: int | None = None       # 基础随机种子（短文本重试时 seed+i）
    include_meta: bool = False    # True 时返回 JSON（含 verified/attempts/asrText），否则返回 WAV 流

class TTSStreamRequest(BaseModel):
    text: str
    voice: str = "aiden"
    language: str = "English"
    instructions: str | None = None
    max_chunk_chars: int = 200
    verify: bool | None = None    # None/False=默认不校验（流式实时性优先），True=逐块校验

class TTSStreamPcmRequest(BaseModel):
    """流式 PCM 合成请求（对话朗读用，见 docs/STREAMING_ARCHITECTURE_ANALYSIS.md）"""
    text: str
    voice: str = "aiden"
    language: str = "English"
    instructions: str | None = None
    chunk_steps: int | None = None   # 覆盖配置；每片帧数（1 帧≈80ms 音频）
    max_new_tokens: int | None = None
    # 采样种子。缺省（null）时由服务端**按文本派生固定 seed**，
    # 保证同一文本每次点击生成出完全相同的音频（可复现，无需缓存）。
    seed: int | None = None

def _cleanup(path: str):
    try: os.remove(path)
    except OSError: pass

def _save_wav(wav_data, sr, prefix="tts"):
    """保存音频到临时文件并返回路径和文件名"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=f"{prefix}_")
    sf.write(tmp.name, wav_data, sr)
    return tmp.name

def _meta_headers(meta: dict) -> dict:
    """把校验元信息附加为响应头（保持 WAV 流响应兼容）"""
    return {
        "X-TTS-Verified": str(meta.get("verified", False)).lower(),
        "X-TTS-Attempts": str(meta.get("attempts", 0)),
        "X-TTS-Strategy": str(meta.get("strategy", "")),
        "X-TTS-Duration": f'{meta.get("duration", 0.0):.3f}',
        "X-TTS-Seed": str(meta.get("seed") if meta.get("seed") is not None else ""),
        "X-TTS-Asr-Text": urllib.parse.quote(meta.get("asr_text") or ""),
        "X-TTS-Asr-Confidence": (f'{meta["confidence"]:.3f}' if meta.get("confidence") is not None else ""),
    }

@router.post("/synthesize")
async def synthesize(request: Request, req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if req.output_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Only wav output is supported currently.")
    try:
        async with request.app.state.model_lock:
            wavs, sr, meta = request.app.state.model.generate(
                text=req.text, voice=req.voice, language=req.language,
                instructions=req.instructions, verify=req.verify, seed=req.seed,
            )
        wav = wavs[0]

        if req.include_meta:
            # JSON 模式：返回音频 URL + 校验元信息（适合听写客户端读取）
            audio_dir = Path("tts-audio")
            audio_dir.mkdir(parents=True, exist_ok=True)
            filename = f"tts_{uuid.uuid4().hex}.wav"
            filepath = audio_dir / filename
            sf.write(str(filepath), wav, sr)
            return JSONResponse({
                "audioUrl": f"/tts-audio/{filename}",
                "duration": round(meta.get("duration", len(wav) / sr), 3),
                "verified": meta.get("verified", False),
                "attempts": meta.get("attempts", 0),
                "strategy": meta.get("strategy"),
                "seed": meta.get("seed"),
                "asrText": meta.get("asr_text"),
                "asrConfidence": meta.get("confidence"),
            })

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tmp = tf.name
        sf.write(tmp, wav, sr)
        return FileResponse(
            tmp, media_type="audio/wav",
            filename=f"tts_{uuid.uuid4().hex}.wav",
            headers=_meta_headers(meta),
            background=BackgroundTask(_cleanup, tmp),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def synthesize_stream(request: Request, req: TTSStreamRequest):
    """
    流式 TTS：按句子逐块生成音频，通过 SSE 推送每个句子的音频 URL。
    
    SSE 事件格式：
    - event: chunk    data: {"index": 0, "text": "...", "audioUrl": "...", "duration": 2.5}
    - event: done     data: {"totalChunks": 5, "totalDuration": 12.3}
    - event: error    data: {"message": "..."}
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # 分句
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    buf = ""
    max_chars = req.max_chunk_chars or 200
    for s in sentences:
        if buf and len(buf) + len(s) > max_chars:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = buf + " " + s if buf else s
    if buf.strip():
        chunks.append(buf.strip())

    if not chunks:
        chunks = [text]

    async def event_stream():
        total_duration = 0.0
        audio_dir = Path("tts-audio")
        audio_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk_text in enumerate(chunks):
            try:
                # 极短文本补标点
                ct = chunk_text
                if len(ct) < 80 and not re.search(r'[.!?]$', ct):
                    ct = ct.rstrip(',;:') + '.'

                async with request.app.state.model_lock:
                    wavs, sr, meta = request.app.state.model.generate(
                        text=ct, voice=req.voice, language=req.language,
                        instructions=req.instructions if i == 0 else None,
                        # 流式默认不校验（实时性优先），客户端可显式开启
                        verify=req.verify if req.verify is not None else False,
                    )
                wav = wavs[0]
                duration = len(wav) / sr
                total_duration += duration

                # 保存到文件
                filename = f"tts_stream_{uuid.uuid4().hex}.wav"
                filepath = audio_dir / filename
                sf.write(str(filepath), wav, sr)

                # 通过 SSE 推送音频信息
                event_data = {
                    "index": i,
                    "text": chunk_text,
                    "audioUrl": f"/tts-audio/{filename}",
                    "duration": round(duration, 2),
                    "verified": meta.get("verified"),
                    "attempts": meta.get("attempts"),
                }
                yield f"event: chunk\ndata: {json.dumps(event_data)}\n\n"

            except Exception as e:
                error_data = {"index": i, "message": str(e)}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                return

        done_data = {"totalChunks": len(chunks), "totalDuration": round(total_duration, 2)}
        yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.post("/synthesize-stream")
async def synthesize_stream_pcm(request: Request, req: TTSStreamPcmRequest):
    """
    流式合成 —— 边生成边下发**裸 PCM**（int16LE / 单声道）。

    与 /synthesize 的区别：
    - /synthesize        ：整段生成完才返回，首字节要等全部生成（长文本 30s+）
    - /synthesize-stream ：首片约 330ms 即到达（实测与文本长度无关）

    响应：
    - Content-Type: audio/L16;rate=<sr>;channels=1   （原始有符号 16bit 小端 PCM）
    - X-Audio-Sample-Rate / X-Audio-Channels / X-Audio-Bits / X-Audio-Format
    - X-TTS-TTFA-Ms：服务端测得的首片耗时（便于定位链路瓶颈）

    注意：PCM **不含**头部与时长信息，客户端必须按响应头解释
    （采样率 24000、单声道、16bit、小端）；总时长只能靠字节数累计推断。

    实现要点：
    - 生成跑在**独立线程**里（模型是阻塞的），通过 `asyncio.Queue` 把分片
      送回事件循环，保证顺序且不阻塞其他请求；
    - 全程持有 `app.state.model_lock`（约束 P1：GPU 调用必须串行）；
    - 客户端断开时置取消标志，尽快停掉生成，避免空跑占用 GPU。
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    cfg = getattr(request.app.state, "tts_config", None)
    chunk_steps = req.chunk_steps or (cfg.stream_chunk_steps if cfg else 8)
    max_new = req.max_new_tokens or (cfg.stream_max_new_tokens if cfg else 4096)

    # 极短文本补标点（与 /stream 保持一致；不注入任何情绪指令，约束 C1）
    if len(text) < 80 and not re.search(r'[.!?]$', text):
        text = text.rstrip(',;:') + '.'

    app = request.app
    q: asyncio.Queue = asyncio.Queue(maxsize=8)   # 有界：给 GPU 侧反压，避免无界堆积
    state = {"cancel": False, "error": None}

    def _worker():
        """独立线程里跑阻塞式流式生成，把分片塞进队列（顺序由队列保证）。

        ⚠️ 这里**不能在满队列上无限阻塞**：客户端停止播放后就不再读取，
        队列（maxsize=8）很快填满，若此时 worker 阻塞在 put 上，它就永远看不到
        `state["cancel"]`，会一直卡到 join 超时——表现为"停止后再朗读要等好几秒"。
        因此用带超时的 put：超时即检查取消标志，取消则立刻退出（实测把
        "取消→释放"从 ~5s 降到 ~0.7s，见 bench/measure_cancel_latency.py）。
        """
        def enqueue(item) -> bool:
            """带超时入队；返回 False 表示应停止（取消/超时）。"""
            while True:
                if state["cancel"]:
                    return False
                try:
                    asyncio.run_coroutine_threadsafe(
                        q.put(item), app.state.stream_loop).result(timeout=0.2)
                    return True
                except TimeoutError:
                    continue
                except Exception:
                    return False

        try:
            for wav, sr in app.state.model.generate_stream(
                text=text, voice=req.voice, language=req.language,
                instructions=req.instructions,
                chunk_steps=chunk_steps, max_new_tokens=max_new,
                seed=req.seed,
            ):
                if state["cancel"]:
                    break
                pcm = np.asarray(wav, dtype=np.float32) * 32767.0
                pcm = pcm.clip(-32768, 32767).astype(np.int16)
                if not enqueue(("audio", pcm.tobytes(), sr)):
                    _logger.info("流式 PCM 客户端已断开，提前停止生成")
                    break
        except Exception as e:  # noqa: BLE001
            state["error"] = e
        finally:
            try:
                asyncio.run_coroutine_threadsafe(
                    q.put(("end", None, None)), app.state.stream_loop).result(timeout=2)
            except Exception:  # 事件循环已关闭 / 队列已满
                pass

    # ── 预取首片 ──
    # 目的：① 响应头能带上**真实采样率**；② 把"等首片"的时间准确计入 TTFA 日志。
    # 锁在「预取前获取」，在「生成结束时释放」——生成期间必须一直持锁（约束 P1）。
    await app.state.model_lock.acquire()
    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    t0 = time.perf_counter()
    try:
        first = await asyncio.wait_for(q.get(), timeout=60.0)
    except asyncio.TimeoutError:
        state["cancel"] = True
        th.join(timeout=5)
        app.state.model_lock.release()
        raise HTTPException(status_code=504, detail="TTS 流式生成首片超时（60s）")

    if state["error"] is not None:
        th.join(timeout=5)
        app.state.model_lock.release()
        raise HTTPException(status_code=500, detail=f"TTS 流式生成失败: {state['error']}")

    kind0, payload0, sr0 = first
    first_ms = (time.perf_counter() - t0) * 1000
    sr_hint = sr0 or 24000

    async def body():
        total = 0
        try:
            if kind0 == "audio" and payload0:
                _logger.info(
                    f"流式 PCM 首片: text_len={len(text)} ttfa={first_ms:.0f}ms sr={sr_hint}"
                )
                total += len(payload0)
                yield payload0
            while True:
                kind, payload, _ = await q.get()
                if kind == "end":
                    break
                total += len(payload)
                yield payload
        except asyncio.CancelledError:
            # 客户端断开：尽快停掉生成，避免空跑占 GPU
            state["cancel"] = True
            raise
        finally:
            state["cancel"] = True
            th.join(timeout=5)
            if state["error"] is not None:
                _logger.error(f"流式 PCM 生成失败: {state['error']}")
            else:
                _logger.success(
                    f"流式 PCM 完成: {total} bytes "
                    f"(~{total / 2 / sr_hint:.1f}s 音频), 首片 {first_ms:.0f}ms"
                )
            if app.state.model_lock.locked():
                try:
                    app.state.model_lock.release()
                except RuntimeError:
                    pass

    return StreamingResponse(
        body(),
        media_type=f"audio/L16;rate={sr_hint};channels=1",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Audio-Sample-Rate": str(sr_hint),
            "X-Audio-Channels": "1",
            "X-Audio-Bits": "16",
            "X-Audio-Format": "pcm_s16le",
            "X-TTS-TTFA-Ms": f"{first_ms:.0f}",
        },
    )

@router.get("/voices")
async def get_voices():
    # 与模型 README 官方音色清单一致（推荐使用音色母语生成最佳质量）
    voices = [
        {"id": "ryan", "name": "Ryan", "gender": "Male", "language": "English", "desc": "富有节奏感的动感男声", "type": "preset", "native": "English"},
        {"id": "aiden", "name": "Aiden", "gender": "Male", "language": "English", "desc": "阳光的美式男声，音色明亮", "type": "preset", "native": "English"},
        {"id": "vivian", "name": "Vivian", "gender": "Female", "language": "Chinese", "desc": "明亮、略带锋芒的年轻女性声音", "type": "preset", "native": "Chinese"},
        {"id": "serena", "name": "Serena", "gender": "Female", "language": "Chinese", "desc": "温暖、温柔的年轻女性声音", "type": "preset", "native": "Chinese"},
        {"id": "uncle_fu", "name": "Uncle_Fu", "gender": "Male", "language": "Chinese", "desc": "经验丰富的成熟男性嗓音", "type": "preset", "native": "Chinese"},
        {"id": "dylan", "name": "Dylan", "gender": "Male", "language": "Chinese", "desc": "年轻的北京男性嗓音", "type": "preset", "native": "Chinese (Beijing Dialect)"},
        {"id": "eric", "name": "Eric", "gender": "Male", "language": "Chinese", "desc": "活泼的成都男声", "type": "preset", "native": "Chinese (Sichuan Dialect)"},
        {"id": "ono_anna", "name": "Ono_Anna", "gender": "Female", "language": "Japanese", "desc": "活泼的日本女性声音", "type": "preset", "native": "Japanese"},
        {"id": "sohee", "name": "Sohee", "gender": "Female", "language": "Korean", "desc": "温暖的韩国女性声音", "type": "preset", "native": "Korean"},
    ]
    voice_dir = Path("voices")
    if voice_dir.exists():
        for f in sorted(voice_dir.glob("*.json")):
            voices.append({"id": f.stem, "name": f.stem, "type": "cloned"})
    return {"voices": voices}

@router.post("/clone")
async def clone_voice(reference_audio: UploadFile = File(...), transcript: str = Form(""), voice_name: str = Form("")):
    if not voice_name:
        raise HTTPException(status_code=400, detail="voice_name is required")
    voice_dir = Path("voices")
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice_id = f"cloned_{uuid.uuid4().hex[:8]}"
    (voice_dir / f"{voice_id}.json").write_text(str({"name": voice_name, "transcript": transcript}))
    return {"voice_id": voice_id, "message": "Voice cloned successfully (simulated)"}
