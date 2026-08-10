"""TTS 模块路由 —— 文本转语音 API (WebSocket 由 server.py 管理)"""
import os, re, uuid, tempfile, json, urllib.parse
from pathlib import Path
import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

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
