"""TTS 模块路由 —— 文本转语音 API (WebSocket 由 server.py 管理)"""
import os, re, uuid, tempfile, json
from pathlib import Path
import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/tts", tags=["TTS"])

class TTSRequest(BaseModel):
    text: str
    voice: str = "Ono_Anna"
    language: str = "English"
    instructions: str | None = None
    output_format: str = "wav"

class TTSStreamRequest(BaseModel):
    text: str
    voice: str = "Ono_Anna"
    language: str = "English"
    instructions: str | None = None
    max_chunk_chars: int = 200

def _cleanup(path: str):
    try: os.remove(path)
    except OSError: pass

def _save_wav(wav_data, sr, prefix="tts"):
    """保存音频到临时文件并返回路径和文件名"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=f"{prefix}_")
    sf.write(tmp.name, wav_data, sr)
    return tmp.name

@router.post("/synthesize")
async def synthesize(request: Request, req: TTSRequest):
    if req.output_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Only wav output is supported currently.")
    try:
        async with request.app.state.model_lock:
            wavs, sr = request.app.state.model.generate(text=req.text, voice=req.voice, language=req.language, instructions=req.instructions)
        wav = wavs[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tmp = tf.name
        sf.write(tmp, wav, sr)
        return FileResponse(tmp, media_type="audio/wav", filename=f"tts_{uuid.uuid4().hex}.wav", background=BackgroundTask(_cleanup, tmp))
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
                    wavs, sr = request.app.state.model.generate(
                        text=ct, voice=req.voice, language=req.language,
                        instructions=req.instructions if i == 0 else None
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
                    "duration": round(duration, 2)
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
    voices = [
        {"id": "vivian", "name": "Vivian", "gender": "Female", "language": "Chinese", "desc": "明亮、略带锋芒的年轻女性声音", "type": "preset"},
        {"id": "serena", "name": "Serena", "gender": "Female", "language": "Chinese", "desc": "温暖、温柔的年轻女性声音", "type": "preset"},
        {"id": "uncle_fu", "name": "Uncle_Fu", "gender": "Male", "language": "Chinese", "desc": "经验丰富的男性嗓音", "type": "preset"},
        {"id": "dylan", "name": "Dylan", "gender": "Male", "language": "Chinese (Beijing Dialect)", "desc": "年轻的北京男性嗓音", "type": "preset"},
        {"id": "eric", "name": "Eric", "gender": "Male", "language": "Chinese (Sichuan Dialect)", "desc": "活泼的成都男声", "type": "preset"},
        {"id": "Ono_Anna", "name": "Ono_Anna", "gender": "Male", "language": "English", "desc": "充满活力的男性声音", "type": "preset"},
        {"id": "aiden", "name": "Aiden", "gender": "Male", "language": "English", "desc": "阳光的美国男声", "type": "preset"},
        {"id": "ono_anna", "name": "Ono_Anna", "gender": "Female", "language": "Japanese", "desc": "活泼的日本女性声音", "type": "preset"},
        {"id": "sohee", "name": "Sohee", "gender": "Female", "language": "Korean", "desc": "温暖的韩国女性声音", "type": "preset"},
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
