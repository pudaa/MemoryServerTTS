"""TTS 模块路由 —— 文本转语音 API (WebSocket 由 server.py 管理)"""
import os, uuid, tempfile
from pathlib import Path
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/tts", tags=["TTS"])

class TTSRequest(BaseModel):
    text: str
    voice: str = "Ono_Anna"
    language: str = "English"
    instructions: str | None = None
    output_format: str = "wav"

def _cleanup(path: str):
    try: os.remove(path)
    except OSError: pass

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
