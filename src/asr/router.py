"""ASR 模块路由 —— 语音识别 API"""
import asyncio, os, tempfile
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

router = APIRouter(prefix="/api/v1/asr", tags=["ASR"])

def _cleanup(path: str):
    try: os.remove(path)
    except OSError: pass

@router.post("/transcribe")
async def transcribe_audio(request: Request, audio: UploadFile = File(...),
                           language: str | None = Form(None), task: str = Form("transcribe"),
                           beam_size: int = Form(5), word_timestamps: bool = Form(False)):
    if audio.filename is None:
        raise HTTPException(status_code=400, detail="audio is required")
    if not audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as tf:
        tf.write(await audio.read())
        tmp = tf.name
    normalized = None
    try:
        import soundfile as sf
        if os.path.getsize(tmp) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        data, sr = sf.read(tmp)
        normalized = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(normalized, data, sr, subtype="PCM_16")
        audio_to_use = normalized
    except HTTPException: raise
    except Exception: audio_to_use = tmp
    try:
        # ⚠️ 必须放到线程池执行：faster-whisper 的 transcribe 是**阻塞**调用，
        # 若直接 await（在事件循环里跑），整个 worker 的事件循环会被占住，
        # 同时进行的流式 TTS 分片就发不出去。
        # 实测：30s 音频转写 4.8s 期间，服务端事件循环卡顿达 4770ms，
        # 健康接口延迟 17ms → 3677ms，TTS 首字节 435ms → 4978ms。
        # 用 asyncio.to_thread 卸载后事件循环可继续发送分片（见 bench/measure_*）。
        result = await asyncio.to_thread(
            request.app.state.asr_model.transcribe,
            audio_path=audio_to_use, language=language, task=task,
            beam_size=beam_size, word_timestamps=word_timestamps,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup(tmp)
        if normalized: _cleanup(normalized)

@router.get("/models")
async def get_asr_models(request: Request):
    return {"models": request.app.state.asr_model.get_supported_models()}
