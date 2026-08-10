"""
MemoryServerTTS API 主入口
模块化路由: tts / asr / pronunciation / ocr / dashboard
管理后台: http://localhost:8000/admin
"""
import asyncio, base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from src.tts.router import router as tts_router
from src.asr.router import router as asr_router
from src.pronunciation.router import router as pronunciation_router
from src.ocr.router import router as ocr_router
from src.dictation.router import router as dictation_router
from src.dashboard.router import router as dashboard_router

from src.tts.model_loader import TTSModelManager
from src.asr.model_loader import ASRModelManager
from src.pronunciation.evaluator import PronunciationEvaluator
from src.pronunciation.phoneme_evaluator import PhonemeEvaluator
from src.ocr.engine import OCREngine
from src.ocr.config import OCRConfig
from src.tts.config import TTSConfig
from src.common.logging import get_logger

_server_logger = get_logger("SERVER")

app = FastAPI(title="MemoryServerTTS API", version="1.0.0")
app.state.model_lock = asyncio.Lock()

app.include_router(tts_router)
app.include_router(asr_router)
app.include_router(pronunciation_router)
app.include_router(ocr_router)
app.include_router(dictation_router)
app.include_router(dashboard_router)

# 流式/合成输出音频的静态访问（/stream、include_meta 返回的 audioUrl 依赖此挂载）
app.mount("/tts-audio", StaticFiles(directory="tts-audio", check_dir=False), name="tts-audio")


@app.on_event("startup")
async def startup_event():
    app.state.tts_config = TTSConfig()
    app.state.model = TTSModelManager(config=app.state.tts_config)
    app.state.asr_model = ASRModelManager()
    app.state.pronunciation_evaluator = PronunciationEvaluator()
    app.state.phoneme_evaluator = PhonemeEvaluator(app.state.asr_model)
    app.state.ocr_config = OCRConfig()
    app.state.ocr_engine = OCREngine(config=app.state.ocr_config)
    app.state.ocr_engine.load()


# ── TTS WebSocket (保留在主文件) ──
@app.websocket("/api/v1/tts/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "end":
                await websocket.send_json({"type": "end_of_stream"}); break
            if data.get("type") != "text_chunk":
                await websocket.send_json({"type": "error", "message": "Unsupported message type"}); continue
            async with app.state.model_lock:
                wavs, sr, _meta = app.state.model.generate(
                    text=data["data"], voice=data.get("voice", "aiden"),
                    language=data.get("language", "English"),
                    instructions=data.get("instructions"), streaming=True,
                    verify=False,  # 实时流式不做 ASR 校验
                )
            wav = wavs[0]
            pcm = (wav * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
            await websocket.send_json({"type": "audio_chunk", "sample_rate": sr, "format": "pcm16", "data": base64.b64encode(pcm).decode()})
    except WebSocketDisconnect:
        _server_logger.info("WebSocket 客户端断开")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


# ── 健康检查 ──
@app.get("/api/v1/health")
async def health_check():
    try:
        import torch
        if torch.cuda.is_available():
            vu = torch.cuda.memory_allocated() / 1024**2
            vt = torch.cuda.get_device_properties(0).total_memory / 1024**2
        else:
            vu = vt = 0
        return {"status": "healthy", "model_loaded": True, "gpu_available": torch.cuda.is_available(), "vram_used_mb": vu, "vram_total_mb": vt}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
