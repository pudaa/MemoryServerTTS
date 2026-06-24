import os
import uuid
import base64
import tempfile
import asyncio
from pathlib import Path
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from src.model_loader import TTSModelManager
from src.asr_model_loader import ASRModelManager
from src.pronunciation_evaluator import PronunciationEvaluator
from src.phoneme_evaluator import PhonemeEvaluator
from src.ocr_engine import OCREngine, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_DOC_EXTENSIONS
from src.ocr_config import OCRConfig

import numpy as np

app = FastAPI(title="MemoryServerTTS API", version="1.0.0")
app.state.model_lock = asyncio.Lock()

# 启动时加载模型
@app.on_event("startup")
async def startup_event():
    app.state.model = TTSModelManager()
    app.state.asr_model = ASRModelManager()
    app.state.pronunciation_evaluator = PronunciationEvaluator()
    app.state.phoneme_evaluator = PhonemeEvaluator(app.state.asr_model)
    # OCR 引擎延迟加载，配置由 config/ocr_config.yaml 管理
    app.state.ocr_config = OCRConfig()
    app.state.ocr_engine = OCREngine(config=app.state.ocr_config)

# 请求模型
class TTSRequest(BaseModel):
    text: str
    voice: str = "Ono_Anna"
    language: str = "English"
    instructions: str | None = None
    output_format: str = "wav"

class ASRRequest(BaseModel):
    language: str | None = None
    task: str = "transcribe"
    beam_size: int = 5
    word_timestamps: bool = False

class PronunciationScoreRequest(BaseModel):
    reference_text: str | None = None

class PronunciationPair(BaseModel):
    student: str
    reference: str

class PhonemeScoreRequest(BaseModel):
    audio: str
    reference_text: str
    language: str | None = None


def _cleanup_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def _snake_to_camel(data):
    """递归将 dict 的 key 从 snake_case 转为 camelCase"""
    if isinstance(data, dict):
        return {_to_camel(k): _snake_to_camel(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_snake_to_camel(item) for item in data]
    return data


def _to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])

# REST API
@app.post("/api/v1/tts/synthesize") 
async def synthesize(request: TTSRequest):
    if request.output_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Only wav output is supported currently.")

    try:
        async with app.state.model_lock:
            model = app.state.model
            wavs, sr = model.generate(
                text=request.text,
                voice=request.voice,
                language=request.language,
                instructions=request.instructions,
            )

        wav = wavs[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
        sf.write(temp_path, wav, sr)
        return FileResponse(
            temp_path,
            media_type="audio/wav",
            filename=f"tts_{uuid.uuid4().hex}.wav",
            background=BackgroundTask(_cleanup_file, temp_path),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket 流式服务
@app.websocket("/api/v1/tts/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            if message_type == "end":
                await websocket.send_json({"type": "end_of_stream"})
                break
            if message_type != "text_chunk":
                await websocket.send_json({"type": "error", "message": "Unsupported message type"})
                continue

            async with app.state.model_lock:
                model = app.state.model
                wavs, sr = model.generate(
                    text=data["data"],
                    voice=data.get("voice", "Ono_Anna"),
                    language=data.get("language", "English"),
                    instructions=data.get("instructions"),
                    streaming=True,
                )

            wav = wavs[0]
            pcm = (wav * 32767.0).clip(-32768, 32767).astype(np.int16)
            pcm_bytes = pcm.tobytes()
            await websocket.send_json({
                "type": "audio_chunk",
                "sample_rate": sr,
                "format": "pcm16",
                "data": base64.b64encode(pcm_bytes).decode(),
            })
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


# 音色列表接口（官方支持的9个音色，详见Qwen3-TTS README）
@app.get("/api/v1/tts/voices")
async def get_voices():
    voices = [
        {"id": "vivian", "name": "Vivian", "gender": "Female", "language": "Chinese", "desc": "明亮、略带锋芒的年轻女性声音", "type": "preset"},
        {"id": "serena", "name": "Serena", "gender": "Female", "language": "Chinese", "desc": "温暖、温柔的年轻女性声音", "type": "preset"},
        {"id": "uncle_fu", "name": "Uncle_Fu", "gender": "Male", "language": "Chinese", "desc": "经验丰富的男性嗓音，音色低沉柔和", "type": "preset"},
        {"id": "dylan", "name": "Dylan", "gender": "Male", "language": "Chinese (Beijing Dialect)", "desc": "年轻的北京男性嗓音，音色清晰自然", "type": "preset"},
        {"id": "eric", "name": "Eric", "gender": "Male", "language": "Chinese (Sichuan Dialect)", "desc": "活泼的成都男声，带着一丝沙哑明亮", "type": "preset"},
        {"id": "Ono_Anna", "name": "Ono_Anna", "gender": "Male", "language": "English", "desc": "充满活力的男性声音，节奏感强劲", "type": "preset"},
        {"id": "aiden", "name": "Aiden", "gender": "Male", "language": "English", "desc": "阳光的美国男声，中音清晰", "type": "preset"},
        {"id": "ono_anna", "name": "Ono_Anna", "gender": "Female", "language": "Japanese", "desc": "活泼的日本女性声音，音色轻盈灵巧", "type": "preset"},
        {"id": "sohee", "name": "Sohee", "gender": "Female", "language": "Korean", "desc": "温暖的韩国女性声音，情感丰富", "type": "preset"},
    ]
    voice_dir = Path("voices")
    if voice_dir.exists():
        for file in sorted(voice_dir.glob("*.json")):
            voices.append({"id": file.stem, "name": file.stem, "type": "cloned"})
    return {"voices": voices}

# 音色克隆接口（简化版）
@app.post("/api/v1/tts/clone")
async def clone_voice(reference_audio: UploadFile = File(...), transcript: str = Form(""), voice_name: str = Form("")):
    if not voice_name:
        raise HTTPException(status_code=400, detail="voice_name is required")

    voice_dir = Path("voices")
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice_id = f"cloned_{uuid.uuid4().hex[:8]}"
    dest = voice_dir / f"{voice_id}.json"
    dest.write_text(str({"name": voice_name, "transcript": transcript}))
    return {"voice_id": voice_id, "message": "Voice cloned successfully (simulated)"}


# 健康检查接口
@app.get("/api/v1/health")
async def health_check():
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else 0
            vram_used = torch.cuda.memory_allocated() / 1024**2
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        else:
            gpu_util = 0
            vram_used = 0
            vram_total = 0

        return {
            "status": "healthy",
            "model_loaded": True,
            "gpu_available": gpu_available,
            "gpu_utilization_percent": gpu_util,
            "vram_used_mb": vram_used,
            "vram_total_mb": vram_total,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
    
    
# ASR 转录 接口
@app.post("/api/v1/asr/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), 
                          language: str | None = Form(None),
                          task: str = Form("transcribe"),
                          beam_size: int = Form(5),
                          word_timestamps: bool = Form(False)):
    if audio.filename == None:
        raise HTTPException(status_code=400, detail="audio is required")
    if not audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as temp_file:
        content = await audio.read()
        temp_file.write(content)
        temp_path = temp_file.name

    # 校验非空并标准化
    normalized_path = None
    try:
        import soundfile as sf
        if os.path.getsize(temp_path) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        data, sample_rate = sf.read(temp_path)
        normalized_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(normalized_path, data, sample_rate, subtype="PCM_16")
        audio_to_use = normalized_path
    except HTTPException:
        raise
    except Exception:
        audio_to_use = temp_path  # 回退到原始文件

    try:
        result = app.state.asr_model.transcribe(
            audio_path=audio_to_use,
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_file(temp_path)
        if normalized_path:
            _cleanup_file(normalized_path)

@app.post("/api/v1/pronunciation/score")
async def score_pronunciation(student_audio: UploadFile = File(...),
                             reference_audio: UploadFile = File(...)):
    if student_audio.filename == None or reference_audio.filename == None:
        raise HTTPException(status_code=400, detail="student_audio is required")
    if not student_audio.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported student audio format")
    if not reference_audio.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported reference audio format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(student_audio.filename).suffix) as student_temp:
        student_content = await student_audio.read()
        student_temp.write(student_content)
        student_path = student_temp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(reference_audio.filename).suffix) as ref_temp:
        ref_content = await reference_audio.read()
        ref_temp.write(ref_content)
        ref_path = ref_temp.name

    try:
        result = app.state.pronunciation_evaluator.pronunciation_score(
            student_audio=student_path,
            reference_audio=ref_path,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_file(student_path)
        _cleanup_file(ref_path)

@app.post("/api/v1/pronunciation/batch-score")
async def batch_score_pronunciation(request: list[PronunciationPair]):
    pairs = [{"student": p.student, "reference": p.reference} for p in request]
    results = app.state.pronunciation_evaluator.batch_pronunciation_score(pairs)
    return {"results": results}


# ─── 基于 G2P+ASR 音素对齐的发音评价（无需参考音频）───

@app.post("/api/v1/pronunciation/phoneme-score")
async def phoneme_score(
    student_audio: UploadFile = File(...),
    reference_text: str = Form(...),
    language: str | None = Form(None),
):
    """
    基于音素对齐的发音评价 —— 不需要标准参考音频，只需参考文本。
    内部流程：ASR转录 → G2P转音素 → 词级对齐 → 音素比对 → 综合评分。
    """
    if student_audio.filename is None or not reference_text.strip():
        raise HTTPException(status_code=400, detail="student_audio and reference_text are required")
    if not student_audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(student_audio.filename).suffix) as temp_file:
        content = await student_audio.read()
        temp_file.write(content)
        temp_path = temp_file.name

    # 校验文件非空
    if os.path.getsize(temp_path) == 0:
        _cleanup_file(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    print("language:", language)
    # 音频格式标准化：用 soundfile 重编码，确保解码器兼容
    normalized_path = None
    try:
        import soundfile as sf
        data, sample_rate = sf.read(temp_path)
        # 如果音频静音或极短
        duration = len(data) / max(sample_rate, 1)
        if duration < 0.3:
            _cleanup_file(temp_path)
            raise HTTPException(status_code=400, detail=f"Audio too short ({duration:.1f}s), minimum 0.3s required")

        # 重写为标准 16-bit PCM WAV
        normalized_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(normalized_path, data, sample_rate, subtype="PCM_16")
        audio_to_use = normalized_path
    except HTTPException:
        raise
    except Exception as e:
        _cleanup_file(temp_path)
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    try:
        result = app.state.phoneme_evaluator.evaluate(
            audio_path=audio_to_use,
            reference_text=reference_text,
            language=language,
        )
        return _snake_to_camel(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_file(temp_path)
        if normalized_path:
            _cleanup_file(normalized_path)


@app.post("/api/v1/pronunciation/phoneme-batch-score")
async def phoneme_batch_score(
    request: list[PhonemeScoreRequest],
):
    """
    批量音素评价 —— 每个评分项需提供音频路径和参考文本。
    音频需先上传到服务端可访问的路径。
    """
    pairs = [{"audio": item.audio, "reference_text": item.reference_text}
             for item in request]
    results = app.state.phoneme_evaluator.batch_evaluate(pairs)
    return {"results": results}


@app.post("/api/v1/pronunciation/phoneme-score-with-text")
async def phoneme_score_with_text(
    student_audio: UploadFile = File(...),
    reference_text: str = Form(...),
    language: str | None = Form(None),
):
    """
    与 phoneme-score 相同接口，使用 Query 参数传递参考文本（便于快速测试）。
    """
    return await phoneme_score(
        student_audio=student_audio,
        reference_text=reference_text,
        language=language,
    )

@app.get("/api/v1/asr/models")
async def get_asr_models():
    models = app.state.asr_model.get_supported_models()
    return {"models": models}


# ═══════════════════════════════════════════════════════════
# OCR 图片/文档文字识别
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/ocr/scan")
async def ocr_scan_image(
    image: UploadFile = File(...),
    language: str | None = Form(None),
):
    """
    扫描图片中的文字 —— 适用于作文/听写结果等场景的图片转文字。

    支持格式: PNG, JPG, JPEG, BMP, TIFF, WEBP

    返回提取的完整文本、逐行文本、置信度等信息。
    """
    if image.filename is None:
        raise HTTPException(status_code=400, detail="image file is required")

    ext = Path(image.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {ext}. Supported: {SUPPORTED_IMAGE_EXTENSIONS}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        content = await image.read()
        temp_file.write(content)
        temp_path = temp_file.name

    if os.path.getsize(temp_path) == 0:
        _cleanup_file(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    # 动态切换语言（如果有指定）
    engine: OCREngine = app.state.ocr_engine
    cfg: OCRConfig = app.state.ocr_config
    if language and language != cfg.lang:
        cfg.lang = language
        engine.reload()

    try:
        if not engine.ready:
            engine.load()
        result = engine.predict_image(temp_path)
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "OCR failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_file(temp_path)


@app.post("/api/v1/ocr/scan-file")
async def ocr_scan_document(
    file: UploadFile = File(...),
    language: str | None = Form(None),
):
    """
    扫描文档中的文字（PDF 等）—— 扩展支持。

    支持格式: PDF
    返回逐页文本及完整合并文本。
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="file is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document format: {ext}. Supported: {SUPPORTED_DOC_EXTENSIONS}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    if os.path.getsize(temp_path) == 0:
        _cleanup_file(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    engine: OCREngine = app.state.ocr_engine
    if language and language != engine.lang:
        engine = OCREngine(engine=engine.engine, lang=language, device=engine.device)
        engine.load()

    try:
        if not engine.ready:
            engine.load()

        if ext == ".pdf":
            result = engine.predict_pdf(temp_path)
        else:
            result = engine.predict_image(temp_path)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "OCR failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_file(temp_path)


@app.get("/api/v1/ocr/health")
async def ocr_health():
    """OCR 服务健康检查"""
    engine: OCREngine = app.state.ocr_engine
    cfg: OCRConfig = app.state.ocr_config
    return {
        "ocr_available": engine.ready,
        "device_actual": engine.actual_device,
        "config": cfg.summary(),
        "available_tiers": OCRConfig.available_tiers(),
        "available_presets": {
            k: v["desc"] for k, v in OCRConfig.available_presets().items()
        },
        "supported_image_formats": list(SUPPORTED_IMAGE_EXTENSIONS),
        "supported_doc_formats": list(SUPPORTED_DOC_EXTENSIONS),
    }


# 主程序入口，支持 python src/server.py 直接启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)