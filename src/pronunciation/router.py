"""Pronunciation 模块路由 —— 发音评价 API"""
import os, tempfile
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/pronunciation", tags=["Pronunciation"])

class PronunciationPair(BaseModel):
    student: str
    reference: str

class PhonemeScoreRequest(BaseModel):
    audio: str
    reference_text: str
    language: str | None = None

def _cleanup(path: str):
    try: os.remove(path)
    except OSError: pass

def _snake_to_camel(data):
    if isinstance(data, dict): return {_to_camel(k): _snake_to_camel(v) for k, v in data.items()}
    if isinstance(data, list): return [_snake_to_camel(item) for item in data]
    return data

def _to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])

@router.post("/score")
async def score_pronunciation(request: Request, student_audio: UploadFile = File(...), reference_audio: UploadFile = File(...)):
    if student_audio.filename is None or reference_audio.filename is None:
        raise HTTPException(status_code=400, detail="Both audio files required")
    if not student_audio.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported student audio format")
    if not reference_audio.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported reference audio format")
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(student_audio.filename).suffix) as tf:
        tf.write(await student_audio.read()); student_path = tf.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(reference_audio.filename).suffix) as tf:
        tf.write(await reference_audio.read()); ref_path = tf.name
    try:
        result = request.app.state.pronunciation_evaluator.pronunciation_score(student_audio=student_path, reference_audio=ref_path)
        return result
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: _cleanup(student_path); _cleanup(ref_path)

@router.post("/batch-score")
async def batch_score_pronunciation(request: Request, req: list[PronunciationPair]):
    pairs = [{"student": p.student, "reference": p.reference} for p in req]
    return {"results": request.app.state.pronunciation_evaluator.batch_pronunciation_score(pairs)}

@router.post("/phoneme-score")
async def phoneme_score(request: Request, student_audio: UploadFile = File(...), reference_text: str = Form(...), language: str | None = Form(None)):
    if student_audio.filename is None or not reference_text.strip():
        raise HTTPException(status_code=400, detail="student_audio and reference_text are required")
    if not student_audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(student_audio.filename).suffix) as tf:
        tf.write(await student_audio.read()); tmp = tf.name
    if os.path.getsize(tmp) == 0:
        _cleanup(tmp); raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    normalized = None
    try:
        import soundfile as sf
        data, sr = sf.read(tmp)
        duration = len(data) / max(sr, 1)
        if duration < 0.3: _cleanup(tmp); raise HTTPException(status_code=400, detail=f"Audio too short ({duration:.1f}s)")
        normalized = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(normalized, data, sr, subtype="PCM_16"); audio_to_use = normalized
    except HTTPException: raise
    except Exception as e: _cleanup(tmp); raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")
    try:
        result = request.app.state.phoneme_evaluator.evaluate(audio_path=audio_to_use, reference_text=reference_text, language=language)
        return _snake_to_camel(result)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup(tmp)
        if normalized:
            _cleanup(normalized)

@router.post("/phoneme-batch-score")
async def phoneme_batch_score(request: Request, req: list[PhonemeScoreRequest]):
    pairs = [{"audio": item.audio, "reference_text": item.reference_text} for item in req]
    return {"results": request.app.state.phoneme_evaluator.batch_evaluate(pairs)}

@router.post("/phoneme-score-with-text")
async def phoneme_score_with_text(request: Request, student_audio: UploadFile = File(...), reference_text: str = Form(...), language: str | None = Form(None)):
    return await phoneme_score(request=request, student_audio=student_audio, reference_text=reference_text, language=language)
