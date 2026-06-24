"""OCR 模块路由 —— 图片/文档文字识别 API"""
import os, tempfile
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from src.ocr.engine import OCREngine, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_DOC_EXTENSIONS
from src.ocr.config import OCRConfig

router = APIRouter(prefix="/api/v1/ocr", tags=["OCR"])

def _cleanup(path: str):
    try: os.remove(path)
    except OSError: pass

@router.post("/scan")
async def ocr_scan_image(request: Request, image: UploadFile = File(...), language: str | None = Form(None)):
    if image.filename is None:
        raise HTTPException(status_code=400, detail="image file is required")
    ext = Path(image.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
        tf.write(await image.read())
        tmp = tf.name
    if os.path.getsize(tmp) == 0:
        _cleanup(tmp); raise HTTPException(status_code=400, detail="Uploaded image is empty")
    engine: OCREngine = request.app.state.ocr_engine
    cfg: OCRConfig = request.app.state.ocr_config
    if language and language != cfg.lang:
        cfg.lang = language; engine.reload()
    try:
        if not engine.ready: engine.load()
        result = engine.predict_image(tmp)
        if not result.get("success"): raise HTTPException(status_code=500, detail=result.get("error", "OCR failed"))
        return result
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: _cleanup(tmp)

@router.post("/scan-file")
async def ocr_scan_document(request: Request, file: UploadFile = File(...), language: str | None = Form(None)):
    if file.filename is None:
        raise HTTPException(status_code=400, detail="file is required")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_DOC_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported document format: {ext}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
        tf.write(await file.read())
        tmp = tf.name
    if os.path.getsize(tmp) == 0:
        _cleanup(tmp); raise HTTPException(status_code=400, detail="Uploaded file is empty")
    engine: OCREngine = request.app.state.ocr_engine
    cfg: OCRConfig = request.app.state.ocr_config
    if language and language != cfg.lang:
        cfg.lang = language; engine.reload()
    try:
        if not engine.ready: engine.load()
        result = engine.predict_pdf(tmp) if ext == ".pdf" else engine.predict_image(tmp)
        if not result.get("success"): raise HTTPException(status_code=500, detail=result.get("error", "OCR failed"))
        return result
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: _cleanup(tmp)

@router.get("/health")
async def ocr_health(request: Request):
    engine: OCREngine = request.app.state.ocr_engine
    cfg: OCRConfig = request.app.state.ocr_config
    return {
        "ocr_available": engine.ready, "device_actual": engine.actual_device,
        "config": cfg.summary(),
        "available_tiers": OCRConfig.available_tiers(),
        "available_presets": {k: v["desc"] for k, v in OCRConfig.available_presets().items()},
        "supported_image_formats": list(SUPPORTED_IMAGE_EXTENSIONS),
        "supported_doc_formats": list(SUPPORTED_DOC_EXTENSIONS),
    }
