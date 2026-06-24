"""
OCR 引擎模块 —— 基于 PaddleOCR (PP-OCRv6) 的图片/文档文字提取
统一由 OCRConfig 管理配置 (模型档位、预处理、检测参数等)。
"""
import os, time
from pathlib import Path
from typing import Optional
from src.common.logging import get_logger
from src.ocr.config import OCRConfig

_logger = get_logger("OCR")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_DOC_EXTENSIONS = {".pdf"}


class OCREngine:
    """PaddleOCR 封装引擎，配置由 OCRConfig 统一管理"""

    def __init__(self, config: "OCRConfig | None" = None):
        """
        Args:
            config: OCRConfig 实例。为 None 时自动加载默认配置文件。
        """
        from src.ocr.config import OCRConfig

        self._config = config if config is not None else OCRConfig()
        self._ocr = None
        self._ready = False
        self._actual_device = None
        self._init_kwargs = {}

    # ── 只读属性 ──

    @property
    def config(self) -> "OCRConfig":
        """返回当前使用的配置对象 (可运行时修改)"""
        return self._config

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def actual_device(self) -> Optional[str]:
        """返回实际使用的推理设备 (gpu/cpu)"""
        return self._actual_device

    # ── 模型加载 ──

    def load(self):
        """加载 OCR 模型（首次自动下载模型，支持 GPU→CPU 自动降级）"""
        if self._ready:
            return

        # 抑制 ONNX Runtime 图优化警告
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="paddleocr")
        try:
            from onnxruntime import set_default_logger_severity
            set_default_logger_severity(3)  # 3=ERROR, 仅显示错误
        except Exception:
            pass

        self._init_kwargs = self._config.to_paddleocr_kwargs()

        _logger.info(
            f"加载 PaddleOCR: engine={self._config.engine}, "
            f"tier={self._config.model_tier} "
            f"({self._config.detection_model_name} / {self._config.recognition_model_name}), "
            f"preset={self._config.preprocess_preset}, "
            f"device={self._config.device}"
        )
        try:
            from paddleocr import PaddleOCR

            try:
                self._ocr = PaddleOCR(**self._init_kwargs)
                self._actual_device = self._config.device
                self._ready = True
                _logger.info(
                    f"PaddleOCR 就绪: device={self._actual_device}, "
                    f"tier={self._config.model_tier}, "
                    f"{self._config.preprocess_desc}"
                )
            except RuntimeError as e:
                error_msg = str(e)
                if self._config.device == "gpu" and any(
                    kw in error_msg for kw in ("GPU", "CUDA", "provider")
                ):
                    _logger.warning(f"GPU 不可用，降级到 CPU: {error_msg[:150]}")
                    cpu_kwargs = {**self._init_kwargs, "device": "cpu"}
                    self._ocr = PaddleOCR(**cpu_kwargs)
                    self._actual_device = "cpu"
                    self._ready = True
                    _logger.info("PaddleOCR 就绪: device=cpu (降级模式)")
                    return
                else:
                    raise
        except ImportError as e:
            _logger.error(f"PaddleOCR 未安装: {e}")
            raise
        except Exception as e:
            _logger.error(f"PaddleOCR 加载失败: {e}")
            raise

    def reload(self):
        """强制重新加载模型（配置变更后调用）"""
        self.unload()
        self.load()

    # ── 图片识别 ──

    def predict_image(self, image_path: str) -> dict:
        """
        识别单张图片中的文字

        Returns:
            {
                "success": bool,
                "text": str,           # 完整文本
                "lines": list[str],    # 每行文本
                "boxes": list,         # 文本框坐标
                "confidences": list,   # 每行置信度
                "avg_confidence": float,
                "language": str,
                "processing_time_ms": float,
                "preprocess": dict,    # 本次执行的预处理
                "model_tier": str,     # 使用的模型档位
            }
        """
        if not self._ready:
            self.load()

        if not os.path.exists(image_path):
            return {"success": False, "error": f"文件不存在: {image_path}"}

        ext = Path(image_path).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            return {
                "success": False,
                "error": f"不支持的图片格式: {ext}，支持: {SUPPORTED_IMAGE_EXTENSIONS}",
            }

        t_start = time.perf_counter()
        try:
            result = self._ocr.predict(image_path)
            elapsed_ms = (time.perf_counter() - t_start) * 1000

            lines, boxes, confidences = [], [], []
            preprocess_applied = {}

            for res in result:
                rec_texts = res.get("rec_texts", [])
                rec_boxes = res.get("rec_boxes", [])
                rec_scores = res.get("rec_scores", [])

                ms = res.get("model_settings", {})
                if "use_doc_orientation_classify" in ms:
                    preprocess_applied["orientation_correction"] = ms["use_doc_orientation_classify"]
                if "use_doc_unwarping" in ms:
                    preprocess_applied["unwarping"] = ms["use_doc_unwarping"]
                if "angle" in res:
                    preprocess_applied["detected_angle"] = res["angle"]

                for i, text in enumerate(rec_texts):
                    text = text.strip() if isinstance(text, str) else str(text)
                    if text:
                        lines.append(text)
                        if i < len(rec_boxes):
                            box = rec_boxes[i]
                            boxes.append(box.tolist() if hasattr(box, 'tolist') else list(box))
                        if rec_scores and i < len(rec_scores):
                            confidences.append(float(rec_scores[i]))

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                "success": True,
                "text": "\n".join(lines),
                "lines": lines,
                "boxes": boxes,
                "confidences": confidences,
                "avg_confidence": round(avg_conf, 4),
                "language": self._config.lang,
                "processing_time_ms": round(elapsed_ms, 2),
                "preprocess": preprocess_applied,
                "model_tier": self._config.model_tier,
            }

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            _logger.error(f"OCR 识别失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": round(elapsed_ms, 2),
            }

    # ── PDF 识别 ──

    def predict_pdf(self, pdf_path: str, page_range: Optional[tuple] = None) -> dict:
        """识别 PDF 文件中的文字（逐页处理）"""
        if not self._ready:
            self.load()

        if not os.path.exists(pdf_path):
            return {"success": False, "error": f"文件不存在: {pdf_path}"}

        ext = Path(pdf_path).suffix.lower()
        if ext != ".pdf":
            return {"success": False, "error": f"不支持的文档格式: {ext}"}

        t_start = time.perf_counter()
        try:
            result = self._ocr.predict(pdf_path)
            elapsed_ms = (time.perf_counter() - t_start) * 1000

            pages, all_lines = [], []
            results_list = list(result) if hasattr(result, "__iter__") else [result]

            for page_idx, res in enumerate(results_list):
                rec_texts = res.get("rec_texts", []) if isinstance(res, dict) else []
                page_lines = [
                    t.strip() for t in rec_texts
                    if isinstance(t, str) and t.strip()
                ]
                pages.append({
                    "page": page_idx + 1,
                    "text": "\n".join(page_lines),
                    "lines": page_lines,
                })
                all_lines.extend(page_lines)

            return {
                "success": True,
                "total_pages": len(pages),
                "pages": pages,
                "full_text": "\n".join(all_lines),
                "processing_time_ms": round(elapsed_ms, 2),
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            _logger.error(f"PDF OCR 失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": round(elapsed_ms, 2),
            }

    # ── 资源释放 ──

    def unload(self):
        """释放 OCR 模型资源"""
        if self._ocr is not None:
            del self._ocr
            self._ocr = None
        self._ready = False
        _logger.info("PaddleOCR 引擎已卸载")


