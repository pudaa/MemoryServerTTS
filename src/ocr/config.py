"""
OCR 配置管理 —— 统一管理 PP-OCRv6 模型档位、预处理、检测参数等
继承自 src.common.base_config.BaseConfig
"""
from pathlib import Path
from typing import Optional, Literal
from src.common.base_config import BaseConfig
from src.common.logging import get_logger

_logger = get_logger("OCR")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "ocr.yaml"

_MODEL_TIER_MAP = {
    "tiny": {"det": "PP-OCRv6_tiny_det", "rec": "PP-OCRv6_tiny_rec", "approx_size_mb": 5, "desc": "最小模型"},
    "small": {"det": "PP-OCRv6_small_det", "rec": "PP-OCRv6_small_rec", "approx_size_mb": 10, "desc": "平衡档"},
    "medium": {"det": "PP-OCRv6_medium_det", "rec": "PP-OCRv6_medium_rec", "approx_size_mb": 20, "desc": "最高精度"},
}

_PREPROCESS_PRESETS = {
    "essay": {"use_doc_orientation_classify": True, "use_doc_unwarping": True, "use_textline_orientation": False, "desc": "作文拍照: 矫正方向+透视扭曲"},
    "dictation": {"use_doc_orientation_classify": True, "use_doc_unwarping": True, "use_textline_orientation": False, "desc": "听写拍照: 矫正方向+透视扭曲"},
    "document": {"use_doc_orientation_classify": False, "use_doc_unwarping": False, "use_textline_orientation": False, "desc": "文档扫描: 无预处理"},
    "fast": {"use_doc_orientation_classify": False, "use_doc_unwarping": False, "use_textline_orientation": False, "desc": "快速模式: 无预处理"},
    "full": {"use_doc_orientation_classify": True, "use_doc_unwarping": True, "use_textline_orientation": True, "desc": "完整预处理: 全开"},
}

ModelTier = Literal["tiny", "small", "medium"]
PreprocessPreset = Literal["essay", "dictation", "document", "fast", "full"]
InferenceEngine = Literal["onnxruntime", "paddle_static", "transformers"]
InferenceDevice = Literal["gpu", "cpu"]
OcrLanguage = Literal["en", "ch", "Multilingual"]


class OCRConfig(BaseConfig):
    """OCR 配置管理器"""
    env_prefix = "OCRCONF_"
    _default_config_path = DEFAULT_CONFIG_PATH

    @property
    def engine(self) -> InferenceEngine:
        return self._env("engine") or self._get("ocr.engine", "onnxruntime")
    @property
    def device(self) -> InferenceDevice:
        return self._env("device") or self._get("ocr.device", "gpu")
    @property
    def lang(self) -> OcrLanguage:
        return self._env("lang") or self._get("ocr.lang", "en")
    @lang.setter
    def lang(self, value: OcrLanguage):
        self._set("ocr.lang", value)

    @property
    def model_tier(self) -> ModelTier:
        return self._env("model_tier") or self._get("ocr.model_tier", "small")
    @model_tier.setter
    def model_tier(self, value: ModelTier):
        if value not in _MODEL_TIER_MAP:
            raise ValueError(f"Invalid model_tier: {value}")
        self._set("ocr.model_tier", value)

    @property
    def detection_model_name(self) -> str:
        return self._get("ocr.text_detection_model_name", None) or _MODEL_TIER_MAP[self.model_tier]["det"]
    @property
    def recognition_model_name(self) -> str:
        return self._get("ocr.text_recognition_model_name", None) or _MODEL_TIER_MAP[self.model_tier]["rec"]

    @classmethod
    def available_tiers(cls) -> dict:
        return dict(_MODEL_TIER_MAP)

    @property
    def preprocess_preset(self) -> Optional[PreprocessPreset]:
        p = self._env("preprocess_preset") or self._get("ocr.preprocess_preset", None)
        return p if p != "null" else None
    @preprocess_preset.setter
    def preprocess_preset(self, value: Optional[PreprocessPreset]):
        if value is not None and value not in _PREPROCESS_PRESETS:
            raise ValueError(f"Invalid preprocess_preset: {value}")
        self._set("ocr.preprocess_preset", value or "null")

    @property
    def use_doc_orientation_classify(self) -> bool:
        p = self.preprocess_preset
        return _PREPROCESS_PRESETS[p]["use_doc_orientation_classify"] if (p and p in _PREPROCESS_PRESETS) else self._get("ocr.use_doc_orientation_classify", False)
    @property
    def use_doc_unwarping(self) -> bool:
        p = self.preprocess_preset
        return _PREPROCESS_PRESETS[p]["use_doc_unwarping"] if (p and p in _PREPROCESS_PRESETS) else self._get("ocr.use_doc_unwarping", False)
    @property
    def use_textline_orientation(self) -> bool:
        p = self.preprocess_preset
        return _PREPROCESS_PRESETS[p]["use_textline_orientation"] if (p and p in _PREPROCESS_PRESETS) else self._get("ocr.use_textline_orientation", False)
    @property
    def preprocess_desc(self) -> str:
        p = self.preprocess_preset
        return _PREPROCESS_PRESETS[p]["desc"] if (p and p in _PREPROCESS_PRESETS) else "手动配置"

    @classmethod
    def available_presets(cls) -> dict:
        return dict(_PREPROCESS_PRESETS)

    @property
    def det_limit_side_len(self) -> int: return int(self._get("ocr.detection.limit_side_len", 960))
    @property
    def det_thresh(self) -> float: return float(self._get("ocr.detection.thresh", 0.3))
    @property
    def det_box_thresh(self) -> float: return float(self._get("ocr.detection.box_thresh", 0.6))
    @property
    def det_unclip_ratio(self) -> float: return float(self._get("ocr.detection.unclip_ratio", 1.5))

    def to_paddleocr_kwargs(self) -> dict:
        return {
            "engine": self.engine,
            "text_detection_model_name": self.detection_model_name,
            "text_recognition_model_name": self.recognition_model_name,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
            "text_det_limit_side_len": self.det_limit_side_len,
            "text_det_thresh": self.det_thresh,
            "text_det_box_thresh": self.det_box_thresh,
            "text_det_unclip_ratio": self.det_unclip_ratio,
        }

    def summary(self) -> dict:
        return {
            "engine": self.engine, "device": self.device, "lang": self.lang,
            "model_tier": self.model_tier,
            "detection_model": self.detection_model_name,
            "recognition_model": self.recognition_model_name,
            "preprocess_preset": self.preprocess_preset, "preprocess_desc": self.preprocess_desc,
            "use_orientation": self.use_doc_orientation_classify,
            "use_unwarping": self.use_doc_unwarping,
            "det_limit_side_len": self.det_limit_side_len, "det_thresh": self.det_thresh,
        }

    def print_summary(self):
        s = self.summary()
        print("=" * 50)
        print("  OCR 配置")
        print("=" * 50)
        for l, v in [("引擎", f"{s['engine']}/{s['device']}/{s['lang']}"), ("模型档位", s["model_tier"]),
                      ("检测模型", s["detection_model"]), ("识别模型", s["recognition_model"]),
                      ("预处理", f"{s['preprocess_preset']} — {s['preprocess_desc']}"),
                      ("检测参数", f"limit={s['det_limit_side_len']} thresh={s['det_thresh']}")]:
            print(f"  {l}: {v}")
        print("=" * 50)
