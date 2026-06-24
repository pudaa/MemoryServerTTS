"""
OCR 配置管理模块 —— 统一管理 PP-OCRv6 模型档位、预处理、检测参数等

支持:
  - YAML 配置文件加载
  - 模型档位预设 (tiny/small/medium → 实际模型名称)
  - 预处理预设 (essay/dictation/document/fast/full)
  - 环境变量覆盖 (OCRCONF_ 前缀)
  - PaddleOCR 初始化参数生成

用法:
  from src.ocr_config import OCRConfig

  # 加载默认配置
  cfg = OCRConfig()

  # 切换模型档位
  cfg.model_tier = "tiny"

  # 生成 PaddleOCR 初始化参数
  kwargs = cfg.to_paddleocr_kwargs()
"""
import os
import logging
from pathlib import Path
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# ── 模型档位 → 实际模型名称映射 ──
_MODEL_TIER_MAP = {
    "tiny": {
        "det": "PP-OCRv6_tiny_det",
        "rec": "PP-OCRv6_tiny_rec",
        "approx_size_mb": 5,
        "desc": "最小模型，适合移动端/实时场景",
    },
    "small": {
        "det": "PP-OCRv6_small_det",
        "rec": "PP-OCRv6_small_rec",
        "approx_size_mb": 10,
        "desc": "平衡档，速度与精度折中",
    },
    "medium": {
        "det": "PP-OCRv6_medium_det",
        "rec": "PP-OCRv6_medium_rec",
        "approx_size_mb": 20,
        "desc": "最高精度，适合服务端批量处理",
    },
}

# ── 预处理预设 ──
_PREPROCESS_PRESETS = {
    "essay": {
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": True,
        "use_textline_orientation": False,
        "desc": "作文拍照: 矫正方向+透视扭曲",
    },
    "dictation": {
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": True,
        "use_textline_orientation": False,
        "desc": "听写拍照: 矫正方向+透视扭曲",
    },
    "document": {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "desc": "文档扫描: 无预处理 (图片已规整)",
    },
    "fast": {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "desc": "快速模式: 无预处理, 速度优先",
    },
    "full": {
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": True,
        "use_textline_orientation": True,
        "desc": "完整预处理: 全部开启, 精度优先",
    },
}

# 有效的值范围
ModelTier = Literal["tiny", "small", "medium"]
PreprocessPreset = Literal["essay", "dictation", "document", "fast", "full"]
InferenceEngine = Literal["onnxruntime", "paddle_static", "transformers"]
InferenceDevice = Literal["gpu", "cpu"]
OcrLanguage = Literal["en", "ch", "Multilingual"]

# 默认配置路径
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ocr_config.yaml"


class OCRConfig:
    """OCR 配置管理器"""

    def __init__(self, config_path: Optional[str | Path] = None):
        """
        Args:
            config_path: YAML 配置文件路径，默认 config/ocr_config.yaml
                         设为 None 则使用内置默认值
        """
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._data = {}
        self._loaded = False
        self._env_prefix = "OCRCONF_"

        if self._config_path.exists():
            self._load_yaml()
        else:
            logger.warning(f"配置文件不存在: {self._config_path}，使用内置默认值")
            self._loaded = True

    # ── YAML 加载 ──

    def _load_yaml(self):
        """从 YAML 文件加载配置"""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML 未安装，无法解析配置文件")
            self._loaded = True
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            self._loaded = True
            logger.info(f"OCR 配置已加载: {self._config_path}")
        except Exception as e:
            logger.error(f"配置文件解析失败: {e}")
            self._loaded = True

    # ── 环境变量覆盖 ──

    def _env(self, key: str, default=None):
        """读取环境变量覆盖值，格式: OCCONF_ENGINE, OCCONF_MODEL_TIER"""
        env_key = f"{self._env_prefix}{key.upper()}"
        return os.environ.get(env_key, default)

    # ── OCR 核心属性 ──

    @property
    def engine(self) -> InferenceEngine:
        return self._env("engine") or self._get("ocr.engine", "onnxruntime")

    @property
    def device(self) -> InferenceDevice:
        return self._env("device") or self._get("ocr.device", "gpu")

    @property
    def lang(self) -> OcrLanguage:
        return self._env("lang") or self._get("ocr.lang", "en")

    @property
    def ocr_version(self) -> str:
        return self._get("ocr.ocr_version", "PP-OCRv6")

    # ── 模型档位 ──

    @property
    def model_tier(self) -> ModelTier:
        return self._env("model_tier") or self._get("ocr.model_tier", "medium")

    @model_tier.setter
    def model_tier(self, value: ModelTier):
        if value not in _MODEL_TIER_MAP:
            raise ValueError(f"无效的 model_tier: {value}，可选: {list(_MODEL_TIER_MAP.keys())}")
        self._set("ocr.model_tier", value)

    @property
    def detection_model_name(self) -> str:
        """实际检测模型名称 (从 model_tier 解析)"""
        explicit = self._get("ocr.text_detection_model_name", None)
        if explicit:
            return explicit
        return _MODEL_TIER_MAP[self.model_tier]["det"]

    @property
    def recognition_model_name(self) -> str:
        """实际识别模型名称 (从 model_tier 解析)"""
        explicit = self._get("ocr.text_recognition_model_name", None)
        if explicit:
            return explicit
        return _MODEL_TIER_MAP[self.model_tier]["rec"]

    @property
    def model_tier_info(self) -> dict:
        """当前模型档位详情"""
        return {
            "tier": self.model_tier,
            "det_model": self.detection_model_name,
            "rec_model": self.recognition_model_name,
            **_MODEL_TIER_MAP[self.model_tier],
        }

    @classmethod
    def available_tiers(cls) -> dict:
        """所有可用的模型档位"""
        return dict(_MODEL_TIER_MAP)

    # ── 预处理 ──

    @property
    def preprocess_preset(self) -> Optional[PreprocessPreset]:
        preset = self._env("preprocess_preset") or self._get("ocr.preprocess_preset", None)
        return preset if preset != "null" else None

    @preprocess_preset.setter
    def preprocess_preset(self, value: Optional[PreprocessPreset]):
        if value is not None and value not in _PREPROCESS_PRESETS:
            raise ValueError(
                f"无效的 preprocess_preset: {value}，可选: {list(_PREPROCESS_PRESETS.keys())}"
            )
        self._set("ocr.preprocess_preset", value or "null")

    @property
    def use_doc_orientation_classify(self) -> bool:
        preset = self.preprocess_preset
        if preset and preset in _PREPROCESS_PRESETS:
            return _PREPROCESS_PRESETS[preset]["use_doc_orientation_classify"]
        return self._get("ocr.use_doc_orientation_classify", False)

    @property
    def use_doc_unwarping(self) -> bool:
        preset = self.preprocess_preset
        if preset and preset in _PREPROCESS_PRESETS:
            return _PREPROCESS_PRESETS[preset]["use_doc_unwarping"]
        return self._get("ocr.use_doc_unwarping", False)

    @property
    def use_textline_orientation(self) -> bool:
        preset = self.preprocess_preset
        if preset and preset in _PREPROCESS_PRESETS:
            return _PREPROCESS_PRESETS[preset]["use_textline_orientation"]
        return self._get("ocr.use_textline_orientation", False)

    @property
    def preprocess_desc(self) -> str:
        preset = self.preprocess_preset
        if preset and preset in _PREPROCESS_PRESETS:
            return _PREPROCESS_PRESETS[preset]["desc"]
        return "手动配置"

    @classmethod
    def available_presets(cls) -> dict:
        return dict(_PREPROCESS_PRESETS)

    # ── 检测参数 ──

    @property
    def det_limit_side_len(self) -> int:
        return int(self._get("ocr.detection.limit_side_len", 960))

    @property
    def det_thresh(self) -> float:
        return float(self._get("ocr.detection.thresh", 0.3))

    @property
    def det_box_thresh(self) -> float:
        return float(self._get("ocr.detection.box_thresh", 0.6))

    @property
    def det_unclip_ratio(self) -> float:
        return float(self._get("ocr.detection.unclip_ratio", 1.5))

    # ── 性能 ──

    @property
    def rec_batch_size(self) -> int:
        return int(self._get("ocr.performance.rec_batch_size", 6))

    @property
    def max_concurrency(self) -> int:
        return int(self._get("ocr.performance.max_concurrency", 1))

    # ── 配置导出 ──

    def to_paddleocr_kwargs(self) -> dict:
        """生成 PaddleOCR 初始化参数字典"""
        return {
            "lang": self.lang,
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
        """返回当前配置概览"""
        return {
            "config_path": str(self._config_path),
            "engine": self.engine,
            "device": self.device,
            "lang": self.lang,
            "model_tier": self.model_tier,
            "detection_model": self.detection_model_name,
            "recognition_model": self.recognition_model_name,
            "preprocess_preset": self.preprocess_preset,
            "preprocess_desc": self.preprocess_desc,
            "use_orientation": self.use_doc_orientation_classify,
            "use_unwarping": self.use_doc_unwarping,
            "use_textline_ori": self.use_textline_orientation,
            "det_limit_side_len": self.det_limit_side_len,
            "det_thresh": self.det_thresh,
            "det_box_thresh": self.det_box_thresh,
            "det_unclip_ratio": self.det_unclip_ratio,
            "rec_batch_size": self.rec_batch_size,
        }

    def print_summary(self):
        """打印当前配置概览"""
        s = self.summary()
        print("=" * 55)
        print("  OCR 配置概览")
        print("=" * 55)
        rows = [
            ("引擎", f"{s['engine']} / {s['device']} / {s['lang']}"),
            ("模型档位", s["model_tier"]),
            ("检测模型", s["detection_model"]),
            ("识别模型", s["recognition_model"]),
            ("预处理", f"{s['preprocess_preset']} — {s['preprocess_desc']}"),
            ("  方向矫正", s["use_orientation"]),
            ("  透视矫正", s["use_unwarping"]),
            ("  行翻转矫正", s["use_textline_ori"]),
            (
                "检测参数",
                f"limit={s['det_limit_side_len']} thresh={s['det_thresh']} "
                f"box={s['det_box_thresh']} unclip={s['det_unclip_ratio']}",
            ),
            ("识别批大小", s["rec_batch_size"]),
        ]
        for label, value in rows:
            print(f"  {label:14s}: {value}")
        print("=" * 55)

    # ── 内部辅助 ──

    def _get(self, dotted_key: str, default=None):
        """从配置字典获取值，支持点分隔路径如 'ocr.engine'"""
        keys = dotted_key.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict):
                node = node.get(k)
                if node is None:
                    return default
            else:
                return default
        return node

    def _set(self, dotted_key: str, value):
        """设置配置值到内存 (不写回文件)"""
        keys = dotted_key.split(".")
        node = self._data
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    def reload(self):
        """重新加载配置文件"""
        self._data = {}
        self._load_yaml()
