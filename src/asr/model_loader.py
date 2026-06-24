import os
import torch
from faster_whisper import WhisperModel
from src.common.logging import get_logger

_logger = get_logger("ASR")


class ASRModelManager:
    """Faster Whisper模型管理器，支持音频转录"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """加载Faster Whisper模型"""
        model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        _logger.info(f"加载 Faster-Whisper: {model_size} on {device} (compute={compute_type})")

        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            _logger.success(f"模型加载成功: {model_size}")
        except Exception as e:
            raise RuntimeError(f"Faster Whisper模型加载失败: {e}")

    def transcribe(self, audio_path: str, language: str = "", 
                   task: str = "transcribe", beam_size: int = 5,
                   word_timestamps: bool = False):
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码 (如 'en', 'zh', 'ja')，None则自动检测
            task: 任务类型 ('transcribe' 或 'translate')
            beam_size: 束搜索大小
            word_timestamps: 是否返回单词级时间戳
            
        Returns:
            dict: 包含转录文本、语言、置信度等信息
        """
        try:
            # Faster-Whisper 不接受空字符串作为 language，转为 None 让其自动检测
            lang = language if language else None
            segments, info = self.model.transcribe(
                audio_path,
                language=lang,
                task=task,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
            )

            result = {
                "text": "",
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": [],
            }

            for segment in segments:
                segment_dict = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": segment.avg_logprob,
                }
                
                if word_timestamps and segment.words is not None:
                    segment_dict["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in segment.words
                    ]
                
                result["segments"].append(segment_dict)
                result["text"] += segment.text + " "

            result["text"] = result["text"].strip()
            return result

        except Exception as e:
            raise RuntimeError(f"音频转录失败: {e}")

    def get_supported_models(self):
        """获取支持的模型列表"""
        return [
            "tiny", "base", "small", "medium", "large-v1", 
            "large-v2", "large-v3", "distil-large-v2"
        ]