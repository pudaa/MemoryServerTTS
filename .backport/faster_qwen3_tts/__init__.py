"""
faster-qwen3-tts（backport 到 transformers 4.57.3）

来源: https://github.com/andimarafioti/faster-qwen3-tts  v0.4.0 (MIT)
本目录是**打了 2 处兼容补丁**的本地副本，使其能在 projects 现有的
transformers 4.57.3 + qwen-tts 0.1.1 + torch 2.7.0+cu128 组合上运行，
从而避免把 transformers 升到 5.x（那会牵动 ASR / OCR / 发音模块）。

补丁点（详见各文件内 "backport" 注释）:
  1. StaticLayer.lazy_initialization(dummy_k, dummy_k)
     → transformers 4.x 只接受 1 个参数，补 try/except 兼容两种签名
  2. DynamicCache.layers[li].keys/.values
     → transformers 4.57 用 __getitem__ 返回 (k, v)，补双分支兼容

GGML 后端未声明依赖，故此处不导出（保留文件，按需显式导入）。
"""
from .model import FasterQwen3TTS

__version__ = "0.4.0+backport1"
__all__ = ["FasterQwen3TTS"]
