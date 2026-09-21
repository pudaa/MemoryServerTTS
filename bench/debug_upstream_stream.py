"""定位 upstream 后端下 generate_stream 报错的具体原因（直接调，不经 HTTP）"""
import os
import sys
import traceback

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba-cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
sys.path.insert(0, os.getcwd())
_BP = os.path.join(os.getcwd(), ".backport")
if os.path.isdir(_BP):
    sys.path.insert(0, _BP)

os.environ["TTSCONF_BACKEND"] = "upstream"
os.environ["QWEN_TTS_MODEL_PATH"] = r".\models\qwen-0.6b"

from src.tts.config import TTSConfig      # noqa: E402
from src.tts.model_loader import TTSModelManager  # noqa: E402

cfg = TTSConfig()
print(f"cfg.backend = {cfg.backend}   (期望 upstream)")

m = TTSModelManager(config=cfg)
print(f"实际 backend = {m.backend}")
print(f"model 类型   = {type(m.model).__name__}")
print(f"有 .model ?  = {hasattr(m.model, 'model')}")
inner = getattr(m.model, "model", None)
print(f"inner 类型   = {type(inner).__name__}")
print(f"inner 有 generate_custom_voice ? {hasattr(inner, 'generate_custom_voice')}")

print("\n--- 调用 generate_stream（upstream 分支）---")
try:
    n = 0
    for wav, sr in m.generate_stream(text="Hello there, my friend.", voice="aiden",
                                     language="English"):
        n += 1
        print(f"  片{n}: {len(wav)} samples, sr={sr}")
        break
    print(f"OK，拿到 {n} 片")
except Exception:
    traceback.print_exc()
