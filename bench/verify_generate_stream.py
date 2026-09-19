"""隔离测试：model.generate_stream() 本身是否正确逐片产出

绕过 HTTP/队列，直接消费生成器，判断问题出在模型层还是 router 的线程桥接层。

用法（memory-tts-bp 环境）:
    python bench/verify_generate_stream.py --model ./models/qwen-0.6b
"""
import argparse
import os
import sys
import time

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba-cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
_BP = os.path.join(os.getcwd(), ".backport")
if os.path.isdir(_BP):
    sys.path.insert(0, _BP)
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen-0.6b")
    ap.add_argument("--voice", default="aiden")
    args = ap.parse_args()

    from src.tts.config import TTSConfig
    from src.tts.model_loader import TTSModelManager

    cfg = TTSConfig()
    print(f"=== load {args.model} ===", flush=True)
    m = TTSModelManager(config=cfg)

    for label, text in [("短句", "Hello there, my friend."),
                        ("长文本", "That's a really thoughtful question, and I'm glad you asked it. "
                                  "Let me walk through it step by step.")]:
        print(f"\n--- {label} ({len(text)} 字符) ---", flush=True)
        t0 = time.perf_counter()
        n = 0
        total = 0
        first = None
        sr = None
        try:
            for wav, sr in m.generate_stream(
                text=text, voice=args.voice, language="English",
            ):
                if first is None:
                    first = time.perf_counter() - t0
                n += 1
                total += len(wav)
                if n <= 3 or n % 20 == 0:
                    print(f"  片{n}: {len(wav)} samples ({len(wav)/sr:.2f}s音频) "
                          f"t={time.perf_counter()-t0:.2f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  !! 生成失败: {type(e).__name__}: {e}", flush=True)
            continue
        el = time.perf_counter() - t0
        dur = total / sr if sr else 0
        print(f"  完成: 片数={n} 总样本={total} 音频={dur:.2f}s 总耗时={el:.2f}s "
              f"首片={first*1000:.0f}ms RTF_wall={el/dur if dur else 0:.3f}x", flush=True)


if __name__ == "__main__":
    main()
