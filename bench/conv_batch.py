"""批处理可行性：把 N 条文本一次 generate，看单条成本是否下降。

与"并发 N 个请求"是两回事——约束 P1 要求所有 GPU 调用串行。
批处理是**一次前向算 N 条**，能在不违反锁的前提下提高吞吐。

用法（memory-tts-bp 环境）:
    python bench/conv_batch.py --model ./models/qwen-0.6b --batch 1,2,4
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

import torch  # noqa: E402

TEXTS = [
    "Sure, I can help you with that.",
    "That sounds like a good plan to me.",
    "Let me think about it for a moment.",
    "I think you should give it another try.",
    "Reading aloud every day really helps.",
    "Keep practicing a little bit each day.",
    "Would you like me to explain it again?",
    "That is a really thoughtful question.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen-0.6b")
    ap.add_argument("--batch", default="1,2,4")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    from faster_qwen3_tts import FasterQwen3TTS
    batches = [int(x) for x in args.batch.split(",") if x]

    print(f"=== load {args.model} ===", flush=True)
    m = FasterQwen3TTS.from_pretrained(
        args.model, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    m.warmup(prefill_len=100)

    print(f"\n{'batch':>6s} {'n_texts':>8s} {'wall':>9s} {'audio_tot':>10s} "
          f"{'per_text':>9s} {'RTF':>7s} {'VRAM':>8s}")
    print("-" * 70)
    for b in batches:
        texts = TEXTS[:b]
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            try:
                wavs, sr = m.generate_custom_voice(
                    texts, speaker="aiden", language="English", max_new_tokens=4096)
            except Exception as e:
                print(f"{b:6d}  FAILED: {type(e).__name__}: {str(e)[:70]}")
                break
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            n = len(wavs) if isinstance(wavs, list) else 1
            tot = sum(len(w) / sr for w in wavs) if isinstance(wavs, list) else len(wavs[0]) / sr
            vram = torch.cuda.max_memory_allocated() / 2**20
            print(f"{b:6d} {n:8d} {dt:8.2f}s {tot:9.2f}s {dt/n:8.2f}s "
                  f"{dt/tot:7.3f} {vram:7.0f}MB", flush=True)


if __name__ == "__main__":
    main()
