"""首片延迟验证：non_streaming_mode=True vs False 对 TTFA 的影响

动机：`generate_custom_voice_streaming` 的 streaming 是"帧级"的——
先对**整段文本**做 prefill，再逐帧吐音频。若整段 prefill 就决定了后续节奏，
那"按句切分"才有意义；若 nsm=False 能真正边读文本边出音频，则长文本首片会更快。

本脚本对同一段 198 词文本分别测两种模式的 TTFA / 首片音频长度 / 总时长。

用法（memory-tts-bp 环境，项目根目录）:
    python bench/stream_ttfa.py --model ./models/qwen-0.6b --repeat 2
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

LONG = (
    "That's a really thoughtful question, and I'm glad you asked it. Let me walk through it step by step. "
    "First, it helps to understand what you're actually trying to achieve, because the best method depends "
    "entirely on your goal. If your aim is fluency, then daily practice matters far more than long study "
    "sessions once a week. Second, try to make the practice enjoyable, since you'll stick with something "
    "you like much longer than something that feels like a chore. You could watch short videos, read simple "
    "articles, or talk with a partner about topics you genuinely care about. Third, don't be afraid of "
    "making mistakes, because mistakes show you exactly where your understanding is still incomplete. "
    "Every correction you receive is a small step forward, even when it feels uncomfortable in the moment. "
    "Finally, be patient with yourself. Real progress in a language happens slowly and then suddenly, "
    "so keep going even during the weeks when nothing seems to improve. If you can practice a little "
    "every single day, you will be surprised by how far you have come in just a few months. Would you "
    "like me to suggest a simple weekly schedule you could follow?"
)

# 第一句（用于"按句切分"对比）
FIRST = "That's a really thoughtful question, and I'm glad you asked it."


def run_stream(m, text, chunk_size, nsm):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    first_t = None
    first_audio = 0
    chunks = []
    for audio_chunk, sr, timing in m.generate_custom_voice_streaming(
        text=text, speaker="aiden", language="English",
        max_new_tokens=4096, chunk_size=chunk_size, non_streaming_mode=nsm,
    ):
        if first_t is None:
            first_t = time.perf_counter() - t0
            first_audio = len(audio_chunk) / sr
        chunks.append(audio_chunk)
    total = time.perf_counter() - t0
    torch.cuda.synchronize()
    full = np.concatenate(chunks)
    dur = len(full) / sr
    return first_t, first_audio, total, dur, len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen-0.6b")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=8)
    args = ap.parse_args()

    from faster_qwen3_tts import FasterQwen3TTS
    print(f"=== load {args.model} ===", flush=True)
    m = FasterQwen3TTS.from_pretrained(
        args.model, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    m.warmup(prefill_len=120)

    print(f"\nchunk_size={args.chunk_size}  (每片约 {args.chunk_size/12.5*1000:.0f}ms 音频)")
    print(f"{'case':34s} {'nsm':>5s} {'TTFA':>8s} {'首片音频':>9s} {'总耗时':>8s} "
          f"{'音频':>7s} {'片数':>5s}")
    print("-" * 88)

    cases = [
        ("198词 整段", LONG),
        ("仅第一句", FIRST),
    ]
    for label, text in cases:
        for nsm in (True, False):
            for i in range(args.repeat):
                try:
                    ft, fa, tot, dur, n = run_stream(m, text, args.chunk_size, nsm)
                    print(f"{label:34s} {str(nsm):>5s} {ft*1000:7.0f}ms {fa:8.2f}s "
                          f"{tot:7.2f}s {dur:6.2f}s {n:5d}", flush=True)
                except Exception as e:
                    print(f"{label:34s} {str(nsm):>5s}  FAILED: {type(e).__name__}: {str(e)[:50]}")
                    break

    print("\n=== 关键解读 ===")
    print("若 nsm=True 的 TTFA 远大于 nsm=False，说明整段 prefill 拖慢了首片，")
    print("=> 按句切分（Step 1）能显著降低首片延迟。")
    print("若两者接近，说明首片延迟由固定开销决定，按句切分收益有限。")


if __name__ == "__main__":
    main()
