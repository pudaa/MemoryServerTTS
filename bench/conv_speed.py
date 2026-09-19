"""对话场景能力评估：200 词回复的实际延迟 + 1.7B vs 0.6B 对比 + GPU 利用率

针对真实场景：AI 对话回复 ≤200 词，需要评估
  - 到第一块音频的延迟（TTFA）——用户多快听到声音
  - 到全部音频的延迟（用户多快听完/字幕走完）
  - 生成期 GPU 利用率——判断还有多少余量

用法（memory-tts-bp 环境，从项目根目录运行）:
    python bench/conv_speed.py --model ./models/qwen-0.6b --repeat 2
    python bench/conv_speed.py --model ./models/qwen-1.7b --repeat 2
    # 扫描静态 KV 的 max_seq_len 看是否影响单步耗时
    python bench/conv_speed.py --model ./models/qwen-0.6b --seq-lens 2048,1024,768
"""
import argparse
import os
import sys
import threading
import time

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba-cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

_BP = os.path.join(os.getcwd(), ".backport")
if os.path.isdir(_BP):
    sys.path.insert(0, _BP)

import numpy as np  # noqa: E402
import torch  # noqa: E402

SPEAKER = "aiden"
LANGUAGE = "English"

# 对话场景的典型回复长度
CASES = [
    ("chat_short", "Sure, I can help you with that."),
    ("chat_mid", "That's a great question. Let me think about it for a moment. "
                 "I believe the best approach is to start small and build up gradually, "
                 "so you don't get overwhelmed by too many details at once."),
    ("chat_200w",
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
     "like me to suggest a simple weekly schedule you could follow?"),
]


class GPUWatcher(threading.Thread):
    """后台采样 nvidia-smi，统计生成期 GPU 利用率"""

    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        # 注意：不能命名成 _stop —— Thread 内部有同名方法，会被覆盖
        self._halt = threading.Event()
        import subprocess
        self._subprocess = subprocess

    def run(self):
        while not self._halt.is_set():
            try:
                out = self._subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                if out:
                    parts = [p.strip() for p in out.split(",")]
                    self.samples.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)

    def report(self):
        if not self.samples:
            return "no samples"
        u = [s[0] for s in self.samples]
        m = [s[1] for s in self.samples]
        p = [s[2] for s in self.samples]
        return (f"GPU util: mean={np.mean(u):5.1f}% max={np.max(u):5.1f}% | "
                f"VRAM: mean={np.mean(m):6.0f}MB max={np.max(m):6.0f}MB | "
                f"power: mean={np.mean(p):5.1f}W max={np.max(p):5.1f}W  (n={len(u)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen-0.6b")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--seq-lens", default="", help="逗号分隔，扫描 max_seq_len；空则用 2048")
    ap.add_argument("--outdir", default="bench/conv-out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    from faster_qwen3_tts import FasterQwen3TTS

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x] or [2048]

    for seq_len in seq_lens:
        print(f"\n{'='*100}\n=== {args.model}  max_seq_len={seq_len} ===", flush=True)
        t0 = time.perf_counter()
        m = FasterQwen3TTS.from_pretrained(
            args.model, device="cuda", dtype=torch.bfloat16,
            attn_implementation="sdpa", max_seq_len=seq_len)
        print(f"load={time.perf_counter()-t0:.1f}s", flush=True)
        m.warmup(prefill_len=120)
        print(f"VRAM after load+warmup: {torch.cuda.memory_allocated()/2**20:.0f}MB allocated", flush=True)

        for key, text in CASES:
            words = len(text.split())
            print(f"\n--- {key} ({words} words, {len(text)} chars) ---", flush=True)

            # 非流式：到全部音频的延迟
            for i in range(args.repeat):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                wavs, sr = m.generate_custom_voice(
                    text=text, speaker=SPEAKER, language=LANGUAGE, max_new_tokens=4096)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t1
                dur = len(wavs[0]) / sr
                print(f"  nonstream run{i+1}: total={dt:6.2f}s audio={dur:6.2f}s "
                      f"RTF_wall={dt/dur:.3f}x RTFx={dur/dt:.2f}x", flush=True)
                if i == 0:
                    import soundfile as sf
                    tag = os.path.basename(os.path.normpath(args.model))  # e.g. qwen-0.6b
                    sf.write(os.path.join(args.outdir, f"{key}_{tag}_s{seq_len}.wav"), wavs[0], sr)

            # 流式 + GPU 采样：TTFA 与利用率
            w = GPUWatcher()
            w.start()
            t1 = time.perf_counter()
            first = None
            chunks = []
            for audio_chunk, csr, timing in m.generate_custom_voice_streaming(
                text=text, speaker=SPEAKER, language=LANGUAGE,
                max_new_tokens=4096, chunk_size=args.chunk_size,
            ):
                if first is None:
                    first = time.perf_counter() - t1
                chunks.append(audio_chunk)
            total = time.perf_counter() - t1
            torch.cuda.synchronize()
            w.stop()
            full = np.concatenate(chunks)
            dur = len(full) / csr
            print(f"  streaming: TTFA={first*1000:6.0f}ms total={total:6.2f}s audio={dur:6.2f}s "
                  f"chunks={len(chunks)} RTF_wall={total/dur:.3f}x", flush=True)
            print(f"  {w.report()}", flush=True)
            if first is not None:
                print(f"  => 用户 {first*1000:.0f}ms 后听到第一声；"
                      f"{first+dur:.1f}s 后听完（音频 {dur:.1f}s）", flush=True)

        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
