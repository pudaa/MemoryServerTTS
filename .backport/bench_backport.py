"""在 transformers 4.57.3 上验证 backport 版 faster_qwen3_tts（CUDA Graph）

用法（memory-tts-bp 环境，从项目根目录运行）:
    python .backport/bench_backport.py --model ./models/qwen-1.7b
"""
import argparse
import os
import sys
import time

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba-cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

# 让本地 backport 副本优先于 site-packages
sys.path.insert(0, os.path.join(os.getcwd(), ".backport"))

import torch  # noqa: E402


def vram(tag):
    torch.cuda.synchronize()
    print(f"[VRAM:{tag}] allocated={torch.cuda.memory_allocated()/2**20:.0f}MB "
          f"peak={torch.cuda.max_memory_allocated()/2**20:.0f}MB "
          f"reserved={torch.cuda.memory_reserved()/2**20:.0f}MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen-1.7b")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--text", default="Hello. This is a speed benchmark test.")
    ap.add_argument("--speaker", default="aiden")
    ap.add_argument("--language", default="English")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--outdir", default="bench/out-bp")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    import faster_qwen3_tts
    print(f"faster_qwen3_tts from: {faster_qwen3_tts.__file__}", flush=True)
    print(f"version: {faster_qwen3_tts.__version__}", flush=True)
    import transformers
    print(f"transformers: {transformers.__version__}", flush=True)

    from faster_qwen3_tts import FasterQwen3TTS

    print(f"=== load {args.model} ===", flush=True)
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        args.model, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    print(f"load_seconds={time.perf_counter()-t0:.1f}", flush=True)
    vram("after_load")

    t0 = time.perf_counter()
    model.warmup(prefill_len=100)
    print(f"warmup_seconds={time.perf_counter()-t0:.1f}", flush=True)
    vram("after_warmup")

    print(f"\n--- A. non-streaming ---", flush=True)
    ts, ds = [], []
    for i in range(args.repeat):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=args.text, speaker=args.speaker, language=args.language,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        a = len(wavs[0]) / sr
        ts.append(dt)
        ds.append(a)
        print(f"  run{i+1}: total={dt:.3f}s audio={a:.2f}s RTF_wall={dt/a:.2f}x "
              f"RTFx={a/dt:.2f}x", flush=True)
        if i == 0:
            import soundfile as sf
            sf.write(os.path.join(args.outdir, "bp_nostream.wav"), wavs[0], sr)
    print(f"  mean: total={sum(ts)/len(ts):.3f}s audio={sum(ds)/len(ds):.2f}s "
          f"RTF_wall={(sum(ts)/len(ts))/(sum(ds)/len(ds)):.2f}x", flush=True)
    vram("after_nostream")

    print(f"\n--- B. streaming chunk_size={args.chunk_size} ---", flush=True)
    tt, tot = [], []
    for i in range(args.repeat):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        first = None
        chunks = []
        for audio_chunk, csr, timing in model.generate_custom_voice_streaming(
            text=args.text, speaker=args.speaker, language=args.language,
            chunk_size=args.chunk_size,
        ):
            if first is None:
                first = time.perf_counter() - t1
            chunks.append(audio_chunk)
        total = time.perf_counter() - t1
        torch.cuda.synchronize()
        import numpy as np
        full = np.concatenate(chunks)
        a = len(full) / csr
        tt.append(first)
        tot.append(total)
        print(f"  run{i+1}: TTFA={first*1000:.0f}ms total={total:.3f}s audio={a:.2f}s "
              f"chunks={len(chunks)} RTF_wall={total/a:.2f}x", flush=True)
        if i == 0:
            import soundfile as sf
            sf.write(os.path.join(args.outdir, "bp_stream.wav"), full, csr)
    print(f"  TTFA mean={sum(tt)/len(tt)*1000:.0f}ms  total mean={sum(tot)/len(tot):.3f}s",
          flush=True)

    print("\n--- C. single word ---", flush=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(
        text="Ahead.", speaker=args.speaker, language=args.language, max_new_tokens=256)
    torch.cuda.synchronize()
    print(f"  'Ahead.' total={time.perf_counter()-t1:.3f}s audio={len(wavs[0])/sr:.2f}s",
          flush=True)
    import soundfile as sf
    sf.write(os.path.join(args.outdir, "bp_word.wav"), wavs[0], sr)
    vram("final")


if __name__ == "__main__":
    main()
