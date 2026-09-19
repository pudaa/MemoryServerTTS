"""faster-qwen3-tts (CUDA Graph) 对比基准

与 bench/bench_tts.py（上游动态KV基线）在**同一环境、同一文本、同一模型**下对比。

注意：本脚本必须在 memory-tts-perf 环境中运行（transformers 5.x + qwen-tts-hf）。

用法:
    python bench/bench_faster.py --model ./models/qwen-1.7b --repeat 3
"""
import argparse
import os
import time

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba-cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import torch  # noqa: E402


def vram(tag):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 2**20
    peak = torch.cuda.max_memory_allocated() / 2**20
    resv = torch.cuda.memory_reserved() / 2**20
    print(f"[VRAM:{tag}] allocated={alloc:.0f}MB peak={peak:.0f}MB reserved={resv:.0f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen-1.7b")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--text", default="Hello. This is a speed benchmark test.")
    ap.add_argument("--speaker", default="aiden")
    ap.add_argument("--language", default="English")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--outdir", default="bench/out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"=== load {args.model} (CUDA graph backend) ===", flush=True)
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        args.model, device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=args.max_seq_len,
    )
    t_load = time.perf_counter() - t0
    print(f"load_seconds={t_load:.1f}", flush=True)
    vram("after_load")

    # 显式预热：触发 CUDA Graph 捕获
    t0 = time.perf_counter()
    try:
        model.warmup(prefill_len=100)
        print(f"warmup_seconds={time.perf_counter()-t0:.1f}", flush=True)
    except Exception as e:
        print(f"warmup failed: {type(e).__name__}: {e}", flush=True)
    vram("after_warmup")

    def stats(ts):
        return f"min={min(ts):.3f}s mean={sum(ts)/len(ts):.3f}s max={max(ts):.3f}s"

    # ---------- A. 非流式 ----------
    print(f"\n--- A. non-streaming  text={args.text!r} ---", flush=True)
    times, durs = [], []
    for i in range(args.repeat):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=args.text, speaker=args.speaker, language=args.language,
            max_new_tokens=args.max_new,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        audio_s = len(wavs[0]) / sr
        times.append(dt)
        durs.append(audio_s)
        print(f"  run{i+1}: total={dt:.3f}s audio={audio_s:.2f}s "
              f"RTF_wall={dt/audio_s:.2f}x RTFx={audio_s/dt:.2f}x", flush=True)
        if i == 0:
            import soundfile as sf
            sf.write(os.path.join(args.outdir, f"fast_nostream_{args.speaker}.wav"), wavs[0], sr)
    print(f"  {stats(times)}  mean_audio={sum(durs)/len(durs):.2f}s  "
          f"mean_RTF_wall={(sum(times)/len(times))/(sum(durs)/len(durs)):.2f}x  "
          f"RTFx={(sum(durs)/len(durs))/(sum(times)/len(times)):.2f}x", flush=True)
    vram("after_nostream")

    # ---------- B. 流式（关注 TTFA） ----------
    print(f"\n--- B. streaming chunk_size={args.chunk_size} ---", flush=True)
    ttfa_list, tot_list, nchunks = [], [], []
    for i in range(args.repeat):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        first = None
        cnt = 0
        chunks = []
        for audio_chunk, csr, timing in model.generate_custom_voice_streaming(
            text=args.text, speaker=args.speaker, language=args.language,
            max_new_tokens=args.max_new, chunk_size=args.chunk_size,
        ):
            if first is None:
                first = time.perf_counter() - t1
            chunks.append(audio_chunk)
            cnt += 1
        total = time.perf_counter() - t1
        torch.cuda.synchronize()
        if first is None:
            print("  no chunks produced", flush=True)
            continue
        import numpy as np
        full = np.concatenate(chunks)
        audio_s = len(full) / csr
        ttfa_list.append(first)
        tot_list.append(total)
        nchunks.append(cnt)
        print(f"  run{i+1}: TTFA={first*1000:.0f}ms total={total:.3f}s audio={audio_s:.2f}s "
              f"chunks={cnt} RTF_wall={total/audio_s:.2f}x", flush=True)
        if i == 0:
            import soundfile as sf
            sf.write(os.path.join(args.outdir, f"fast_stream_{args.speaker}.wav"), full, csr)
    if ttfa_list:
        print(f"  TTFA {stats(ttfa_list)}", flush=True)
        print(f"  total {stats(tot_list)}  chunks={nchunks}", flush=True)
    vram("final")

    # ---------- C. 单词语音（听写场景） ----------
    print("\n--- C. single word 'ahead' ---", flush=True)
    for i in range(2):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text="Ahead.", speaker=args.speaker, language=args.language,
            max_new_tokens=256,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        print(f"  run{i+1}: total={dt:.3f}s audio={len(wavs[0])/sr:.2f}s", flush=True)
        if i == 0:
            import soundfile as sf
            sf.write(os.path.join(args.outdir, f"fast_word_{args.speaker}.wav"), wavs[0], sr)


if __name__ == "__main__":
    main()
