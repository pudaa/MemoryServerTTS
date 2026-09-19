"""A/B 音频的 ASR 内容校验：确认 fast 方案输出能被正确识别为原文

对 bench/ab/ 下每个 wav 做 Whisper 回读，并用词级 WER 与原文比较。
用法（主环境 memory-tts 运行；读 .backport 的 compare 不依赖）:
    python bench/ab_verify_asr.py [--dir bench/ab]
"""
import argparse
import glob
import json
import os
import re
import sys

import soundfile as sf

sys.path.insert(0, os.getcwd())

_LANG_TO_ISO = {"English": "en", "Chinese": "zh", "Japanese": "ja", "Korean": "ko"}


def norm_words(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff']+", " ", s)
    return [w for w in s.split() if w]


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 0.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench/ab")
    args = ap.parse_args()

    meta_path = os.path.join(args.dir, "results.json")
    if not os.path.exists(meta_path):
        print(f"缺少 {meta_path}；请先运行 bench/ab_generate.py")
        return
    rows = json.load(open(meta_path, encoding="utf-8"))

    from faster_whisper import WhisperModel
    print("加载 Faster-Whisper base (cuda/fp16)...", flush=True)
    asr = WhisperModel("base", device="cuda", compute_type="float16")

    print(f"\n{'case':8s} {'side':5s} {'dur':>6s} {'WER':>6s}  ASR 回读")
    print("-" * 100)
    summary = {}
    for r in sorted(rows, key=lambda x: (x["key"], x["side"])):
        path = os.path.join(args.dir, r["file"])
        if not os.path.exists(path):
            print(f"{r['key']:8s} {r['side']:5s}  MISSING {r['file']}")
            continue
        segs, info = asr.transcribe(path, language="en", beam_size=5)
        hyp = " ".join(s.text for s in segs).strip()
        d, sr = sf.read(path)
        dur = len(d) / sr
        w = wer(r["text"], hyp)
        print(f"{r['key']:8s} {r['side']:5s} {dur:6.2f} {w:6.3f}  {hyp!r}")
        summary.setdefault(r["side"], []).append(w)

    print("\n=== 平均 WER（越低越好）===")
    for side, ws in summary.items():
        print(f"  {side:5s} n={len(ws)} mean_WER={sum(ws)/len(ws):.4f} "
              f"max={max(ws):.4f} fail(WER>0.2)={sum(1 for x in ws if x > 0.2)}")

    # 落盘供报告引用
    out = os.path.join(args.dir, "asr_wer.json")
    json.dump({k: {"n": len(v), "mean_wer": sum(v) / len(v), "max_wer": max(v), "raw": v}
               for k, v in summary.items()},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
