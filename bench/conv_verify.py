"""校验长回复的内容保真度与语速，解释 0.6B/1.7B 音频时长差异

回答两个问题：
  1. 长文本生成是否完整（有无漏句/复读/跑飞）——词级 WER + 逐句还原率
  2. 语速（词/秒）——为什么同样的文本两个模型音频时长差 5-8s

用法（主环境 memory-tts 跑，离线）:
    python bench/conv_verify.py --dir bench/conv-out
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.getcwd())

TEXT_200W = (
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
TEXT_MID = ("That's a great question. Let me think about it for a moment. "
            "I believe the best approach is to start small and build up gradually, "
            "so you don't get overwhelmed by too many details at once.")
TEXT_SHORT = "Sure, I can help you with that."

REF = {"chat_200w": TEXT_200W, "chat_mid": TEXT_MID, "chat_short": TEXT_SHORT}


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
    ap.add_argument("--dir", default="bench/conv-out")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if not files:
        print(f"{args.dir} 下没有 wav")
        return

    from faster_whisper import WhisperModel
    print("加载 Faster-Whisper base (cuda/fp16)...", flush=True)
    asr = WhisperModel("base", device="cuda", compute_type="float16")

    print(f"\n{'file':26s} {'dur':>6s} {'words':>6s} {'w/s':>6s} {'WER':>6s}  "
          f"{'n_seg':>5s}  回读开头")
    print("-" * 118)
    rows = []
    for f in files:
        d, sr = sf.read(f)
        if d.ndim > 1:
            d = d.mean(axis=1)
        dur = len(d) / sr
        segs, info = asr.transcribe(f, language="en", beam_size=5)
        segs = list(segs)
        hyp = " ".join(s.text for s in segs).strip()
        nw = len(norm_words(hyp))
        key = os.path.basename(f)
        for k in REF:  # 文件名形如 chat_200w_qwen-0.6b_s2048.wav
            if key.startswith(k):
                key = k
                break
        else:
            key = ""
        ref = REF.get(key, "")
        w = wer(ref, hyp) if ref else float("nan")
        print(f"{os.path.basename(f):26s} {dur:6.2f} {nw:6d} {nw/dur:6.2f} {w:6.3f}  "
              f"{len(segs):5d}  {hyp[:52]!r}")
        rows.append({"file": os.path.basename(f), "dur": round(dur, 2), "words": nw,
                     "wps": round(nw / dur, 3), "wer": round(w, 4), "n_seg": len(segs)})

    print("\n=== 语速与保真度汇总 ===")
    for tag in ["qwen-0.6b", "qwen-1.7b"]:
        sub = [r for r in rows if tag in r["file"]]
        if not sub:
            continue
        print(f"  {tag}: n={len(sub)} 平均语速={np.mean([r['wps'] for r in sub]):.2f} 词/秒 "
              f"平均WER={np.mean([r['wer'] for r in sub]):.4f}")

    import json
    out = os.path.join(args.dir, "verify.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
