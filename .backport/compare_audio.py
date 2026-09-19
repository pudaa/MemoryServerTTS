"""对比基线（动态KV）与 backport（CUDA Graph）生成的音频是否质量一致。

因为采样随机、且静态/动态KV的 kernel 归约顺序不同，两者**不会逐样本相同**，
本脚本比较客观可比的量：时长、RMS、峰值、以及 ASR 回读是否能识别出目标文本。
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "Hello. This is a speed benchmark test."


def describe(path):
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    rms = float(np.sqrt(np.mean(d**2)))
    return {
        "file": os.path.basename(path),
        "sr": sr,
        "dur": len(d) / sr,
        "peak": float(np.abs(d).max()),
        "rms": rms,
        "db": 20 * np.log10(rms) if rms > 0 else -99,
        "clipped": int((np.abs(d) > 0.999).sum()),
        "silence_ratio": float(np.mean(np.abs(d) < 1e-4)),
    }


def main():
    files = []
    for pat in ["bench/out/*.wav", "bench/out-bp/*.wav", "bench/out-bp06/*.wav",
                "bench/out-baseline/*.wav"]:
        files += sorted(glob.glob(os.path.join(ROOT, pat)))
    if not files:
        print("no wav files found")
        return
    print(f"{'file':30s} {'sr':>6s} {'dur':>7s} {'peak':>6s} {'rms':>7s} "
          f"{'dBFS':>7s} {'clip':>5s} {'sil%':>6s}")
    print("-" * 82)
    for f in files:
        d = describe(f)
        print(f"{d['file']:30s} {d['sr']:6d} {d['dur']:7.3f} {d['peak']:6.3f} "
              f"{d['rms']:7.4f} {d['db']:7.2f} {d['clipped']:5d} "
              f"{d['silence_ratio']*100:6.1f}")

    # ASR 回读（用主环境的 faster-whisper，验证内容正确性）
    print("\n=== ASR 回读（判断内容是否为目标句）===")
    try:
        sys.path.insert(0, ROOT)
        from faster_whisper import WhisperModel
        m = WhisperModel("base", device="cuda", compute_type="float16")
        for f in files:
            segs, info = m.transcribe(f, language="en", beam_size=5)
            text = " ".join(s.text for s in segs).strip()
            print(f"  {os.path.basename(f):30s} -> {text!r}")
    except Exception as e:
        print(f"  ASR skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
