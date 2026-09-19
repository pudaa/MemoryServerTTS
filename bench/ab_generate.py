"""生成 A/B 试听对照：基线（动态KV） vs CUDA Graph（backport）

输出 bench/ab/ 目录：
  - base_*.wav / fast_*.wav  成对音频（同一文本）
  - index.html               带播放器的试听页（含时长/耗时/RTF 对照）

用法:
    # 基线（主环境 memory-tts）
    python bench/ab_generate.py --side base  --model ./models/qwen-1.7b

    # fast（memory-tts-bp + backport）
    python bench/ab_generate.py --side fast  --model ./models/qwen-1.7b

    # 只用 backport 环境跑两个 side（推荐，保证变量唯一）
    python bench/ab_generate.py --side both  --model ./models/qwen-1.7b
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba-cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

# 让 backport 副本优先（仅当存在时）
_BP = os.path.join(os.getcwd(), ".backport")
if os.path.isdir(_BP):
    sys.path.insert(0, _BP)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

OUTDIR = os.path.join("bench", "ab-out")
SPEAKER = "aiden"
LANGUAGE = "English"

# 覆盖聊天朗读的真实长度分布 + 听写单词
CASES = [
    ("word1", "Ahead."),
    ("word2", "Well."),
    ("short1", "How are you?"),
    ("short2", "That sounds great!"),
    ("chat1", "I think you should give it another try tomorrow morning."),
    ("chat2", "Reading aloud every day is one of the fastest ways to improve your pronunciation and fluency."),
    ("long1", "Learning a new language takes time and patience. Don't worry if you make mistakes, because "
              "every mistake is a chance to learn something new. Keep practicing a little bit each day, "
              "and you will be surprised by how much progress you make."),
]


def gen_baseline(model_path):
    """上游动态 KV 路径"""
    from qwen_tts import Qwen3TTSModel

    print(f"=== BASELINE load {model_path} ===", flush=True)
    m = Qwen3TTSModel.from_pretrained(
        model_path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    results = []
    for key, text in CASES:
        torch.cuda.synchronize()
        t = time.perf_counter()
        wavs, sr = m.generate_custom_voice(
            text=text, language=LANGUAGE, speaker=SPEAKER, instruct=None,
            non_streaming_mode=True, temperature=0.9, top_k=50, top_p=1.0,
            repetition_penalty=1.05, max_new_tokens=2048,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        wav = wavs[0]
        path = os.path.join(OUTDIR, f"base_{key}.wav")
        sf.write(path, wav, sr)
        dur = len(wav) / sr
        results.append({"key": key, "text": text, "side": "base", "file": os.path.basename(path),
                        "gen_s": round(dt, 3), "dur_s": round(dur, 3),
                        "rtf_wall": round(dt / dur, 3)})
        print(f"  {key:7s} gen={dt:6.2f}s audio={dur:5.2f}s RTF={dt/dur:5.2f}x", flush=True)
    return results


def gen_fast(model_path):
    """CUDA Graph（backport）路径"""
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"=== FAST load {model_path} (CUDA graph) ===", flush=True)
    m = FasterQwen3TTS.from_pretrained(
        model_path, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    m.warmup(prefill_len=100)
    results = []
    for key, text in CASES:
        torch.cuda.synchronize()
        t = time.perf_counter()
        wavs, sr = m.generate_custom_voice(
            text=text, language=LANGUAGE, speaker=SPEAKER,
            max_new_tokens=2048, temperature=0.9, top_k=50, top_p=1.0,
            repetition_penalty=1.05,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        wav = wavs[0]
        path = os.path.join(OUTDIR, f"fast_{key}.wav")
        sf.write(path, wav, sr)
        dur = len(wav) / sr
        results.append({"key": key, "text": text, "side": "fast", "file": os.path.basename(path),
                        "gen_s": round(dt, 3), "dur_s": round(dur, 3),
                        "rtf_wall": round(dt / dur, 3)})
        print(f"  {key:7s} gen={dt:6.2f}s audio={dur:5.2f}s RTF={dt/dur:5.2f}x", flush=True)
    return results


def write_html(rows):
    by_key = {}
    for r in rows:
        by_key.setdefault(r["key"], {})[r["side"]] = r
    parts = ["""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>TTS A/B 试听对照 — 基线 vs CUDA Graph</title>
<style>
body{font-family:system-ui,Segoe UI,Microsoft YaHei,sans-serif;margin:24px;max-width:1100px;background:#fafafa;color:#222}
h1{font-size:20px} table{border-collapse:collapse;width:100%;background:#fff}
th,td{border:1px solid #ddd;padding:8px 10px;font-size:13px;vertical-align:top}
th{background:#f0f0f0;text-align:left} .txt{max-width:320px}
audio{width:230px;height:32px;display:block;margin:2px 0}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;color:#fff}
.slow{background:#c0392b} .fast{background:#27ae60}
.note{background:#fff8e1;border-left:4px solid #ffb300;padding:10px 14px;margin:16px 0;font-size:13px}
</style></head><body>
<h1>TTS A/B 试听对照 — 基线（动态 KV） vs CUDA Graph（backport）</h1>
<div class="note"><b>怎么听：</b>同一句话上下两行，先听 <b>BASE</b> 再听 <b>FAST</b>，重点判断
<b>音色是否一致、韵律是否自然、有没有机械感或拼接感</b>。耗时是实测墙钟时间（同一台 4060 Laptop）。
生成是随机采样，两次音频内容相同但韵律细节不会逐样本一致，这是正常的。</div>
<table><tr><th>case</th><th>文本</th><th>方案</th><th>音频</th><th>生成耗时</th><th>音频时长</th><th>RTF</th></tr>
"""]
    for key, _ in CASES:
        d = by_key.get(key, {})
        first = True
        for side, label, cls in [("base", "BASE 基线", "slow"), ("fast", "FAST CUDA Graph", "fast")]:
            r = d.get(side)
            if not r:
                continue
            span = f'<td class="txt" rowspan="2">{r["text"]}</td>' if first else ""
            parts.append(
                f'<tr><td>{key if first else ""}</td>{span}'
                f'<td><span class="badge {cls}">{label}</span></td>'
                f'<td><audio controls preload="none" src="{r["file"]}"></audio></td>'
                f'<td>{r["gen_s"]:.2f}s</td><td>{r["dur_s"]:.2f}s</td>'
                f'<td>{r["rtf_wall"]:.2f}×</td></tr>'
            )
            first = False
    # 汇总
    def agg(side):
        rs = [r for r in rows if r["side"] == side]
        if not rs:
            return None
        return (sum(r["gen_s"] for r in rs) / len(rs),
                sum(r["dur_s"] for r in rs) / len(rs),
                sum(r["rtf_wall"] for r in rs) / len(rs))
    a, b = agg("base"), agg("fast")
    if a and b:
        parts.append("</table><h2>汇总（各 case 均值）</h2><table>"
                     "<tr><th>方案</th><th>平均生成耗时</th><th>平均音频时长</th><th>平均 RTF</th></tr>")
        parts.append(f'<tr><td>BASE 基线</td><td>{a[0]:.2f}s</td><td>{a[1]:.2f}s</td><td>{a[2]:.2f}×</td></tr>')
        parts.append(f'<tr><td>FAST CUDA Graph</td><td>{b[0]:.2f}s</td><td>{b[1]:.2f}s</td><td>{b[2]:.2f}×</td></tr>')
        parts.append(f'<tr><td><b>提速</b></td><td colspan="3"><b>{a[0]/b[0]:.1f}×</b></td></tr></table>')
    parts.append("</body></html>")
    with open(os.path.join(OUTDIR, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["base", "fast", "both"], default="both")
    ap.add_argument("--model", default="./models/qwen-1.7b")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    meta_path = os.path.join(OUTDIR, "results.json")
    if os.path.exists(meta_path):
        rows = json.load(open(meta_path, encoding="utf-8"))
    rows = [r for r in rows if r["side"] != args.side or args.side == "both"]

    if args.side in ("base", "both"):
        rows += gen_baseline(args.model)
        torch.cuda.empty_cache()
    if args.side in ("fast", "both"):
        rows += gen_fast(args.model)

    json.dump(rows, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    write_html(rows)
    print(f"\n生成完毕 -> {OUTDIR}/index.html（用浏览器打开试听）", flush=True)


if __name__ == "__main__":
    main()
