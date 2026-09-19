"""决定性分步剖析：Talker vs Code Predictor vs 采样/glue 各占多少

动机：`faster_qwen3_tts` 的 ms_per_step 只含 AR 循环，**不含 codec 解码**；
且需要确认每步时间到底花在 Talker 还是 Code Predictor 上——
这决定了"还有没有 2x 空间"：
  Predictor 占比 >=60% => 改 Talker（量化/FA2/缓存长度）都到不了 2x
  Predictor 占比 <=40% => 值得继续扫 max_seq_len / mask-free SDPA

方法：用 CUDA Event 包住 predictor_graph.run / talker_graph.run / 其余 glue，
**不在每步 synchronize**（device Event 是异步的，不会像 .item() 那样破坏流水），
最后一次性 elapsed_time 全部求和。另单独测量 codec 解码并确认其设备。

用法（memory-tts-bp 环境，项目根目录）:
    python bench/profile_steps.py --model ./models/qwen-0.6b --runs 3
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

PROMPT = (
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen-0.6b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--outdir", default="bench/prof-out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from faster_qwen3_tts import FasterQwen3TTS
    import faster_qwen3_tts.generate as G

    print(f"=== load {args.model} ===", flush=True)
    m = FasterQwen3TTS.from_pretrained(
        args.model, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    m.warmup(prefill_len=120)

    # ---- 给 decode 循环打桩：用 CUDA Event 记录各段 GPU 时间 ----
    stats = {"pred": [], "talker": [], "steps": [], "wall": []}
    orig_pred_run = type(m.predictor_graph).run
    orig_talker_run = type(m.talker_graph).run

    def make_wrap(orig, key):
        def wrapped(self, *a, **kw):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = orig(self, *a, **kw)
            e1.record()
            stats[key].append((e0, e1))
            return out
        return wrapped

    type(m.predictor_graph).run = make_wrap(orig_pred_run, "pred")
    type(m.talker_graph).run = make_wrap(orig_talker_run, "talker")

    # ---- 捕获每次 fast_generate 的 steps 与 wall ----
    orig_fg = G.fast_generate

    def fg_wrap(*a, **kw):
        t0 = time.perf_counter()
        codes, timing = orig_fg(*a, **kw)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        stats["steps"].append(timing.get("steps", 0))
        stats["wall"].append(wall)
        return codes, timing

    G.fast_generate = fg_wrap

    # ---- 关键：让 model.py 里 import 的 fast_generate 也指向包装版 ----
    import faster_qwen3_tts.model as M
    M.fast_generate = fg_wrap

    print(f"\n=== run {args.runs}x on 198-word prompt ===", flush=True)
    for r in range(args.runs):
        stats["pred"].clear()
        stats["talker"].clear()
        wavs, sr = m.generate_custom_voice(
            text=PROMPT, speaker="aiden", language="English", max_new_tokens=4096)
        dur = len(wavs[0]) / sr
        npred, ntalk = len(stats["pred"]), len(stats["talker"])
        # 一次性求和各段耗时
        t_pred = sum(e0.elapsed_time(e1) for e0, e1 in stats["pred"]) / 1000
        t_talk = sum(e0.elapsed_time(e1) for e0, e1 in stats["talker"]) / 1000
        steps = stats["steps"][-1]
        wall = stats["wall"][-1]
        other = wall - t_pred - t_talk
        print(f"  run{r+1}: steps={steps} audio={dur:.1f}s wall={wall:.2f}s "
              f"| predictor={t_pred:.2f}s ({100*t_pred/wall:.0f}%) "
              f"talker={t_talk:.2f}s ({100*t_talk/wall:.0f}%) "
              f"other={other:.2f}s ({100*other/wall:.0f}%) "
              f"| calls pred={npred} talker={ntalk}", flush=True)
        if steps:
            print(f"          per-step: pred={1000*t_pred/npred if npred else 0:.2f}ms "
                  f"talker={1000*t_talk/ntalk if ntalk else 0:.2f}ms "
                  f"total={1000*wall/steps:.2f}ms", flush=True)

    # ---- codec 解码单独计时 + 设备确认 ----
    print("\n=== codec (speech_tokenizer) 单独测量 ===", flush=True)
    import numpy as np

    st = None
    obj = m
    for path in ("model.speech_tokenizer", "speech_tokenizer",
                 "model.model.speech_tokenizer"):
        cur = m
        ok = True
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                ok = False
                break
        if ok:
            st = cur
            print(f"  found via .{path}", flush=True)
            break
    if st is None:
        # 兜底：递归找带 decode 的对象
        seen = set()

        def walk(o, depth=0):
            if depth > 4 or id(o) in seen:
                return None
            seen.add(id(o))
            if hasattr(o, "decode") and "Tokenizer" in type(o).__name__:
                return o
            for k, v in vars(o).items() if hasattr(o, "__dict__") else []:
                if isinstance(v, torch.nn.Module):
                    r = walk(v, depth + 1)
                    if r is not None:
                        return r
            return None

        st = walk(m)
        print(f"  found via walk: {type(st).__name__ if st else None}", flush=True)

    inner = getattr(st, "model", st)
    try:
        dev = next(inner.parameters()).device
    except Exception:
        dev = "?"
    print(f"  tokenizer={type(st).__name__} inner={type(inner).__name__} device={dev}", flush=True)

    for n_frames in (50, 200, 800):
        fake = torch.randint(0, 2048, (n_frames, 16), dtype=torch.long, device="cuda")
        for i in range(2):
            torch.cuda.synchronize()
            t = time.perf_counter()
            wavs2, sr2 = st.decode([{"audio_codes": fake}])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            a = len(np.asarray(wavs2[0]).flatten()) / sr2
            print(f"  frames={n_frames:4d} ({n_frames/12.5:5.1f}s audio) "
                  f"codec_time={dt:6.3f}s  ({1000*dt/n_frames:5.2f} ms/frame)  "
                  f"audio_out={a:.1f}s", flush=True)

    # ---- 恢复打桩 ----
    type(m.predictor_graph).run = orig_pred_run
    type(m.talker_graph).run = orig_talker_run
    G.fast_generate = orig_fg
    M.fast_generate = orig_fg


if __name__ == "__main__":
    main()
