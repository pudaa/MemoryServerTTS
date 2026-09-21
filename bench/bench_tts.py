"""Qwen3-TTS 分阶段性能基准脚本

用途：把 Qwen3-TTS 的一次生成拆成两个阶段分别计时，用来判断"慢"到底出在哪一段：

  stage 1  AR 生成  : talker(28层) + code_predictor(15次串行) 生成 16 组语音码
  stage 2  codec    : speech_tokenizer.decode() 把语音码还原成波形

结论（2026-09-19，RTX 4060 Laptop 8G）：stage 1 占 96–99%，stage 2 仅 1–4%。
详见 docs/QWEN3_TTS_RESEARCH_REPORT.md §7。

用法（必须先从项目根目录启动，脚本依赖相对路径 ./models/...）:

    python bench/bench_tts.py --model ./models/qwen-1.7b --repeat 2

⚠️ 环境要求：必须先设置 NUMBA_CACHE_DIR 指向可写目录，否则 `import qwen_tts`
   会卡死（librosa → numba JIT 缓存目录不可写，见 bench/README.md）。

    $env:NUMBA_CACHE_DIR = "$PWD\.numba-cache"
    python bench/bench_tts.py --model ./models/qwen-1.7b
"""
import argparse
import os
import time

# ── 必须在导入 librosa/numba 之前设置，否则 import 卡死 ──
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba-cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import torch  # noqa: E402


def vram(tag):
    """打印当前/峰值显存占用（MB）"""
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 2**20
    peak = torch.cuda.max_memory_allocated() / 2**20
    resv = torch.cuda.memory_reserved() / 2**20
    print(f"[VRAM:{tag}] allocated={alloc:.0f}MB peak={peak:.0f}MB reserved={resv:.0f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen-1.7b")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--texts", default="Hello. This is a speed benchmark test.",
                    help="多段文本用 | 分隔")
    ap.add_argument("--speaker", default="aiden")
    ap.add_argument("--language", default="English")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"=== load {args.model} dtype={args.dtype} attn={args.attn} ===")
    t0 = time.perf_counter()
    from qwen_tts import Qwen3TTSModel

    m = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=dt, attn_implementation=args.attn
    )
    print(f"load_seconds={time.perf_counter()-t0:.1f}")
    vram("after_load")

    inner = m.model
    cfg = inner.config.talker_config
    cpc = cfg.code_predictor_config
    print(f"num_code_groups={cfg.num_code_groups} talker_layers={cfg.num_hidden_layers} "
          f"talker_hidden={cfg.hidden_size}")
    print(f"code_predictor: layers={cpc.num_hidden_layers} hidden={cpc.hidden_size} "
          f"vocab={cpc.vocab_size}")
    print(f"frame_rate_hz ~= {cfg.position_id_per_seconds}")

    for text in [t for t in args.texts.split("|") if t]:
        print(f"\n--- text={text!r} speaker={args.speaker} lang={args.language} ---")
        for i in range(args.repeat):
            # ---------- stage 1: AR 语音码生成 ----------
            input_ids = m._tokenize_texts([m._build_assistant_text(text)])
            gen_kwargs = m._merge_generate_kwargs(
                temperature=0.9, top_k=50, top_p=1.0,
                repetition_penalty=1.05, max_new_tokens=args.max_new,
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            codes_list, _ = inner.generate(
                input_ids=input_ids,
                instruct_ids=[None],
                languages=[args.language],
                speakers=[args.speaker],
                non_streaming_mode=True,
                **gen_kwargs,
            )
            torch.cuda.synchronize()
            t_ar = time.perf_counter() - t1

            n_frames = int(codes_list[0].shape[0])
            secs = n_frames / cfg.position_id_per_seconds

            # ---------- stage 2: codec 波形解码 ----------
            t2 = time.perf_counter()
            wavs, sr = inner.speech_tokenizer.decode([{"audio_codes": c} for c in codes_list])
            torch.cuda.synchronize()
            t_voc = time.perf_counter() - t2

            audio_s = len(wavs[0]) / sr
            total = t_ar + t_voc
            print(
                f"run{i+1}: frames={n_frames} (~{secs:.2f}s audio) "
                f"| stage1_AR={t_ar:.3f}s stage2_codec={t_voc:.3f}s total={total:.3f}s "
                f"| audio={audio_s:.2f}s RTF_total={total/audio_s:.2f}x "
                f"RTF_AR={t_ar/audio_s:.2f}x RTF_codec={t_voc/audio_s:.2f}x "
                f"| AR_share={100*t_ar/total:.0f}%"
            )
            vram(f"run{i+1}")


if __name__ == "__main__":
    main()
