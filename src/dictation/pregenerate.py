"""
离线词库预生成 CLI（听写场景，Phase 2）

强制使用 1.7B 模型（0.6B 不支持 instruct，听写指令会被静默丢弃），
每个词 best-of-N 多 seed 生成 + ASR 校验择优，仅最优候选入库。

用法:
  python -m src.dictation.pregenerate --words ahead,behind,cat
  python -m src.dictation.pregenerate --file words.txt
  python -m src.dictation.pregenerate --file words.csv --best-of 5

词表格式（--file）:
  - 纯单词行:  ahead                → 使用 --voice / --language / --instruct 默认值
  - CSV 行:    ahead,aiden,English,Speak calmly
               共 4 列: word,voice,language,instruct（instruct 可留空）
"""
import argparse
import os
import sys
import time

# 固定 1.7B（TTSCONF_MODEL_PATH 覆盖 tts.yaml 的 model_path）
_DEFAULT_MODEL_PATH = "./models/qwen-1.7b"


def _parse_args():
    p = argparse.ArgumentParser(description="词库离线预生成（固定 1.7B + best-of-N + ASR 校验）")
    p.add_argument("--words", type=str, default="",
                   help="逗号分隔的单词列表，如 ahead,behind,cat")
    p.add_argument("--file", type=str, default="",
                   help="词表文件：纯单词一行一个，或 CSV（word,voice,language,instruct）")
    p.add_argument("--voice", type=str, default="",
                   help="默认音色（未在词表指定时使用；留空按语言自动匹配母语音色）")
    p.add_argument("--language", type=str, default="English", help="默认语言")
    p.add_argument("--instruct", type=str, default="",
                   help="默认指令（业务侧风格控制，如 Speak in a calm tone）")
    p.add_argument("--best-of", type=int, default=0,
                   help="候选数（默认取配置 dictation.best_of=3）")
    p.add_argument("--seed-base", type=int, default=0,
                   help="候选 seed 起点（默认取配置 dictation.seed_base=1000）")
    p.add_argument("--model-path", type=str, default=_DEFAULT_MODEL_PATH,
                   help=f"模型路径（默认 {_DEFAULT_MODEL_PATH}，勿改 0.6B）")
    return p.parse_args()


def _load_specs(args):
    """解析词表为 [(word, voice, language, instruct)]，缺失字段用默认值"""
    specs = []
    default_voice = args.voice.strip()
    default_instruct = args.instruct.strip() or None

    def add(line):
        line = line.strip()
        if not line or line.startswith("#"):
            return
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            # CSV: word,voice,language[,instruct]
            word = parts[0]
            voice = parts[1] or default_voice
            language = parts[2] if len(parts) > 2 and parts[2] else args.language
            instruct = parts[3] if len(parts) > 3 and parts[3] else default_instruct
            specs.append((word, voice, language, instruct))
        else:
            specs.append((line, default_voice, args.language, default_instruct))

    if args.words:
        for w in args.words.split(","):
            add(w)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                add(line)
    return specs


def main():
    args = _parse_args()
    os.environ["TTSCONF_MODEL_PATH"] = args.model_path
    os.environ.setdefault("LOG_LEVEL", "INFO")

    from src.tts.config import TTSConfig
    from src.tts.model_loader import TTSModelManager
    from src.asr.model_loader import ASRModelManager
    from src.dictation import cache, generator
    from src.dictation.spec import normalize_spec

    specs = _load_specs(args)
    if not specs:
        print("词表为空：请用 --words 或 --file 提供单词")
        sys.exit(2)

    print(f"加载模型: {args.model_path}")
    t0 = time.time()
    cfg = TTSConfig()
    model_mgr = TTSModelManager(config=cfg)
    asr_mgr = ASRModelManager()
    if getattr(model_mgr.model, "tts_model_size", "") != "1b7":
        print(f"[警告] 当前模型为 {getattr(model_mgr.model, 'tts_model_size', '?')}，"
              f"不是 1.7B；instruct 可能失效")

    best_of = args.best_of or cfg.dictation_best_of
    seed_base = args.seed_base or cfg.dictation_seed_base

    print(f"预生成 {len(specs)} 个词（best_of={best_of}）...\n")
    print(f"{'单词':<24} {'score':>6} {'conf':>7} {'时长':>6}  {'状态'}")
    print("-" * 70)
    ok_count = 0
    for word, voice, language, instruct in specs:
        word, voice, language, instruct = normalize_spec(
            cfg, word, voice, language, instruct)
        key = cache.cache_key(word, voice, language, instruct, cfg)
        best, failures = generator.generate_best(
            word=word, voice=voice, language=language, instruct=instruct,
            synth_fn=generator.make_synth_fn(model_mgr.model, cfg.short_decode),
            verify_fn=generator.make_verify_fn(
                asr_mgr, cfg.dictation_conf_threshold),
            best_of=best_of, seed_base=seed_base,
            conf_threshold=cfg.dictation_conf_threshold,
        )
        if best is None:
            reason = failures[0]["reason"] if failures else "unknown"
            print(f"{word:<24} {'-':>6} {'-':>7} {'-':>6}  失败 ({reason})")
            continue
        cache.save(cfg, key, best["wav"], best["sr"], {
            "word": word, "voice": voice, "language": language,
            "instruct": instruct,
            "gen_config_version": cache.gen_config_version(cfg),
            "seed": best["seed"], "attempts": len(failures) + 1,
            "verified": True, "asr_text": best["asr_text"],
            "avg_logprob": best["avg_logprob"],
            "quality_score": best["score"], "duration": best["duration"],
        })
        ok_count += 1
        print(f"{word:<24} {best['score']:>6.3f} "
              f"{str(best['avg_logprob']):>7} {best['duration']:>5.2f}s  ✓")

    s = cache.summary(cfg)
    print("-" * 70)
    print(f"完成: {ok_count}/{len(specs)} 成功 | 缓存总数 {s['total']} "
          f"(已校验 {s['verified']}, bad {s['bad']}) | 耗时 {time.time()-t0:.1f}s")
    sys.exit(0 if ok_count == len(specs) else 1)


if __name__ == "__main__":
    main()
