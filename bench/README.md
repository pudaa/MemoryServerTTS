# bench — TTS 性能基准

调研用工具，不属于在线服务链路。用于回答"TTS 到底慢在哪一段"。

## bench_tts.py

把一次 Qwen3-TTS 生成拆成两段分别计时：

| 阶段 | 内容 |
|------|------|
| stage 1 `AR` | talker(28 层) + code_predictor(每帧 15 次串行) 生成 16 组语音码 |
| stage 2 `codec` | `speech_tokenizer.decode()` 把语音码还原成 24kHz 波形 |

RTF 定义：`RTF = 墙钟秒数 / 音频秒数`，所以 **RTF > 1 表示比实时慢**。
（注意与 `faster-qwen3-tts` README 相反，它报的是 `audio/wall`，其 "0.23" 等于本表的 "4.3"。）

### 运行

```powershell
conda activate memory-tts
cd D:\Codes\MemoryServerTTS

# ⚠️ 必须设置，否则 import qwen_tts 会卡死（见下）
$env:NUMBA_CACHE_DIR = "$PWD\.numba-cache"

python bench/bench_tts.py --model ./models/qwen-1.7b --repeat 2
python bench/bench_tts.py --model ./models/qwen-0.6b --repeat 2
```

### ⚠️ NUMBA_CACHE_DIR 必须设置

`import qwen_tts` → `librosa` → `numba` 的 `@jit(cache=True)` 会在**导入期**去建 JIT 缓存目录。
若该目录不可写，numba 不会抛错而是**阻塞**（`tempfile._mkstemp_inner` 挂住），表现为导入卡死：

| 场景 | `import qwen_tts` 耗时 |
|------|----------------------|
| 默认 numba 缓存目录（本机受限） | **>420 s（实测卡死）** |
| `NUMBA_CACHE_DIR` 指向可写目录 | **17.2 s** |

排查方法：`faulthandler.dump_traceback_later(60)` 后 import，看栈顶是不是
`numba/core/caching.py` → `tempfile.py`。

这条与服务启动相关：若服务进程的 numba 缓存目录不可写，**启动会假死**而不是报错。
建议在 `main.py` / `src/server.py` 顶部、`import torch` 之前固定一个进程可写的 `NUMBA_CACHE_DIR`。

### 2026-09-19 实测结果（RTX 4060 Laptop 8G，bf16 + sdpa）

文本 `"Hello. This is a speed benchmark test."`，音色 aiden：

| 模型 | 音频 | stage1 AR | stage2 codec | 合计 | RTF | AR 占比 |
|------|------|-----------|--------------|------|-----|---------|
| 1.7B | 3.12 s | 19.242 s | 0.664 s | 19.906 s | 6.38× | **97 %** |
| 1.7B | 2.72 s | 16.223 s | 0.112 s | 16.335 s | 6.01× | **99 %** |
| 0.6B | 3.12 s | 15.342 s | 0.627 s | 15.970 s | 5.12× | **96 %** |
| 0.6B | 3.28 s | 15.092 s | 0.100 s | 15.192 s | 4.63× | **99 %** |

显存：1.7B 加载后 3 985 MB / reserved 4 112 MB，生成峰值 4 085 MB；
0.6B 加载后 2 057 MB / reserved 2 184 MB，生成峰值 2 162 MB。
（**不含** 同进程的 Faster-Whisper fp16 ~1 GB 与 PP-OCRv6 ~0.5 GB。）

**结论：瓶颈在 stage 1（AR 码生成本身），不在 codec 声码器。**
优化方向应指向"每帧 1 次 talker + 15 次串行 code_predictor 前向"的
Python/kernel-launch 开销（CUDA Graph + 静态 KV cache），而不是声码器。

完整分析见 `docs/QWEN3_TTS_RESEARCH_REPORT.md`。
