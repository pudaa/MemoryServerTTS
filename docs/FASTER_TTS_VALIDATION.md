# CUDA Graph 方案验证报告（faster-qwen3-tts backport）

> 2026-09-19 实测。结论：**可行，且不需要升级 transformers。**
> 主环境（`memory-tts`，transformers 4.57.3 / qwen-tts 0.1.1 / torch 2.7.0+cu128）**全程未被修改**。

---

## 1. 一句话结论

用 `faster-qwen3-tts` 的 **CUDA Graph + 静态 KV cache** 替换上游的动态 KV 解码循环，
在本机 RTX 4060 Laptop 8G 上把短句生成从 **6.2–9.1s 降到 1.4–2.1s（约 4–4.5 倍）**，
流式首块延迟 **TTFA 约 0.35–0.41s**，显存几乎不变（+265MB）。
关键点：其"必须 transformers 5.x"的约束是**包元数据强加的，不是代码需要的**——
打 2 处兼容补丁后可在 **transformers 4.57.3** 上原样运行，性能无损。

---

## 2. 背景修正（重要）

调研阶段的初次测量给出基线 16–19s。**复测后修正为 6.2–9.1s**：

| 测量批次 | 1.7B 基线耗时 | 说明 |
|---|---|---|
| 首次（调研话轮） | 16.2–19.9s | **不可复现，偏高**，已废弃 |
| 复测（本话轮，主环境） | 6.4–9.1s（3 次） | 可复现 |
| 复测（本话轮，perf 环境） | 5.2–6.4s（3 次） | 可复现 |

复测三次结果一致，且两个不同环境互相印证，故**以 6.2–9.1s 为准**。
（首次偏高的原因未完全定位，可能与该轮初始 CUDA 上下文/显存状态有关；已如实记录，
不做掩盖。重要的是：**所有对比都在同一环境、同一会话内 measured back-to-back**，不受此影响。）

上游基线在**两个环境里都**是 stage 1（AR 码生成）占 **96–99%**，codec 仅 1–4% —— 结论不变：
瓶颈是每帧 1 次 talker + 15 次串行 code_predictor 前向的 Python/kernel 启动开销。

---

## 3. 实测对比（同一环境内 back-to-back）

文本 `"Hello. This is a speed benchmark test."`，音色 `aiden`，bf16 + sdpa，非流式。
RTF_wall = 墙钟/音频（<1 才是快于实时）。

| 方案 | 环境 | 总耗时（mean） | 音频 | RTF_wall | RTFx | TTFA（流式） | 常驻显存 |
|---|---|---|---|---|---|---|---|
| 基线（动态 KV） | main / tf 4.57.3 | 6.49–9.06s | 2.9–3.6s | **2.25–2.52×** | 0.40–0.44× | — | 3 993MB |
| 基线（动态 KV） | perf / tf 5.15.1 | 5.17–6.44s | 2.2–2.7s | 2.28–2.51× | 0.40–0.44× | — | 3 990MB |
| **CUDA Graph 1.7B** | perf / tf 5.15.1 | **1.79s** | 3.04s | **0.59×** | 1.70× | **414ms** | 4 247MB |
| **CUDA Graph 1.7B（backport）** | **bp / tf 4.57.3** | **1.65s** | 2.83s | **0.58×** | 1.70× | **414ms** | 4 247MB |
| **CUDA Graph 0.6B（backport）** | **bp / tf 4.57.3** | **1.46s** | 3.20s | **0.46×** | 2.17× | **346ms** | **2 319MB** |

单词语音（`"Ahead."`）：

| 方案 | 耗时 | 音频 |
|---|---|---|
| 基线 1.7B | 约 6–8s（同量级） | — |
| CUDA Graph 1.7B | **0.37s** | 0.48s |
| CUDA Graph 0.6B | **0.52s** | 1.04s |

显存明细（峰值）：

| 方案 | 加载后 allocated / reserved | 生成峰值 |
|---|---|---|
| 1.7B 基线 | 3 985 / 4 112 MB | 4 100 MB |
| 1.7B CUDA Graph | 3 982 / 4 112 MB | 4 327 MB（含图捕获，+265MB） |
| 0.6B 基线 | 2 057 / 2 184 MB | 2 162 MB |
| 0.6B CUDA Graph | 2 054 / 2 182 MB | **2 408 MB** |

CUDA Graph 预热（捕获）耗时仅 **1.5–1.8s**，远优于 `torch.compile` 的 30s–5min。

---

## 4. 与 ASR / OCR 共存的显存账（8 188 MB 可用）

| 组成 | 1.7B + CUDA Graph | 0.6B + CUDA Graph |
|---|---|---|
| TTS | 4 327 MB | 2 408 MB |
| Faster-Whisper base fp16 | ~1 000 MB | ~1 000 MB |
| PP-OCRv6 small (onnxruntime-gpu) | ~500 MB | ~500 MB |
| MuMu 模拟器（本机其他进程） | ~350 MB | ~350 MB |
| **合计** | **~6 180 MB** | **~4 260 MB** |
| 余量 | ~2 000 MB | **~3 900 MB** |

> 建议：**chat 朗读走 0.6B**（余量充足、更快、TTFA 更低）；1.7B 仅在需要 `instruct`
> （听写词库预生成、离线批量）时使用——听写是离线缓存，延迟不敏感（约束 M1/M2）。

---

## 5. 音质与内容校验

用主环境的 Faster-Whisper base 做 ASR 回读（同一批 wav）：

| 样本 | ASR 回读 | 判定 |
|---|---|---|
| 基线 1.7B 非流式 | `Hello, this is a Speed Benchmark test.` | ✅ |
| CUDA Graph 1.7B 非流式 | `Hello, this is a speed benchmark test.` | ✅ |
| CUDA Graph 1.7B 流式 | `Hello, this is a Speed Benchmark test.` | ✅ |
| CUDA Graph 0.6B 非流式 | `Hello, this is a speed benchmark test.` | ✅ |
| CUDA Graph 0.6B 流式 | `Hello, this is a Speed Benchmark test.` | ✅ |
| 单词（1.7B，fast） | `Head.` | ⚠️ 丢首音 |
| 单词（0.6B，fast） | `ahead.` | ✅ |

波形客观量（无削波、电平合理）：

| 样本 | 时长 | 峰值 | RMS | dBFS | 削波 |
|---|---|---|---|---|---|
| 基线 1.7B | 2.96s | 0.852 | 0.1207 | −18.4 | 0 |
| CG 1.7B 非流式 | 3.20s | 0.742 | 0.1715 | −15.3 | 0 |
| CG 1.7B 流式 | 2.80s | 0.805 | 0.1440 | −16.8 | 0 |
| CG 0.6B 非流式 | 3.12s | 0.680 | 0.0974 | −20.2 | 0 |

**结论**：句子级内容、电平、无削波均与基线一致，可安全替换。
短单词存在偶发首音丢失（`ahead` → `Head`），这正是项目**既有 ASR 校验闭环**要处理的情况
（约束 G1/G3）：fast 方案下单次生成仅 0.37–0.52s，**best-of-N 重试完全可行**，
而基线单次 6–8s 使重试代价高昂。→ fast 方案实际上**让校验闭环更实用**。

⚠️ 待人工试听确认（客观指标不能替代听感）：音色一致性、韵律自然度、
流式块边界是否有拼接感。样本在 `bench/out*/`（已 gitignore）。

---

## 6. 关键技术发现：transformers 5.x 是可以避免的

### 6.1 直接安装会踩的坑

`pip install faster-qwen3-tts` 在干净的克隆环境里做 dry-run，会改动：

```
Would install: faster-qwen3-tts-0.4.0  qwen-tts-hf-0.1.1.post1
               transformers-5.17.0  tokenizers-0.23.2
               huggingface-hub-1.31.0  hf-xet-1.6.0
torch: 不动（2.7.0+cu128 保留）✅
```

即 `transformers` 4.57.3 → **5.17.0**、`huggingface_hub` 0.36.2 → **1.31.0** 两个大版本跳跃。
`qwen-tts-hf` 与上游 `qwen-tts` 提供**同一个 `qwen_tts` 模块**，两者不能共存。

### 6.2 装完 5.17.0 直接报错

```
AttributeError: 'MimiConfig' object has no attribute 'rope_theta'
  transformers/modeling_utils.py:_finalize_model_loading → _initialize_missing_keys
  → initialize_weights → mimi._init_weights → rope_fn(config)
```
transformers 5.17 收紧了"缺失权重键"路径，会对 speech_tokenizer 触发权重初始化，
而 `MimiConfig` 已无 `rope_theta`。上游只在 PR #135 里把默认版本定为 **5.15**，
pip 却解析到 5.17 —— 属于版本漂移未设上界。
**修法**：降到 `transformers==5.15.1` 即可恢复正常。

### 6.3 真正的发现：代码根本不需要 5.x

审查 `faster_qwen3_tts` 源码：它**只**从 `qwen_tts` 导入 `Qwen3TTSModel`，
CUDA Graph 逻辑全部自研，仅依赖 transformers 的 `StaticCache` / `DynamicCache`。
`transformers>=5.15.1` 这个约束是 **`qwen-tts-hf` 带进来的传递依赖**，不是代码需求。

在 transformers 4.57.3 上实测，只有 **2 处 API 不兼容**：

| 位置 | 5.x 写法 | 4.57 写法 |
|---|---|---|
| `StaticLayer.lazy_initialization` | `(key, value)` 2 参数 | `(key)` 1 参数 |
| 取 DynamicCache 的 K/V | `cache.layers[i].keys/.values` | `cache[i]` 返回 `(k, v)` |

（对应上游 PR #65 "fix transformers 5.x API breaks in StaticCache and DynamicCache" 的反向移植。）

### 6.4 backport 结果

补齐上面 2 处兼容分支后，在 **transformers 4.57.3 + qwen-tts 0.1.1 + torch 2.7.0+cu128**
上 CUDA Graph 正常捕获并运行，**性能与 5.x 环境完全一致**（1.7B：1.65s / TTFA 414ms）。

```
.backport/faster_qwen3_tts/     # 打补丁的本地副本（来源 v0.4.0, MIT）
├── model.py                    # 未改（仅从 qwen_tts 导入 Qwen3TTSModel）
├── predictor_graph.py          # 补丁 1
├── talker_graph.py             # 补丁 1 + 补丁 2
├── generate.py / streaming.py / sampling.py / utils.py / ggml_backend.py
└── __init__.py                 # 不导出 GGML（需额外原生 wheel）
```

→ **主环境不需要任何依赖变更**，`pip install --no-deps faster-qwen3-tts` 即可。

---

## 7. 复现步骤

```powershell
# 1) 克隆独立环境（不要动 memory-tts）
conda create -n memory-tts-bp --clone memory-tts -y

# 2) 只装包体，不动依赖（transformers 仍为 4.57.3）
& "D:\DevVmEnv\Anaconda_envs\envs\memory-tts-bp\python.exe" -m pip install --no-deps "faster-qwen3-tts==0.4.0"

# 3) 运行 backport 基准（脚本会把 .backport 插入 sys.path 优先）
$env:NUMBA_CACHE_DIR = "$PWD\.numba-cache"
& "D:\DevVmEnv\Anaconda_envs\envs\memory-tts-bp\python.exe" .\.backport\bench_backport.py --model .\models\qwen-1.7b
& "D:\DevVmEnv\Anaconda_envs\envs\memory-tts-bp\python.exe" .\.backport\bench_backport.py --model .\models\qwen-0.6b
```

> 注意：`--no-deps` 是刻意的。不要跑 `pip install faster-qwen3-tts`（不带 `--no-deps`），
> 那会把 transformers 抬到 5.x 并拖入 `qwen-tts-hf` 覆盖 `qwen_tts` 模块。

---

## 8. 下一步（尚未执行）

1. **人工试听** `bench/out*/` 的样本，确认音色/韵律/流式拼接（客观指标无法替代）。
2. 把 fast 路径接进 `TtsModelManager`：**必须保持 `app.state.model_lock` 契约（约束 P1）**。
3. 保持现有 `generate()` 的 3 元组返回与 `verify/seed` 语义（约束 A1），
   短/长文本分治（C2/G1）与单词语音 ASR 校验闭环（G3）不得改动。
4. chat 朗读默认切 0.6B；1.7B 保留给带 `instruct` 的听写词库（约束 M1/M2）。
5. 持久化观测：记录 p50/p95 TTFA 与 RTF，防止回归。

---

## 9. 诚实交代的局限

- **未做人工试听**：音质结论仅基于 ASR 回读 + 波形客观量，不能替代人耳。
- **未接入服务**：本报告只验证了独立进程内的性能，未改任何服务代码，
  也未验证在 `model_lock` 与 ASR/OCR 同进程并发下的表现。
- **首次测量偏高未定位**：16–19s 的初测无法复现，已废弃但不掩埋。
- **上游未修**：`transformers>=5.15.1` 的上界问题、4.x 兼容性都还没进上游；
  backport 是我们自己维护的副本，上游升级后需要重新评估（见 `.backport` 内注释）。
- **样本量小**：10 次左右生成、单一句子与单词；warm-up 后稳态之外的冷启动未系统测量。
