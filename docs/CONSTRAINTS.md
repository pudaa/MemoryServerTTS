# MemoryServerTTS 约束与约定文档（CONSTRAINTS）

> 本文件记录项目在演进过程中确立的**必须遵守的架构/模型/接口/性能约束**。
> 修改相关代码前请先通读本文件；新增约束时在此登记。
> 约束编号：`C`=架构、`M`=模型、`G`=生成与校验、`D`=词库缓存、`A`=接口契约、`P`=性能并发、`T`=测试。

---

## C 架构约束

### C1. TTS 服务端不注入任何情绪/风格指令

- **内容**：TTS 服务端**不得**在生成时自行附加 `instructions`（如"中性语调"）。`instruct` 参数完全由业务侧（听写模块、对话模块）传入，服务端只做透传。
- **为什么**：需要什么情绪由**业务场景**决定；服务端无法判断一个短文本（如 `well`）是听写单词还是日常对话中的一句话。这是其他模块的职责。
- **怎么做**：`src/tts/model_loader.py` 的 `generate()` 只使用调用方传入的 `instructions`；`src/dictation/generator.py` 的 `synth_fn` 只透传 `instruct`。
- **违反后果**：跨业务污染情绪风格；听写模块传入的专门指令被覆盖。

### C2. 单词语音判定是服务端唯一的自动分类依据

- **内容**：短/长文本分治的判定**只能**使用 `is_single_word()`（剥离尾部标点后 1 个词、无内部空白、长度 ≤ `verify_text_threshold`）。不得退化为"字符数 ≤ N 即视为单词"。
- **为什么**：字符数阈值会把短句（如 "Hello. This is a speed benchmark test."）误判为单词，使其承受不必要的严格 ASR 校验与确定性解码。
- **违反后果**：短句生成质量降级、延迟增加。

## M 模型约束

### M1. 0.6B 模型不支持 `instruct`（静默丢弃）

- **内容**：qwen_tts 包对 `tts_model_size == "0b6"` 的模型强制 `instruct = None`。带指令的请求在 0.6B 上**不会报错**，但指令被静默丢弃。
- **怎么做**：带 `instruct` 的场景（词库预生成）必须使用 1.7B；0.6B 仅作无指令降级兜底。词库 CLI 通过 `TTSCONF_MODEL_PATH` 强制 1.7B；服务端在 0.6B + instruct 时通过 `X-Dict-Warning` / 日志提示。

### M2. 主模型为 1.7B，0.6B 仅降级

- **内容**：`config/tts.yaml` 默认 `model_path: ./models/qwen-1.7b`，0.6B 是 fallback。早期"0.6B 优先"的文档/配置已废弃。
- **违反后果**：听写指令失效、单词质量下降。

### M3. 短文本失败的根因是语音码 EOS 缺失

- **内容**：模型停止依赖输出 `codec_eos_token_id`；短文本失败时 EOS 不出现，生成会跑满 `max_new_tokens`。因此**单词语音解码必须限制** `max_new_tokens`（默认 512）并提高 `repetition_penalty`（默认 1.2），禁止回退到 `generation_config.json` 的 8192。
- **为什么**：8192 帧（12Hz ≈ 10 分钟量级）会产生长段无意义音节（"嗯嗯啊啊"）。

## G 生成与校验约束

### G1. 短/长文本策略不得互相混淆

- 单词语音：确定性解码（`decoding.short`）+ ASR 校验闭环 + 换 seed 重试（`verification.max_retries`）。
- 短句/长文本：随机采样（`decoding.long`）+ 轻量时长校验 + 重试 `long_max_retries` 次。
- 校验仅在 `verify is not False` 时启用；`verify=true` 强制校验**仅对单词语音生效**。

### G2. ASR 校验只保证内容正确，不保证质量

- **内容**：ASR 回读通过只代表"音频可被识别为目标单词"，不代表风格中性、无噪声、清晰。质量由 `quality_score`（置信度 0.7 + 时长 0.3）刻画，由管理员通过 `/admin` 词库管理反馈迭代。
- **怎么做**：不要因为"ASR 已通过"就认为音频质量达标；词条必须携带 `quality_score` 供管理员判断。

### G3. 宽松匹配是设计决策，不得随意收紧

- **内容**：校验匹配允许去空格子串包含 / 互为子串 / 编辑距离 ≤ 1（容忍 `ahead ↔ "a head" ↔ "Mm-hmm, ahead."`）。
- **为什么**：Whisper 对极短音频的分词/幻觉前缀是常态，严格匹配会造成误杀与无谓重试。
- **怎么做**：如需收紧，须先评估误杀率并更新 `tests/test_tts_verifier.py`。

## D 词库缓存约束

### D1. 缓存 key 必须包含 instruct 与 gen_config_version

- **内容**：`cache_key = hash(word | voice | language | instruct | gen_config_version)`。不同指令（不同情绪风格）= 不同音频，**不得共用条目**；`gen_config_version` 对生成配方（模型路径 + `decoding.short` + 校验参数 + 布局版本）自动哈希。
- **怎么做**：新增影响生成结果的参数时，必须纳入 key 或 `gen_config_version` 的哈希输入。

### D2. 配置变更即换代，禁止手动清缓存绕行

- **内容**：修改 `decoding.*` / `verification.*` / `model_path` 后，`gen_config_version` 变化，旧条目自然 miss 并按需重生成。**不要**通过手动删除 `word-cache/` 来"处理"缓存问题。
- **为什么**："升级即换代"是有意的机制设计；手动清理会破坏增量生成与审计。

### D3. 听写接口宁缺毋滥

- **内容**：`/api/v1/dictation/audio` 与预生成中，若所有候选均未通过校验 → 返回 `502` 且**不入缓存**。
- **为什么**：坏音频沉淀进缓存比请求失败更糟；听写场景的正确性承诺是产品底线。

### D4. 缓存写入必须原子

- **内容**：先写 `<key>.wav.tmp` / `<key>.json.tmp`，再 `os.replace`；`soundfile.write` 必须显式 `format="WAV"`（旧版 soundfile 无法从 `.wav.tmp` 后缀推断格式）。
- **为什么**：并发请求不得读到半个文件。

### D5. 条目元数据字段

- 每条目 JSON 必须含：`key / word / voice / language / instruct / gen_config_version / seed / verified / asr_text / avg_logprob / quality_score / duration / generated_at / served_count / bad_flags / bad_reason`。
- `bad_flags` 非空时该条目**不再对外服务**；重新生成成功（`save` 覆盖）自动清除 bad 标记。

## A 接口契约约束

### A1. 向后兼容

- `/api/v1/tts/synthesize` 默认返回 WAV 流（新增参数 `verify/seed/include_meta` 均为可选）；校验信息放 `X-TTS-*` 响应头；`include_meta=true` 才返回 JSON。
- `generate()` 返回 3 元组 `(wavs, sr, meta)`——**所有调用点必须适配 3 元组**，不得解包 2 元组。

### A2. 音色-语言匹配

- `voice` 留空时按 `language` 自动匹配母语音色（English→aiden、Chinese→vivian、Japanese→ono_anna、Korean→sohee）；`/api/v1/tts/voices` 列表必须与模型 README 官方清单一致（Ryan/Aiden 英文、Ono_Anna 日文女声等），**不得**自行杜撰音色或错标语言。

### A3. 音频 URL

- `/tts-audio/*` 静态挂载依赖服务启动时 mount；新增返回 `audioUrl` 的接口必须确认对应目录已挂载。

## P 性能与并发约束

### P1. 所有 TTS/词库生成必须持有 model_lock

- 服务内任何调用 `generate_custom_voice` 的路径（在线合成、单词语音校验、词库生成/重生成/后台任务）都必须在 `app.state.model_lock` 内执行（ASR 校验也在锁内，防 GPU 并发 OOM）。
- 后台批量任务**逐词获取/释放锁**（每个词生成后 `await asyncio.sleep(0)` 让出事件循环），避免饿死在线请求。

### P2. 大批量预生成用 CLI，不用服务内后台任务

- 词库预生成首选 `python -m src.dictation.pregenerate`（独立进程，不占在线 GPU/锁）；管理页批量预生成仅用于小批量（几十词以内）。

## T 测试约束

### T1. 校验/判定逻辑变更必须更新并跑通测试

- `tests/test_tts_verifier.py`（19 用例）、`tests/test_dictation_cache.py`（21 用例）、`tests/test_tts_model_loader.py`（10 用例）覆盖校验匹配、缓存、单词语音判定。
- 运行：`python -m unittest discover -s tests -p "test_*.py"`（约 6 秒，无需 GPU）。

### T2. 新约束登记

- 本文件与代码同步演进；新增影响架构/接口/缓存的决策时，在本文件登记并更新 `docs/API_DOCUMENTATION.md` 与 `docs/PROJECT_DOCUMENTATION.md` 对应章节。
