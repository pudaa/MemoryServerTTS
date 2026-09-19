# 三端 TTS 调用链分析与流式改造方案

> 2026-09-19。分析 `Memory`(Android) → `MemoryServer`(Spring Boot) → `MemoryServerTTS`(FastAPI/Python) 的
> 对话朗读链路，定位"一直想做流式播放但做不成"的真实原因，并给出分阶段改造方案。
>
> **前置安全确认**：改造前三个仓库均已确认干净并全部推送（详见 §0）。

---

## 0. 改造前的仓库安全状态（已处理）

| 仓库 | 分支 | 改造前状态 | 处理 |
|---|---|---|---|
| `D:\Codes\MemoryServerTTS` | master | **10 个未推送提交**（本次会话的调研/优化工作） | ✅ 已 `git push` 到 origin |
| `D:\Codes\MemoryServer` | master | 干净，与 origin/master 一致，0 ahead | 无需操作 |
| `D:\Codes\Memory` | master | **36 个未推送提交**（含 216MB LFS 对象）+ 干净工作区 | ✅ 已 `git push` 到 origin |

> Memory 那 36 个提交是真实的丢失风险（本地有、远端没有，且无其他分支/暂存）。
> 现已推送，`git rev-list --left-right --count @{u}...HEAD` 三个仓库均为 `0 0`。

---

## 1. 当前调用链（对话朗读，逐跳）

```
[Android] AiConversationActivity.sendStreamingMessage()
   │  POST /conversation/stream        (SSE, okhttp postStream)
   ▼
[MemoryServer] ConversationController.streamMessage()          ← SseEmitter
   │  1) 语音/文本输入 → userInput
   │  2) aiService.streamChatWithTools(..., chunkCb, doneCb)
   │       chunkCb → SSE event:chunk {text}          （文本边说边到）
   │       doneCb  →
   │         ├─ 保存 assistant 消息
   │         ├─ 评估 → SSE event:eval
   │         ├─ ★ generateConversationAudioAsync()   （@Async，放线程池）
   │         └─ SSE event:done {messageId, audioPending:true}
   ▼
[MemoryServer] ConversationServiceImpl.generateConversationAudioAsync()
   │  ★ 注释原文：「前端无法流式播放，直接使用非流式 TTS 生成完整音频（避免流式 SSE 超时）」
   │  ttsService.textToSpeechWithEmotion(text, emotion, "Ono_Anna", "English")
   ▼
[MemoryServer] TTSServiceImpl.textToSpeechWithEmotion()
   │  POST {tts.server.url}/api/v1/tts/synthesize   （RestTemplate，阻塞）
   │  读超时 120s；整段 wav 一次性读入 byte[] → 落盘 → 返回 URL
   ▼
[MemoryServerTTS] POST /api/v1/tts/synthesize
   │  model.generate() → 整段音频 → soundfile 写完整 wav → 返回文件
   ▼
[MemoryServer] 落盘 → 记 message.audio_url
   │
[Android] startAudioPolling(messageId)
   │  每 1500ms 轮询 GET /conversation/audio/{messageId}，最多 80 次（2 分钟）
   │  拿到 audioUrl 后 → MemoryApiClient.downloadMediaFile() 整文件下载到本地
   │  → MediaPlayer.setDataSource(本地文件) → prepareAsync → start
   ▼
        用户此时才第一次听到声音
```

### 关键事实（均已核实）

| 事实 | 位置 |
|---|---|
| MemoryServerTTS **已经有** `/api/v1/tts/stream`（SSE，逐句推 `audioUrl`） | `src/tts/router.py:100` |
| 但**没有任何调用方使用它**（三端全仓库搜索 `tts/stream` 无结果） | 搜索确认 |
| MemoryServer 明确以"前端无法流式播放"为由放弃流式 | `ConversationServiceImpl.java:269` 注释 |
| Java 侧把**整段 wav 读进 `byte[]`**，无 chunked 转发 | `TTSServiceImpl.java:89-98` |
| Android 播放器只接受**本地文件路径** | `AiConversationAdapter.java:152` |
| `/tts-audio/**` 已强制 JWT → MediaPlayer 不能直接带头播 | `MemoryApiClient.java:353-357` 注释 |

---

## 2. 根因：四道串联的"墙"

流式做不成不是单点问题，是**四道墙叠在一起**，任何一道不拆都通不了：

| # | 墙 | 后果 |
|---|---|---|
| **W1 服务端整段生成** | Python 端 `generate()` 一次性返回整段 wav | 没有可流式的产物 |
| **W2 中间层整段缓冲** | Java 用 `byte[]` 收完再落盘，且**只看 200 状态** | 即使下游流式也被拉平 |
| **W3 鉴权让播放器无法直连** | `/tts-audio/**` 强制 JWT，`MediaPlayer.setDataSource(url)` 无法带 Authorization 头 | 必须"先下载到本地再播"→ 完整下载屏障 |
| **W4 客户端播放模型是"文件级"** | 适配器只接受本地文件路径，无增量/边下边播 | 必须等文件完整 |

加上 §3 的**语义错配**，共同造成：**79 秒的音频要等 34–43 秒全部生成完、再等完整下载，用户才开始听。**

---

## 2.5 ★ 决定性实测：TTFA 与文本长度无关（推翻了"按句切分"的必要性）

脚本 `bench/stream_ttfa.py`（0.6B + CUDA Graph，`chunk_size=8`，每片约 640ms 音频）：

| 文本 | `non_streaming_mode` | **TTFA** | 首片音频 | 总耗时 | 音频时长 |
|---|---|---|---|---|---|
| 198 词整段 | True | 1103 / **332** ms | 0.64 s | 35.2 / 33.8 s | 77.5 s |
| 198 词整段 | False | **340 / 333** ms | 0.64 s | 30.1 / 31.8 s | 68.3 s |
| 仅第一句 | True | **331 / 334** ms | 0.64 s | 1.9 / 1.6 s | 4.2 s |
| 仅第一句 | False | **331 / 332** ms | 0.64 s | 1.7 / 1.9 s | 3.7 s |

**结论：TTFA 恒为 ~330ms，与文本是 1 句还是 198 词无关，与
`non_streaming_mode` 也无关。**

这有两个重要含义：

1. **首片延迟不是问题**——Python 侧**现在就能**在 0.33 秒内吐出第一块 640ms 音频。
2. **"按句切分"并没有首片收益**——因为首片延迟本来就不受文本长度影响。
   按句切分只会换来更多次请求往返，**反而增加总开销**。

> 所以真正卡住流式的**不是 Python 的生成方式**，而是
> **W1/W2/W3/W4 四道墙**（尤其 W2：Java 侧整段缓冲 + 只看 200 状态）。
> 换句话说：**瓶颈在中间层与客户端，不在模型服务。**

### 由此修正方案优先级

- **Step 1 改为"音频流式透传"**（而不是"按句分片"）：
  Python 侧新增一个**流式音频响应**端点（边生成边下发），
  让首片 640ms 在 ~0.33s 就能离开 Python 进程。
- 按句分片降级为**可选优化**（若客户端需要"句级"控制或按句缓存才需要）。

---

## 3. 一个容易被忽略的语义错配

- `/api/v1/tts/stream` 返回的是 **SSE（`text/event-stream`）**，每句一条事件，事件里是 **`audioUrl`**。
- 也就是说：它**不是音频流**，而是"**音频分片就绪通知**"。

因此**即使现在直接接上它**，客户端仍然要：收到 URL → 再发一次 HTTP 请求 → 下载该句 → 才能播。
**每句一次往返**，且仍受 W3（鉴权、不能直连播放）影响。

→ **只把 `/tts/stream` 接起来是不够的**，必须同时解决"音频如何到达播放器"。
（且由 §2.5 可知，这条"按句通知"路线比直接透传音频流**更慢**。）

---

## 4. 可选方案对比

| 方案 | 做法 | 首声延迟 | 改动面 | 风险 |
|---|---|---|---|---|
| **A. 分片音频 + 通知** | Python 逐句写独立 wav；Java 侧一个 SSE 端点把"分片就绪"推给客户端；Android 按序边收边播 | ~0.4–1.5s | 中 | 低（复用现有轮询思路） |
| **B. 服务端 chunked 音频流** | Java 以 `StreamingResponseBody` 转发 Python 的原始音频流，客户端边收边播 | ~0.4s | 大 | 中（需解决 WAV 头/时长未知） |
| **C. WebSocket 推音频** | 复用 Python 侧已有 WS 思路，Java 做代理，Android 用 AudioTrack 播 PCM | ~0.4s | 最大 | 高（三端都要新写通道） |
| **D. 只改播放链路** | 保持整段生成，只做"边下边播" | 34s+ | 小 | 低但**不解决核心问题** |

### 技术约束（必须选方案前定清楚）

1. **WAV 头问题**：标准 WAV 头部要写 `data` 块总长度。若边生成边下发，
   长度未知 → 必须用**流式 wav**（长度字段置 `0xFFFFFFFF`）或改用
   **裸 PCM + 客户端已知采样率**，或**每片一个自包含 wav**。
   `MediaPlayer` 对"长度未知的流式 wav"支持并不可靠，**ExoPlayer 的 progressive 更稳**。
2. **JWT 与播放器**：直连播放需要 `?token=` 查询参数鉴权，或客户端把 Authorization
   注入播放器的请求头（MediaPlayer 支持 `setDataSource(Context, Uri, Map<String,String>)`）。
3. **两个 WAV 拼接**：简单 `cat` 两个 wav 会产生"中间夹一个 RIFF 头"的损坏文件，
   **不能直接拼**；必须自行剥离头再拼 PCM。

---

## 5. 推荐的实施顺序（每步可独立验证、可回滚）

**目标：把"用户等 34–43 秒才开始听"变成"约 1 秒内开始听"，且不破坏现有非流式路径。**

由 §2.5 的实测，**首片 0.33s 在 Python 侧已经现成**，所以工作的重心是
**打通 W2（Java 缓冲）/ W3（鉴权直连）/ W4（客户端文件级播放）**。

1. **Step 1（Python）**：新增**流式音频响应**端点（如 `POST /api/v1/tts/synthesize-stream`），
   以 chunked 方式边生成边下发音频。需决定编码：
   - **裸 PCM + 固定采样率**（最简单、客户端最好处理，但要带元数据头说明 sr/位深/声道）
   - **流式 WAV**（`data` 长度置 `0xFFFFFFFF`）——`MediaPlayer` 支持不可靠，ExoPlayer 较稳
   - 保留现有 `/api/v1/tts/synthesize` 整段接口**不动**（向后兼容）
2. **Step 2（Java）**：新增流式转发端点（`StreamingResponseBody`），
   边读 Python 流边写给客户端；**不要**再读成 `byte[]`。
   同时保留 `/conversation/audio/{messageId}` 轮询接口与
   `generateConversationAudioAsync` 整段路径作为**降级**。
3. **Step 3（鉴权直连，解决 W3）**：给流式音频端点提供
   **播放器可用的鉴权方式**（`?token=` 短时票据，或让客户端注入 `Authorization` 头到播放器）。
   注意安全：票据需短时效 + 绑定用户 + 不可重用。
4. **Step 4（Android）**：新增流式播放路径。
   优先评估 **ExoPlayer progressive**（对"长度未知的流"支持好、可缓冲续播）；
   若不想引入依赖，则退回"顺序分片文件 + 队列播放"。
   现有整段播放路径保留为降级。
5. **Step 5（音频管理）**：见 §6。
6. **Step 6**：端到端压测 + 固化提交（三仓库分别提交）。

> **为什么先做 Step 1**：它完全在本次已验证过的 Python 项目内，
> 风险最低，且能立刻用 `curl` 验证"首片 0.33s 到达"。
> Java/Android 在 Step 1 验证通过前不应改动。

---

## 6. 流式化之后的音频管理（必须一起设计）

现在音频是"一个消息一个文件"，流式后会变成"一个消息 N 个分片"，需要明确：

| 议题 | 现状 | 流式后需要 |
|---|---|---|
| **命名与归属** | `tts_<时间戳>_<uuid8>.wav` | 建议 `<messageId>/<seq:03d>.wav`，天然按消息归组 |
| **生命周期** | 根目录 wav 保留 `tts.audio.max-age-days:7` 天，`words/` 子目录不清理 | 按**消息目录**整体清理；需保证"正在播放"不被删 |
| **清理时机** | `cleanupExpiredAudioFiles()` 按文件 mtime，仅根目录、不递归 | 改为按目录 mtime 递归；**保留未播完/未读消息的音频** |
| **失败与清理** | 生成失败则无 URL，消息永久 `audioPending` | 需要"部分成功"语义：已生成的分片可播，失败分片要能标记/重试 |
| **去重与缓存** | 每次生成新文件 | 同文本+情感+音色是否复用？（注意约束 D1：缓存 key 需含 instruct 与配置版本） |
| **存储增长** | 每消息 1 文件 | 每消息 N 文件，**小文件数量显著上升**；需评估 inode/目录扫描开销 |
| **并发** | 线程池 core=2/max=4，但 Python 侧 `model_lock` 串行 | 多用户同时流式会**争抢串行 GPU**；需排队与背压策略 |
| **播放中断** | 新播放打断旧的 | 分片队列需要"打断即清空队列并释放" |
| **客户端缓存** | 下载到 `externalFilesDir/Audio`，无清理 | 分片会快速堆积，需 LRU 或播完即删 |

**特别注意（并发与背压）**：Python 侧所有生成都在 `app.state.model_lock` 内串行
（约束 P1）。多用户同时朗读时，后到者必然排队。
流式化会让"排队中"变得可见，**必须设计超时与降级**（例如排队过久则退化为只播前 N 句）。

---

## 7. 待确认的决策点

1. **播放通道选型**：ExoPlayer progressive（需引入依赖）vs MediaPlayer + 顺序分片文件（无新依赖）。
2. **鉴权方式**：`?token=` 查询参数 vs 播放器请求头注入。
3. **分片粒度**：按句（自然、停顿好）vs 按固定帧数（延迟更平滑、可能有句中切断）。
4. **是否需要"打断/续播"**：用户中途发新消息时，旧音频立即停还是播完当前句。
5. **Android 播放器**：`AiConversationAdapter` 目前用 `MediaPlayer` 播本地文件，
   改成队列播放需要动 adapter 的状态机（当前 `releaseMediaPlayer()` 由 Activity 生命周期调用）。

---

## 8. 本次分析未做的事（诚实交代）

- **没有修改任何三个项目的代码**，本文档只是分析与方案。
- 未实机端到端跑通现有链路（没有启动 Java/Python 服务、未连真机），
  调用链结论来自**源码阅读**而非运行时验证。
- 未评估 ExoPlayer 引入对 Android 包体积/依赖的影响。
- 未做多用户并发与背压实测。
- 音频管理方案（§6）是设计建议，**尚未实现**。

---

## 9. ★ Step 1 已实现并验证（2026-09-19 更新）

§1–§8 是方案分析；本节记录**实际落地的 Step 1** 与实测结果。

### 9.1 落地内容（仅 MemoryServerTTS，未动另外两个项目）

| 文件 | 改动 |
|---|---|
| `src/tts/router.py` | 新增 `POST /api/v1/tts/synthesize-stream`：chunked 下发**裸 PCM** |
| `src/tts/model_loader.py` | 新增 `generate_stream()`；新增 `backend` 支持（upstream/faster）；新增 `_load_faster()`；`_prepare_text()` 抽出共用文本规范化；新增 `get_supported_speakers/languages` 委托 |
| `src/tts/config.py` | 新增 `backend`、`stream_chunk_steps`、`stream_max_new_tokens`、`stream_max_seq_len` |
| `config/tts.yaml` | 新增 `tts.backend`、`tts.streaming.*` |
| `src/server.py` | startup 时写入 `app.state.stream_loop`（供跨线程回传分片） |

**响应契约**

```
POST /api/v1/tts/synthesize-stream
  {"text": "...", "voice": "aiden", "language": "English", "instructions": null,
   "chunk_steps": 8, "max_new_tokens": 4096}

200 OK
  Content-Type: audio/L16;rate=24000;channels=1
  Transfer-Encoding: chunked
  X-Audio-Sample-Rate: 24000
  X-Audio-Channels: 1
  X-Audio-Bits: 16
  X-Audio-Format: pcm_s16le
  X-TTS-TTFA-Ms: <服务端实测首片耗时>
  body: 原始 int16LE 单声道 PCM（无头、无时长）
```

### 9.2 实测结果（真实 HTTP + chunked，非 TestClient）

用裸 socket 手工解析 chunked 响应（本机 httpx/requests 受代理环境变量影响解析失败）：

| 文本 | 服务端 TTFA | **客户端首字节** | 分片数 | 音频 | 总耗时 | RTF_wall |
|---|---|---|---|---|---|---|
| 短句（23 字符，首次请求） | 848 ms | 861 ms | 3 | 1.92 s | 6.46 s | — |
| 长文本（110 词，首次请求） | 517 ms | **534 ms** | **55** | 34.88 s | 21.11 s | **0.605×** |
| 短句（二次，已热） | 630 ms | 635 ms | 4 | 2.08 s | 6.35 s | — |
| 长文本（二次，已热） | 603 ms | **620 ms** | **56** | 35.52 s | 21.52 s | **0.606×** |

**结论**：
- ✅ **真正在流**：长文本产生 55–56 个 HTTP chunk，**不是攒完再发**。
- ✅ **首字节 ~0.53–0.86s**（离线直测首片 330ms，HTTP 层额外约 200–300ms）。
- ✅ **RTF_wall 0.605×**（生成快于播放），播放不会追不上生成。
- ✅ PCM 校验：int16 范围内、非全零、无触顶爆表。
- ✅ 空文本返回 400。

### 9.3 两个必须知道的坑（已实测确认）

1. **`FastAPI TestClient` 会缓冲整个响应** → 用它测流式会得到"首片=总耗时"的**假象**。
   验证流式必须用**真实 HTTP 服务器 + 流式客户端**（本仓库 `bench/verify_stream_endpoint.py
   --serve` / `--probe-url` 即为这种两进程验证方式）。
2. **流式路径有独立冷启动**：首次流式请求 TTFA 848ms，热后 ~600ms。
   底层 CUDA Graph 的**流式**解码路径与 `warmup()` 覆盖的路径不同，
   建议服务启动时**额外预热一次流式**（本步尚未实现，见 §10）。

### 9.4 向后兼容与安全

- `tts.backend` **默认 `upstream`**，即**默认行为不变**；需显式设置才启用 CUDA Graph。
- `/api/v1/tts/synthesize`、`/stream`、`/voices`、词库接口**均未改动**。
- upstream 后端下调用新端点**不会报错**，但只产出 1 片（整段），
  日志会明确警告"当前后端不支持真正的流式生成"——不会假装在流。
- 既有 50 个单测**全部通过**（改动未破坏校验/缓存/判定逻辑）。
- 全程持有 `app.state.model_lock`（约束 P1），生成结束/客户端断开即释放。

---

## 10. 下一步

1. **Step 2（Java 透传）**：`TTSServiceImpl` 增加流式转发，
   `StreamingResponseBody` 边读 Python 流边写客户端；保留现有整段路径作为降级。
2. **Step 3（鉴权直连）**：短时票据或播放器注入 Authorization（解决 W3）。
3. **Step 4（Android）**：引入 ExoPlayer 播 `audio/L16` PCM（解决 W4）。
4. **启动预热流式路径**（§9.3 第 2 点）。
5. **音频管理**（§6）：流式后不再落盘整段 wav，需重新定义音频保留策略。
