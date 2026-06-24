# MemoryServerTTS API 集成文档

> **项目概述**：MemoryServerTTS 是一个基于 **Qwen3-TTS**、**Faster-Whisper** 和 **PaddleOCR** 的语音与视觉服务，提供文本转语音（TTS）、语音识别（ASR）、发音评价和 OCR 文字识别四大核心功能，适用于英语学习等场景。
>
> **适用场景**：SpringBoot 后端服务通过 HTTP RESTful API 或 WebSocket 接入本服务，实现语音合成、语音转录、发音评分等功能。

---

## 目录

1. [快速启动](#1-快速启动)
2. [技术架构](#2-技术架构)
3. [API 总览](#3-api-总览)
4. [TTS 文本转语音](#4-tts-文本转语音)
   - [4.1 合成语音（REST）](#41-合成语音-rest)
   - [4.2 流式合成（WebSocket）](#42-流式合成-websocket)
   - [4.3 获取音色列表](#43-获取音色列表)
   - [4.4 音色克隆（模拟）](#44-音色克隆模拟)
5. [ASR 语音识别](#5-asr-语音识别)
   - [5.1 转录音频](#51-转录音频)
   - [5.2 获取支持的 ASR 模型](#52-获取支持的-asr-模型)
6. [发音评价](#6-发音评价)
   - [6.1 MFCC+DTW 发音评分（需参考音频）](#61-mfccdtw-发音评分需参考音频)
   - [6.2 批量发音评分（MFCC+DTW）](#62-批量发音评分mfccdtw)
   - [6.3 音素对齐发音评分（仅需参考文本，推荐）](#63-音素对齐发音评分仅需参考文本推荐)
   - [6.4 批量音素评分](#64-批量音素评分)
7. [系统接口](#7-系统接口)
   - [7.1 健康检查](#71-健康检查)
   - [7.2 管理后台](#72-管理后台)
8. [OCR 文字识别](#8-ocr-文字识别)
   - [8.1 图片文字扫描](#81-图片文字扫描)
   - [8.2 文档文字扫描](#82-文档文字扫描)
   - [8.3 OCR 健康检查](#83-ocr-健康检查)
9. [全局错误处理](#9-全局错误处理)
10. [SpringBoot 集成示例](#10-springboot-集成示例)
11. [配置与环境变量](#11-配置与环境变量)
12. [注意事项](#12-注意事项)

---

## 1. 快速启动

### 1.1 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.0.1+（推荐 GPU 版，CUDA 11.8+） |
| 显存 | 全部模块驻留约需 5.5 GB（1.7B TTS + ASR + OCR） |
| 磁盘 | 模型文件约 5-12 GB |

### 1.2 安装与启动

```bash
# 1. 创建虚拟环境（推荐 conda）
conda create -n memory-tts python=3.12
conda activate memory-tts

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载模型（已预置在 models/ 目录下）
#    支持两种模型：
#    - models/qwen-1.7b/  (推荐，效果更好)
#    - models/qwen-0.6b/  (轻量，显存需求低)

# 4. 启动服务
python src/server.py
# 或（推荐，支持热重载）
python main.py
# 或
uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

服务默认启动在 **`http://0.0.0.0:8000`**。

> **热重载**：`main.py` 通过环境变量 `RELOAD=1` 控制热重载模式，默认关闭。调试时可执行 `set RELOAD=1 && python main.py`（Windows）或 `RELOAD=1 python main.py`（Linux/macOS）。

### 1.3 Docker 启动

```bash
docker build -t memory-tts .
docker run -d -p 8000:8000 --gpus all memory-tts
```

---

## 2. 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                     SpringBoot 服务端                          │
│  (通过 HTTP REST / WebSocket 调用 TTS 服务)                   │
└─────────────┬───────────────────────────┬───────────────────┘
              │                           │
        ┌─────▼─────┐              ┌──────▼──────┐
        │  REST API  │              │  WebSocket  │
        │ (8000/tcp) │              │  (8000/tcp) │
        └─────┬─────┘              └──────┬──────┘
              │                           │
┌─────────────▼───────────────────────────▼───────────────────┐
│                     FastAPI 服务 (server.py)                   │
│                                                              │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────┐  │
│  │TTSModelManager│ │ASRModelManager│ │Pronun.Eval│ │Phoneme │  │
│  │ (Qwen3-TTS)   │ │(Faster-Whisper)│ │(MFCC+DTW)│ │Eval    │  │
│  └──────┬───────┘ └──────┬───────┘ └──────────┘ │(G2P+ASR)│  │
│         │                │                       └────┬───┘  │
│  ┌──────▼────────────────▼────────────────────────────▼─────┐ │
│  │        PyTorch (CUDA/CPU) + g2p-en / pypinyin           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 技术方案 | 说明 |
|------|---------|------|
| **TTS** | Qwen3-TTS (`qwen_tts`) | 阿里通义千问3代语音合成模型，支持多语言、多音色、情感指令 |
| **ASR** | Faster-Whisper | 基于 CTranslate2 的 Whisper 加速版，支持转录/翻译 |
| **发音评价（旧）** | MFCC + DTW (`librosa` + `fastdtw`) | 基于声学特征对齐，需参考音频 |
| **发音评价（新）** | G2P + ASR + 音素对齐 | 基于音素语义层比对，仅需参考文本 |
| **G2P** | `g2p-en`（英文）/ `pypinyin`（中文） | 文字→音素转换 |
| **Web 框架** | FastAPI + uvicorn | 高性能异步 Python Web 框架 |
| **调试 UI** | Gradio | 提供可视化调试界面（端口 7860） |

---

## 3. API 总览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/health` | 健康检查 |
| `GET` | `/api/v1/tts/voices` | 获取所有可用音色 |
| `POST` | `/api/v1/tts/synthesize` | 文本合成语音（返回 WAV 文件） |
| `WebSocket` | `/api/v1/tts/stream` | 流式语音合成 |
| `POST` | `/api/v1/tts/clone` | 音色克隆（当前为模拟实现） |
| `POST` | `/api/v1/asr/transcribe` | 音频文件转录 |
| `GET` | `/api/v1/asr/models` | 获取支持 ASR 模型列表 |
| `POST` | `/api/v1/pronunciation/score` | 发音评分（MFCC+DTW，需参考音频） |
| `POST` | `/api/v1/pronunciation/batch-score` | 批量发音评分（MFCC+DTW） |
| `POST` | `/api/v1/pronunciation/phoneme-score` | 🔥 音素对齐发音评分（仅需参考文本） |
| `POST` | `/api/v1/pronunciation/phoneme-score-with-text` | 与 phoneme-score 接口相同（便于兼容测试） |
| `POST` | `/api/v1/pronunciation/phoneme-batch-score` | 批量音素评分（需音频在服务端） |

### 基础 URL

```
http://<server-ip>:8000
```

所有接口统一前缀为 `/api/v1/`。

---

## 4. TTS 文本转语音

### 4.1 合成语音（REST）

`POST /api/v1/tts/synthesize`

将文本合成为语音，返回 WAV 格式的音频文件。

#### 请求体 (JSON)

```json
{
  "text": "Hello, welcome to Memory English Learning App!",
  "voice": "Ono_Anna",
  "language": "English",
  "instructions": "Speak with a happy and encouraging tone.",
  "output_format": "wav"
}
```

#### 参数字段说明

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的文本内容 |
| `voice` | string | ❌ | `"Ono_Anna"` | 音色 ID（详见 4.3 节） |
| `language` | string | ❌ | `"English"` | 语言，支持 `English`、`Chinese`、`Japanese`、`Korean` 等 |
| `instructions` | string | ❌ | `null` | 情感/风格指令，如 `"Speak with a happy tone."` |
| `output_format` | string | ❌ | `"wav"` | 输出格式，当前仅支持 `"wav"` |

#### 成功响应

- **状态码**: `200 OK`
- **Content-Type**: `audio/wav`
- **响应体**: 二进制 WAV 音频数据（文件名格式：`tts_<hex>.wav`）

#### 错误响应

```json
{
  "detail": "Only wav output is supported currently."
}
```

可能的状态码：`400`（参数错误）、`500`（服务内部错误）。

#### SpringBoot 调用示例

```java
// 使用 RestTemplate
RestTemplate restTemplate = new RestTemplate();
String url = "http://localhost:8000/api/v1/tts/synthesize";

// 构造请求
HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.APPLICATION_JSON);

JSONObject requestBody = new JSONObject();
requestBody.put("text", "Hello, welcome to Memory English Learning App!");
requestBody.put("voice", "Ono_Anna");
requestBody.put("language", "English");
requestBody.put("instructions", "Speak with a happy and encouraging tone.");

HttpEntity<String> request = new HttpEntity<>(requestBody.toString(), headers);

// 发送请求，接收音频文件
ResponseEntity<byte[]> response = restTemplate.postForEntity(url, request, byte[].class);
byte[] audioData = response.getBody();

// 保存为 WAV 文件
Files.write(Paths.get("output.wav"), audioData);
```

```java
// 使用 WebClient (Reactive)
WebClient webClient = WebClient.create("http://localhost:8000");
byte[] audioBytes = webClient.post()
    .uri("/api/v1/tts/synthesize")
    .contentType(MediaType.APPLICATION_JSON)
    .bodyValue(new TTSRequest("Hello", "Ono_Anna", "English", null, "wav"))
    .retrieve()
    .bodyToMono(byte[].class)
    .block();
```

---

### 4.2 流式合成（WebSocket）

`WebSocket /api/v1/tts/stream`

适用于长文本或实时性要求高的场景，通过 WebSocket 分块发送文本，服务端返回 PCM16 编码的音频块。

#### 连接地址

```
ws://localhost:8000/api/v1/tts/stream
```

#### 客户端 -> 服务端消息格式

**文本块消息**：
```json
{
  "type": "text_chunk",
  "data": "Hello, welcome to Memory English Learning App!",
  "voice": "Ono_Anna",
  "language": "English",
  "instructions": "Speak with a happy and encouraging tone."
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | ✅ | 固定为 `"text_chunk"` |
| `data` | string | ✅ | 要合成的文本 |
| `voice` | string | ❌ | 音色 ID，默认 `"Ono_Anna"` |
| `language` | string | ❌ | 语言，默认 `"English"` |
| `instructions` | string | ❌ | 情感指令 |

**结束消息**：
```json
{
  "type": "end"
}
```

发送此消息后，服务端会返回 `end_of_stream` 并关闭连接。

#### 服务端 -> 客户端消息格式

**音频块消息**：
```json
{
  "type": "audio_chunk",
  "sample_rate": 24000,
  "format": "pcm16",
  "data": "<base64_encoded_pcm_bytes>"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `"audio_chunk"` |
| `sample_rate` | int | 音频采样率（如 24000 Hz） |
| `format` | string | 音频格式，当前为 `"pcm16"`（16-bit 有符号整数 PCM） |
| `data` | string | Base64 编码的 PCM 音频数据 |

**流结束消息**：
```json
{
  "type": "end_of_stream"
}
```

**错误消息**：
```json
{
  "type": "error",
  "message": "Unsupported message type"
}
```

#### SpringBoot WebSocket 客户端示例

```java
import org.springframework.web.socket.*;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import java.net.URI;
import java.util.Base64;
import javax.sound.sampled.*;

public class TtsWebSocketClient {

    public static void main(String[] args) throws Exception {
        StandardWebSocketClient client = new StandardWebSocketClient();
        WebSocketSession session = client.doHandshake(
            new WebSocketHandler() {
                @Override
                public void afterConnectionEstablished(WebSocketSession session) {
                    System.out.println("WebSocket connected.");

                    // 发送文本块
                    String message = """
                        {
                            "type": "text_chunk",
                            "data": "Hello, welcome to Memory English Learning App!",
                            "voice": "Ono_Anna",
                            "language": "English",
                            "instructions": "Speak with a happy tone."
                        }
                        """;
                    session.sendMessage(new TextMessage(message));

                    // 发送结束信号
                    session.sendMessage(new TextMessage("{\"type\": \"end\"}"));
                }

                @Override
                public void handleMessage(WebSocketSession session, WebSocketMessage<?> message) {
                    String payload = (String) message.getPayload();
                    // 解析 JSON...
                    // 如果是 audio_chunk，解码 data 字段得到 PCM 音频数据
                }

                @Override
                public void handleTransportError(WebSocketSession session, Throwable exception) {
                    exception.printStackTrace();
                }

                @Override
                public void afterConnectionClosed(WebSocketSession session, CloseStatus closeStatus) {
                    System.out.println("Connection closed: " + closeStatus);
                }
            },
            new WebSocketHttpHeaders(),
            URI.create("ws://localhost:8000/api/v1/tts/stream")
        ).get();
    }
}
```

> **PCM 转 WAV 播放说明**：SpringBoot 端收到 Base64 PCM 数据后，需要自行拼接 WAV 头部或直接使用 `javax.sound.sampled.AudioSystem` 播放。PCM 格式为：16-bit 有符号小端序（signed 16-bit little-endian）、单声道。

---

### 4.3 获取音色列表

`GET /api/v1/tts/voices`

返回所有预设音色和已克隆的自定义音色。

#### 请求参数

无。

#### 响应示例

```json
{
  "voices": [
    {
      "id": "vivian",
      "name": "Vivian",
      "gender": "Female",
      "language": "Chinese",
      "desc": "明亮、略带锋芒的年轻女性声音",
      "type": "preset"
    },
    {
      "id": "serena",
      "name": "Serena",
      "gender": "Female",
      "language": "Chinese",
      "desc": "温暖、温柔的年轻女性声音",
      "type": "preset"
    },
    {
      "id": "uncle_fu",
      "name": "Uncle_Fu",
      "gender": "Male",
      "language": "Chinese",
      "desc": "经验丰富的男性嗓音，音色低沉柔和",
      "type": "preset"
    },
    {
      "id": "dylan",
      "name": "Dylan",
      "gender": "Male",
      "language": "Chinese (Beijing Dialect)",
      "desc": "年轻的北京男性嗓音，音色清晰自然",
      "type": "preset"
    },
    {
      "id": "eric",
      "name": "Eric",
      "gender": "Male",
      "language": "Chinese (Sichuan Dialect)",
      "desc": "活泼的成都男声，带着一丝沙哑明亮",
      "type": "preset"
    },
    {
      "id": "Ono_Anna",
      "name": "Ono_Anna",
      "gender": "Male",
      "language": "English",
      "desc": "充满活力的男性声音，节奏感强劲",
      "type": "preset"
    },
    {
      "id": "aiden",
      "name": "Aiden",
      "gender": "Male",
      "language": "English",
      "desc": "阳光的美国男声，中音清晰",
      "type": "preset"
    },
    {
      "id": "ono_anna",
      "name": "Ono_Anna",
      "gender": "Female",
      "language": "Japanese",
      "desc": "活泼的日本女性声音，音色轻盈灵巧",
      "type": "preset"
    },
    {
      "id": "sohee",
      "name": "Sohee",
      "gender": "Female",
      "language": "Korean",
      "desc": "温暖的韩国女性声音，情感丰富",
      "type": "preset"
    }
  ]
}
```

#### 预设音色速查表

| 音色 ID | 语言 | 性别 | 说明 |
|---------|------|------|------|
| `vivian` | 中文 | 女 | 明亮、略带锋芒 |
| `serena` | 中文 | 女 | 温暖、温柔 |
| `uncle_fu` | 中文 | 男 | 低沉柔和 |
| `dylan` | 中文（北京话） | 男 | 清晰自然 |
| `eric` | 中文（四川话） | 男 | 活泼、沙哑明亮 |
| `Ono_Anna` | 英文 | 男 | 充满活力、节奏感强 |
| `aiden` | 英文 | 男 | 阳光、中音清晰 |
| `ono_anna` | 日文 | 女 | 轻盈灵巧 |
| `sohee` | 韩文 | 女 | 情感丰富 |

> **注意**：`Ono_Anna`（英文男声）和 `ono_anna`（日文女声）是大小写不同的两个独立音色，调用时需使用正确的 ID。

如果 `voices/` 目录下存在自定义音色 JSON 文件，也会一并返回，`type` 为 `"cloned"`。

#### SpringBoot 调用示例

```java
RestTemplate restTemplate = new RestTemplate();
String url = "http://localhost:8000/api/v1/tts/voices";
JSONObject response = restTemplate.getForObject(url, JSONObject.class);
JSONArray voices = response.getJSONArray("voices");
```

---

### 4.4 音色克隆（模拟）

`POST /api/v1/tts/clone`

> **注意**：当前版本音色克隆为**模拟实现**（仅保存音频元数据到 `voices/` 目录），并未实际训练或微调模型。

#### 请求

- **Content-Type**: `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `reference_audio` | File | ✅ | 参考音频文件（用于克隆音色） |
| `transcript` | string | ❌ | 音频对应的文本转录 |
| `voice_name` | string | ✅ | 自定义音色名称 |

#### 响应

```json
{
  "voice_id": "cloned_a1b2c3d4",
  "message": "Voice cloned successfully (simulated)"
}
```

---

## 5. ASR 语音识别

### 5.1 转录音频

`POST /api/v1/asr/transcribe`

将上传的音频文件转录为文字，基于 Faster-Whisper 模型。

#### 请求

- **Content-Type**: `multipart/form-data`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `audio` | File | ✅ | - | 音频文件，支持格式：`.wav`、`.mp3`、`.flac`、`.m4a` |
| `language` | string | ❌ | `null`（自动检测） | 语言代码，如 `"en"`、`"zh"`、`"ja"`。留空则自动检测 |
| `task` | string | ❌ | `"transcribe"` | 任务类型：`"transcribe"`（转录）或 `"translate"`（翻译为英文） |
| `beam_size` | int | ❌ | `5` | 束搜索大小，越大结果越准确但越慢 |
| `word_timestamps` | bool | ❌ | `false` | 是否返回单词级时间戳 |

#### 成功响应

```json
{
  "text": "Hello, welcome to Memory English Learning App.",
  "language": "en",
  "language_probability": 0.95,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.2,
      "text": " Hello, welcome to Memory English Learning App.",
      "confidence": -0.12
    }
  ]
}
```

当 `word_timestamps: true` 时，每个 segment 会包含 `words` 字段：

```json
{
  "text": "Hello, welcome to Memory English Learning App.",
  "language": "en",
  "language_probability": 0.95,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": " Hello, welcome to Memory English Learning App.",
      "confidence": -0.12,
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.3,
          "probability": 0.98
        },
        {
          "word": "welcome",
          "start": 0.4,
          "end": 0.8,
          "probability": 0.96
        }
      ]
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 完整转录文本 |
| `language` | string | 检测到的语言代码 |
| `language_probability` | float | 语言检测置信度（0~1） |
| `segments` | array | 分段结果列表 |
| `segments[].id` | int | 分段序号 |
| `segments[].start` | float | 开始时间（秒） |
| `segments[].end` | float | 结束时间（秒） |
| `segments[].text` | string | 该段文字 |
| `segments[].confidence` | float | 置信度（平均对数概率） |
| `segments[].words` | array | 单词级时间戳（仅 `word_timestamps=true` 时） |
| `segments[].words[].word` | string | 单词 |
| `segments[].words[].start` | float | 单词开始时间（秒） |
| `segments[].words[].end` | float | 单词结束时间（秒） |
| `segments[].words[].probability` | float | 单词识别概率 |

#### SpringBoot 调用示例

```java
RestTemplate restTemplate = new RestTemplate();
String url = "http://localhost:8000/api/v1/asr/transcribe";

// 构建 multipart 请求
LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("audio", new FileSystemResource("audio.wav"));
body.add("language", "en");
body.add("task", "transcribe");
body.add("beam_size", 5);
body.add("word_timestamps", true);

HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.MULTIPART_FORM_DATA);

HttpEntity<LinkedMultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

JSONObject response = restTemplate.postForObject(url, requestEntity, JSONObject.class);
String transcribedText = response.getString("text");
```

---

### 5.2 获取支持的 ASR 模型

`GET /api/v1/asr/models`

返回 Faster-Whisper 支持的所有模型尺寸列表。

#### 响应

```json
{
  "models": [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2"
  ]
}
```

当前服务默认使用 `base` 模型，可通过环境变量 `WHISPER_MODEL_SIZE` 修改。

---

## 6. 发音评价

本项目提供两种发音评价方式：

| 方式 | 接口 | 输入 | 原理 | 适用场景 |
|------|------|------|------|---------|
| **MFCC + DTW**（旧） | `/api/v1/pronunciation/score` | 学生录音 + 标准参考音频 | 声学特征对齐 | 有标准参考音频时 |
| **G2P + 音素对齐**（新）🔥 | `/api/v1/pronunciation/phoneme-score` | 学生录音 + 参考文本 | ASR→G2P→音素比对 | 无参考音频、语义层评价 |

### 6.1 MFCC+DTW 发音评分（需参考音频）

`POST /api/v1/pronunciation/score`

比较学生录音和标准参考录音，基于 **MFCC 特征 + DTW 动态时间规整**算法计算发音相似度评分。

#### 请求

- **Content-Type**: `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `student_audio` | File | ✅ | 学生/用户的录音文件（`.wav`、`.mp3`、`.flac`） |
| `reference_audio` | File | ✅ | 标准参考音频文件（同格式） |

#### 成功响应

```json
{
  "score": 85.3,
  "distance": 147.0,
  "max_distance": 1000,
  "level": "good",
  "feedback": "发音良好，注意个别音节的准确性"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | float | 综合评分（0~100，越高越好） |
| `distance` | float | DTW 距离值（越小越好） |
| `max_distance` | float | 归一化最大距离（用于评分计算） |
| `level` | string | 等级：`excellent`(≥90)、`good`(≥75)、`fair`(≥60)、`poor`(≥40)、`very_poor`(<40) |
| `feedback` | string | 中文反馈建议 |

#### SpringBoot 调用示例

```java
RestTemplate restTemplate = new RestTemplate();
String url = "http://localhost:8000/api/v1/pronunciation/score";

LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("student_audio", new FileSystemResource("student.wav"));
body.add("reference_audio", new FileSystemResource("reference.wav"));

HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.MULTIPART_FORM_DATA);

HttpEntity<LinkedMultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

JSONObject response = restTemplate.postForObject(url, requestEntity, JSONObject.class);
double score = response.getDouble("score");
String level = response.getString("level");
String feedback = response.getString("feedback");
```

---

### 6.2 批量发音评分（MFCC+DTW）

`POST /api/v1/pronunciation/batch-score`

对多对学生/参考音频对进行批量评分。

#### 请求体 (JSON)

```json
[
  {
    "student": "/path/to/student1.wav",
    "reference": "/path/to/ref1.wav"
  },
  {
    "student": "/path/to/student2.wav",
    "reference": "/path/to/ref2.wav"
  }
]
```

> **注意**：当前服务端的批量接口要求传入的是**服务端本地文件路径**。建议在 SpringBoot 端先行上传音频文件到 TTS 服务器，再调用此接口。

#### 响应

```json
{
  "results": [
    {
      "score": 85.3,
      "distance": 147.0,
      "max_distance": 1000,
      "level": "good",
      "feedback": "发音良好，注意个别音节的准确性",
      "student_audio": "/path/to/student1.wav",
      "reference_audio": "/path/to/ref1.wav"
    },
    {
      "score": 62.1,
      "distance": 379.0,
      "max_distance": 1000,
      "level": "fair",
      "feedback": "发音基本正确，需要加强练习",
      "student_audio": "/path/to/student2.wav",
      "reference_audio": "/path/to/ref2.wav"
    }
  ]
}
```

---

### 6.3 音素对齐发音评分（仅需参考文本，推荐）🔥

`POST /api/v1/pronunciation/phoneme-score`

基于 **G2P + ASR + 音素对齐** 的发音评价。**不需要标准参考音频**，只需要参考文本。内部流程为：

```
学生录音 → ASR（带时间戳）→ 单词序列 → G2P → 实际音素
参考文本 ───────────────────────→ G2P → 期望音素
                                          ↓
                              编辑距离对齐 → 音素级比对 → 综合评分
```

#### 请求

- **Content-Type**: `multipart/form-data`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `student_audio` | File | ✅ | - | 学生录音文件（`.wav`、`.mp3`、`.flac`、`.m4a`），自动标准化为 16-bit PCM WAV |
| `reference_text` | string | ✅ | - | 期望朗读的参考文本 |
| `language` | string | ❌ | `null`（英语） | 语言代码：`"en"` 英文、`"zh"` 中文 |

> **音频处理**：上传的音频会自动标准化为 16-bit PCM WAV 格式，且要求时长 ≥ 0.3 秒。过短或静音的音频将被拒绝（`400 Bad Request`）。

#### 成功响应

```json
{
  "overall_score": 85.3,
  "phoneme_accuracy": 0.92,
  "word_count_reference": 5,
  "word_count_spoken": 5,
  "asr_transcript": "Hello, welcome to Memory English Learning App.",
  "reference_text": "Hello, welcome to Memory English Learning App.",
  "level": "good",
  "feedback": "发音良好。需重点练习的词汇：welcome",
  "words": [
    {
      "word": "Hello",
      "spoken_word": "Hello",
      "start_time": 0.0,
      "end_time": 0.42,
      "score": 100.0,
      "expected_phonemes": ["HH", "AH", "L", "OW"],
      "actual_phonemes": ["HH", "AH", "L", "OW"],
      "phoneme_accuracy": 1.0,
      "errors": [],
      "status": "correct"
    },
    {
      "word": "welcome",
      "spoken_word": "welcom",
      "start_time": 0.42,
      "end_time": 1.1,
      "score": 80.0,
      "expected_phonemes": ["W", "EH", "L", "K", "AH", "M"],
      "actual_phonemes": ["W", "EH", "L", "K", "AH", "M"],
      "phoneme_accuracy": 1.0,
      "errors": [],
      "status": "mispronounced"
    },
    {
      "word": "Learning",
      "spoken_word": "Learning",
      "start_time": 2.0,
      "end_time": 2.8,
      "score": 88.9,
      "expected_phonemes": ["L", "ER", "N", "IH", "NG"],
      "actual_phonemes": ["L", "AH", "N", "IH", "NG"],
      "phoneme_accuracy": 0.8,
      "errors": [
        {
          "type": "substitution",
          "expected": "ER",
          "actual": "AH",
          "position": 1
        }
      ],
      "status": "mispronounced"
    }
  ]
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `overall_score` | float | 综合评分 0-100，音素准确率 × 100 减去缺失/多余词惩罚 |
| `phoneme_accuracy` | float | 全局音素准确率（0.0~1.0） |
| `word_count_reference` | int | 参考文本中的单词数 |
| `word_count_spoken` | int | ASR 识别出的单词数 |
| `asr_transcript` | string | ASR 转写出的完整文本 |
| `reference_text` | string | 输入的参考文本 |
| `asr_transcript` | string | ASR 转写出的完整文本 |
| `reference_text` | string | 输入的参考文本 |
| `level` | string | 等级：`excellent`/`good`/`fair`/`poor`/`very_poor` |
| `feedback` | string | 中文反馈建议（包含需重点练习的词汇） |
| `words[].word` | string | 参考单词 |
| `words[].spoken_word` | string\|null | 学生实际说出的词，`null` 表示未读 |
| `words[].start_time` | float\|null | 该词在音频中的起始时间（秒） |
| `words[].end_time` | float\|null | 该词在音频中的结束时间（秒） |
| `words[].score` | float | 该词的发音评分 |
| `words[].expected_phonemes` | [string] | 期望音素序列（ARPAbet 去重音） |
| `words[].actual_phonemes` | [string] | 实际音素序列 |
| `words[].phoneme_accuracy` | float | 该词音素准确率 |
| `words[].errors` | array | 音素错误列表 |
| `words[].errors[].type` | string | 错误类型：`substitution`（替换）、`deletion`（缺失）、`insertion`（多余） |
| `words[].errors[].expected` | string\|null | 期望的音素 |
| `words[].errors[].actual` | string\|null | 实际的音素 |
| `words[].errors[].position` | int | 错误在音素序列中的位置 |
| `words[].status` | string | `correct`/`mispronounced`/`missing`/`extra` |

#### SpringBoot 调用示例

```java
/**
 * 音素对齐发音评分（无需参考音频，仅需参考文本）
 */
public JSONObject phonemeScore(File studentAudio, String referenceText, String language) {
    String url = baseUrl + "/api/v1/pronunciation/phoneme-score";

    LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
    body.add("student_audio", new FileSystemResource(studentAudio));
    body.add("reference_text", referenceText);
    if (language != null) body.add("language", language);

    HttpHeaders headers = new HttpHeaders();
    headers.setContentType(MediaType.MULTIPART_FORM_DATA);

    HttpEntity<LinkedMultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
    return restTemplate.postForObject(url, request, JSONObject.class);
}

// 使用示例（注意：响应字段为 camelCase 格式）
File audioFile = new File("student_recording.wav");
String refText = "Hello, welcome to Memory English Learning App!";
JSONObject result = phonemeScore(audioFile, refText, "en");

double score = result.getDouble("overallScore");       // camelCase
String asrTranscript = result.getString("asrTranscript"); // camelCase
JSONArray words = result.getJSONArray("words");
for (int i = 0; i < words.size(); i++) {
    JSONObject word = words.getJSONObject(i);
    if ("mispronounced".equals(word.getString("status"))) {
        System.out.println("发音有问题的词: " + word.getString("word") +
            " → 学生说的是: " + word.optString("spokenWord", "(未读)"));
    }
}
```

---

### 6.4 批量音素评分

`POST /api/v1/pronunciation/phoneme-batch-score`

> **注意**：phoneme-score 系列的响应字段采用 **camelCase** 命名风格（如 `overallScore`、`phonemeAccuracy`、`wordCountReference`、`asrTranscript`、`referenceText`），这是服务端 `_snake_to_camel()` 自动转换的结果。上方 JSON 示例为方便阅读保留了 snake_case，实际调用时请以 camelCase 方式访问。

> **注意**：当前批量接口要求音频已存在于服务端。建议在 SpringBoot 端先将音频上传到 TTS 服务器再调用，或循环调用单条接口。

#### 请求体 (JSON)

```json
[
  {
    "audio": "/data/audio/student1.wav",
    "reference_text": "Hello, welcome to Memory English Learning App!",
    "language": "en"
  },
  {
    "audio": "/data/audio/student2.wav",
    "reference_text": "我喜欢学习英语",
    "language": "zh"
  }
]
```

> 批量接口的 `reference_text` 字段同时充当音频路径标识（当前简化实现）。如需完整批量上传功能，建议先通过文件上传接口将音频保存到服务端。
>
> **注意**：批量音素评分接口当前**不支持逐项指定语言**，所有项统一使用默认语言（英语）。返回的响应字段为 **snake_case** 格式（与单条 phoneme-score 的 camelCase 不同）。

#### 响应

```json
{
  "results": [
    { /* 与单条接口返回格式一致 */ },
    { /* 与单条接口返回格式一致 */ }
  ]
}
```

---

## 7. 系统接口

### 7.1 健康检查

`GET /api/v1/health`

检查服务状态，包括模型加载状态和 GPU 信息。

#### 响应

```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_utilization_percent": 0,
  "vram_used_mb": 4096.0,
  "vram_total_mb": 8192.0
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"healthy"` 或 `"unhealthy"` |
| `model_loaded` | bool | 模型是否已加载 |
| `gpu_available` | bool | 是否可用 GPU |
| `gpu_utilization_percent` | float | GPU 利用率百分比 |
| `vram_used_mb` | float | 已用显存（MB） |
| `vram_total_mb` | float | 总显存（MB） |

#### SpringBoot 调用示例

```java
RestTemplate restTemplate = new RestTemplate();
String url = "http://localhost:8000/api/v1/health";
JSONObject health = restTemplate.getForObject(url, JSONObject.class);
boolean isHealthy = "healthy".equals(health.getString("status"));
```

---

## 8. OCR 文字识别

### 8.1 图片文字扫描

```http
POST /api/v1/ocr/scan
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image | file | 是 | PNG/JPG/BMP/WEBP |
| language | string | 否 | en/ch/Multilingual，默认 en |

**成功响应：**
```json
{
    "success": true,
    "text": "Hello world\nThis is a test",
    "lines": ["Hello world", "This is a test"],
    "confidences": [0.985, 0.972],
    "avg_confidence": 0.9785,
    "model_tier": "small",
    "processing_time_ms": 452
}
```

### 8.2 文档文字扫描

```http
POST /api/v1/ocr/scan-file
Content-Type: multipart/form-data
```

支持 PDF 文件，返回逐页文本。

### 8.3 OCR 健康检查

```http
GET /api/v1/ocr/health
```

返回 OCR 模型状态、当前配置（模型档位/预处理/检测参数）、可用档位和预设列表。

### 8.4 管理后台

```http
GET /admin
```

内嵌的单页管理面板，支持 TTS/ASR/OCR/发音评价的在线调试和测速。

---

## 9. 全局错误处理

所有接口在出错时返回统一格式：

```json
{
  "detail": "错误描述信息"
}
```

常见 HTTP 状态码：

| 状态码 | 含义 | 常见原因 |
|--------|------|---------|
| `400` | 参数错误 | 不支持的音频格式、缺少必填字段 |
| `500` | 服务内部错误 | 模型推理失败、文件读写异常 |

---

## 9. SpringBoot 集成示例

### 9.1 添加依赖

```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-websocket</artifactId>
</dependency>
<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>fastjson</artifactId>
    <version>2.0.x</version>
</dependency>
```

### 9.2 配置类

```java
@Configuration
public class TtsServiceConfig {

    @Value("${tts.service.url:http://localhost:8000}")
    private String ttsServiceUrl;

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }

    @Bean
    public String ttsServiceUrl() {
        return ttsServiceUrl;
    }
}
```

### 9.3 服务调用封装

```java
@Service
public class TtsService {

    @Autowired
    private RestTemplate restTemplate;

    @Value("${tts.service.url}")
    private String baseUrl;

    /**
     * 健康检查
     */
    public boolean healthCheck() {
        try {
            String url = baseUrl + "/api/v1/health";
            JSONObject response = restTemplate.getForObject(url, JSONObject.class);
            return "healthy".equals(response.getString("status"));
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 文本合成语音
     * @return WAV 音频字节数组
     */
    public byte[] synthesize(String text, String voice, String language, String instructions) {
        String url = baseUrl + "/api/v1/tts/synthesize";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        JSONObject body = new JSONObject();
        body.put("text", text);
        body.put("voice", voice != null ? voice : "Ono_Anna");
        body.put("language", language != null ? language : "English");
        body.put("instructions", instructions);
        body.put("output_format", "wav");

        HttpEntity<String> request = new HttpEntity<>(body.toString(), headers);
        ResponseEntity<byte[]> response = restTemplate.postForEntity(url, request, byte[].class);
        return response.getBody();
    }

    /**
     * 转录音频
     */
    public JSONObject transcribe(File audioFile, String language, boolean wordTimestamps) {
        String url = baseUrl + "/api/v1/asr/transcribe";

        LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("audio", new FileSystemResource(audioFile));
        if (language != null) body.add("language", language);
        body.add("word_timestamps", wordTimestamps);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<LinkedMultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
        return restTemplate.postForObject(url, request, JSONObject.class);
    }

    /**
     * 发音评分
     */
    public JSONObject scorePronunciation(File studentAudio, File referenceAudio) {
        String url = baseUrl + "/api/v1/pronunciation/score";

        LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("student_audio", new FileSystemResource(studentAudio));
        body.add("reference_audio", new FileSystemResource(referenceAudio));

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<LinkedMultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
        return restTemplate.postForObject(url, request, JSONObject.class);
    }

    /**
     * 音素对齐发音评分（无需参考音频）
     */
    public JSONObject phonemeScore(File studentAudio, String referenceText, String language) {
        String url = baseUrl + "/api/v1/pronunciation/phoneme-score";

        LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("student_audio", new FileSystemResource(studentAudio));
        body.add("reference_text", referenceText);
        if (language != null) body.add("language", language);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<LinkedMultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
        return restTemplate.postForObject(url, request, JSONObject.class);
    }

    /**
     * 获取音色列表
     */
    public JSONArray getVoices() {
        String url = baseUrl + "/api/v1/tts/voices";
        JSONObject response = restTemplate.getForObject(url, JSONObject.class);
        return response.getJSONArray("voices");
    }
}
```

### 9.4 application.yml 配置

```yaml
tts:
  service:
    url: http://localhost:8000   # TTS 服务地址
```

---

## 10. 配置与环境变量

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `QWEN_TTS_MODEL_PATH` | `"./models/qwen-0.6b"` | 本地 TTS 模型路径（优先加载的小模型） |
| `QWEN_TTS_MODEL` | `"Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"` | HuggingFace 模型名称（本地不存在时的回退） |
| `WHISPER_MODEL_SIZE` | `"base"` | Faster-Whisper 模型尺寸（可选：tiny/base/small/medium/large-v3 等） |
| `RELOAD` | `"0"` | 设为 `"1"` 启用 uvicorn 热重载（开发调试用，生产环境建议关闭） |

### 模型加载策略

TTS 模型加载优先级（**内存友好策略** — 优先加载 0.6B 小模型）：

1. **主尝试**：加载 `QWEN_TTS_MODEL_PATH` 指定的本地路径（默认 `./models/qwen-0.6b`）
2. **回退**：若本地路径不存在，从 HuggingFace 下载 `QWEN_TTS_MODEL`（默认 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`）
3. **降级**：若 0.6B 模型全部加载失败，尝试 1.7B 模型：先 `./models/qwen-1.7b`（本地），再 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`（HuggingFace）

> **注意**：代码实际策略是 **0.6B 优先**（非文档旧版所述的 1.7B 优先），以节省显存。可通过环境变量修改默认路径。当 GPU 支持 Ampere 架构（SM 8.0+）时自动启用 `bfloat16` 精度，否则使用 `float32`。

### 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| FastAPI 主服务 | `8000` | REST + WebSocket |
| Gradio 调试 UI | `7860` | 仅调试使用（位于 `src/debug_ui.py`） |

---

## 11. 注意事项

### 11.1 已知限制

1. **TTS 输出格式**：目前仅支持 WAV 格式输出。
2. **音色克隆**：当前版本为模拟实现。
3. **单线程模型锁**：TTS 模型使用 `asyncio.Lock`，并发场景下需排队。
4. **音素评价依赖**：英文 G2P 需安装 `g2p-en`（含神经网络模型，首次使用会下载约 20MB 的模型文件）；中文 G2P 需 `pypinyin`。
5. **批量音素评分**：`/api/v1/pronunciation/phoneme-batch-score` 当前为简化实现，建议在 SpringBoot 端循环调用单条接口。

### 11.2 性能建议

- 推荐使用 GPU 运行，1.7B 模型在 V100 16GB 上单次合成约 1-3 秒。
- 0.6B 模型适合低显存环境（2-3 GB），合成速度略快但音质稍逊。
- 高并发场景建议部署多个 TTS 服务实例，前端做负载均衡。
- WebSocket 流式合成适合长文本，可减少等待时间。

### 11.3 网络与代理

服务端 `main.py` 默认配置了 SSL 验证禁用（内网/代理环境），生产环境建议配置正确的 CA 证书：

```python
# 恢复 SSL 验证（生产环境）
# os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '0'
```

### 11.4 项目文件结构

```
MemoryServerTTS/
├── main.py                         # 启动入口（支持 RELOAD 环境变量控制热重载）
├── requirements.txt                # Python 依赖
├── Dockerfile                      # Docker 构建文件
├── start_server.bat                # Windows 启动脚本
├── start_server.sh                 # Linux/macOS 启动脚本
├── README.md                       # 项目 README
├── src/
│   ├── __init__.py                 # 空文件，包标记
│   ├── server.py                   # FastAPI 主服务（所有 API 端点）
│   ├── model_loader.py             # TTS 模型管理器（Qwen3-TTS，0.6B→1.7B 两级降级）
│   ├── asr_model_loader.py         # ASR 模型管理器（Faster-Whisper）
│   ├── pronunciation_evaluator.py  # 发音评价器（MFCC + DTW，需参考音频）
│   ├── phoneme_evaluator.py        # 🔥 音素评价器（G2P + ASR 对齐，仅需参考文本）
│   ├── g2p_engine.py               # 🔥 G2P 引擎抽象层（英文 g2p-en / 中文 pypinyin）
│   ├── debug_ui.py                 # Gradio 调试界面（端口 7860）
│   └── test_official.py            # 官方接口测试脚本（含 SSL 配置参考）
├── models/
│   ├── qwen-1.7b/                  # Qwen3-TTS 1.7B 大模型（降级方案）
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── speech_tokenizer/
│   └── qwen-0.6b/                  # Qwen3-TTS 0.6B 小模型（默认主模型）
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       └── speech_tokenizer/
├── voices/                         # 音色克隆数据目录
└── doc/
    ├── API_DOCUMENTATION.md        # 本文件
    ├── phoneme-score-fix.md        # 发音评价接口修复指南
    └── TROUBLESHOOTING_SPRINGBOOT.md  # SpringBoot 接入排错指南
```

---

> **文档版本**：v1.2.0  
> **最后更新**：2026-05-24  
> **如有问题**，请检查服务端日志或调用 `/api/v1/health` 确认服务状态。
