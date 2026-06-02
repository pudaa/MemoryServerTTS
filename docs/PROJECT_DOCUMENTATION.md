# MemoryServerTTS 项目完整技术文档

> **版本**: 1.0.0  
> **最后更新**: 2026-05-29  
> **项目定位**: 基于 **Qwen3-TTS** 和 **Faster-Whisper** 的智能语音服务中间件，提供文本转语音（TTS）、语音识别（ASR）、发音评价三大核心能力，专为英语学习等语言教育场景设计。

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈总览](#2-技术栈总览)
3. [系统架构设计](#3-系统架构设计)
4. [核心模块详解](#4-核心模块详解)
   - [4.1 TTS 模型管理器 — `model_loader.py`](#41-tts-模型管理器--model_loaderpy)
   - [4.2 ASR 模型管理器 — `asr_model_loader.py`](#42-asr-模型管理器--asr_model_loaderpy)
   - [4.3 发音评价器（MFCC+DTW）— `pronunciation_evaluator.py`](#43-发音评价器mfccdtw--pronunciation_evaluatorpy)
   - [4.4 G2P 引擎 — `g2p_engine.py`](#44-g2p-引擎--g2p_enginepy)
   - [4.5 音素发音评价器 — `phoneme_evaluator.py`](#45-音素发音评价器--phoneme_evaluatorpy)
   - [4.6 FastAPI 主服务 — `server.py`](#46-fastapi-主服务--serverpy)
   - [4.7 Gradio 调试界面 — `debug_ui.py`](#47-gradio-调试界面--debug_uipy)
5. [设计思路与关键决策](#5-设计思路与关键决策)
6. [实现流程详解](#6-实现流程详解)
   - [6.1 TTS 语音合成流程](#61-tts-语音合成流程)
   - [6.2 ASR 语音识别流程](#62-asr-语音识别流程)
   - [6.3 发音评价流程（MFCC+DTW）](#63-发音评价流程mfccdtw)
   - [6.4 音素发音评价流程（G2P+ASR）](#64-音素发音评价流程g2pasr)
7. [依赖详解](#7-依赖详解)
8. [配置与环境变量](#8-配置与环境变量)
9. [部署指南](#9-部署指南)
10. [API 接口速查](#10-api-接口速查)
11. [开发与调试](#11-开发与调试)
12. [常见问题与排错](#12-常见问题与排错)

---

## 1. 项目概述

### 1.1 项目背景

MemoryServerTTS 是"记忆英语"（Memory English Learning App）的后端语音服务组件。项目的核心目标是构建一个**高性能、模块化、易于集成的语音服务中间件**，为上层的 Web / 移动端应用（如 SpringBoot 后端服务）提供以下能力：

- **语音合成（TTS）**：将文本转化为自然流畅的语音，支持多种语言、多种音色和情感指令
- **语音识别（ASR）**：将用户语音转写为文字，支持多语言自动检测
- **发音评价**：评估用户发音的准确度，提供逐词、逐音素的精细化反馈

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **多语言 TTS** | 支持中文、英文、日文、韩文等多语言语音合成 |
| **多音色选择** | 9 个预设音色，涵盖不同语言、性别和风格 |
| **情感指令控制** | 通过自然语言描述控制语气、情感和风格 |
| **流式合成** | WebSocket 支持实时流式语音输出 |
| **高精度 ASR** | 基于 Faster-Whisper 的 CTranslate2 加速推理 |
| **双模式发音评价** | MFCC+DTW（需参考音频）和 G2P+ASR（仅需参考文本）两种评价模式 |
| **音素级反馈** | 精确定位到具体单词、具体音素的发音错误类型（替换/遗漏/插入） |
| **调试 UI** | Gradio 可视化界面，方便快速验证 |

---

## 2. 技术栈总览

### 2.1 技术选型

| 层级 | 技术方案 | 选型理由 |
|------|---------|---------|
| **Web 框架** | FastAPI + Uvicorn | 高性能异步框架，原生支持 OpenAPI 文档和 WebSocket |
| **TTS 引擎** | Qwen3-TTS (`qwen_tts`) | 阿里通义千问第 3 代语音合成模型，效果业界领先 |
| **ASR 引擎** | Faster-Whisper | OpenAI Whisper 的 CTranslate2 加速版，推理速度提升 4 倍 |
| **深度学习框架** | PyTorch 2.7+ | 主流深度学习框架，CUDA 加速 |
| **G2P（英文）** | `g2p-en` | 基于 NLTK 的英文 Grapheme-to-Phoneme 转换 |
| **G2P（中文）** | `pypinyin` | 中文拼音转换库 |
| **音频处理** | `librosa` + `soundfile` | MFCC 特征提取、音频格式标准化 |
| **序列对齐** | `fastdtw` + `difflib` | 动态时间规整和文本序列匹配 |
| **调试 UI** | Gradio | 快速搭建机器学习模型演示界面 |
| **容器化** | Docker | 提供标准化的部署环境 |

### 2.2 运行环境要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | 3.10+ | 3.12 |
| PyTorch | 2.0.1+ | 2.7.0+ (CUDA 12.8) |
| 显存 (0.6B 模型) | 2 GB | 4 GB |
| 显存 (1.7B 模型) | 4 GB | 8 GB |
| 磁盘空间 | 3 GB | 10 GB (含模型文件) |
| 操作系统 | Linux / Windows / macOS | Linux (生产) |

---

## 3. 系统架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        客户端应用层                                    │
│    (SpringBoot 后端 / 移动端 App / Web 前端)                          │
└────────────────────┬──────────────────────┬──────────────────────────┘
                     │                      │
              ┌──────▼──────┐       ┌───────▼───────┐
              │  HTTP REST  │       │   WebSocket   │
              │  (JSON/File)│       │  (流式音频)    │
              └──────┬──────┘       └───────┬───────┘
                     │                      │
┌────────────────────▼──────────────────────▼─────────────────────────┐
│                        FastAPI 服务层 (server.py)                     │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   TTS API    │  │   ASR API    │  │  发音评分   │  │  系统接口  │ │
│  │  synthesize  │  │  transcribe  │  │ score/phone │  │ health/   │ │
│  │  stream(WS)  │  │  models      │  │ me-score    │  │ voices    │ │
│  │  voices/     │  │              │  │ batch-score  │  │           │ │
│  │  clone       │  │              │  │             │  │           │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘  └─────┬─────┘ │
│         │                 │                 │               │       │
└─────────┼─────────────────┼─────────────────┼───────────────┼───────┘
          │                 │                 │               │
┌─────────▼─────────────────▼─────────────────▼───────────────▼───────┐
│                        模型管理层 (单例模式)                           │
│                                                                      │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────────────┐  │
│  │  TTSModelManager │  │ ASRModelManager│  │  PhonemeEvaluator   │  │
│  │  (Qwen3TTSModel) │  │ (WhisperModel) │  │  ┌──────────────┐  │  │
│  │  单例 / 懒加载    │  │  单例 / 懒加载  │  │  │  G2PEngine   │  │  │
│  │  0.6B → 1.7B     │  │  base → large  │  │  │  英文/中文    │  │  │
│  │  自动降级         │  │                │  │  └──────────────┘  │  │
│  └──────────────────┘  └────────────────┘  │  ASRModelManager   │  │
│                                             │  (内部依赖)         │  │
│  ┌─────────────────────────────────────────┐└─────────────────────┘  │
│  │  PronunciationEvaluator (MFCC+DTW)       │                       │
│  │  librosa → MFCC → fastdtw → 评分         │                       │
│  └──────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 架构设计原则

1. **单例管理模式**：`TTSModelManager` 和 `ASRModelManager` 均采用单例模式，确保模型在内存中只加载一次，避免重复加载导致的资源浪费。
2. **自动降级策略**：TTS 模型加载时优先尝试 0.6B 轻量模型，失败后自动降级到 1.7B 模型（反之亦然），提高服务可用性。
3. **异步锁保护**：TTS 生成过程使用 `asyncio.Lock` 保护，避免并发请求导致 GPU 显存冲突。
4. **模块化设计**：各功能模块（TTS、ASR、G2P、发音评价）独立封装，通过依赖注入组合使用。
5. **策略模式（G2P）**：使用抽象基类和工厂函数，支持按语言灵活切换 G2P 实现。

---

## 4. 核心模块详解

### 4.1 TTS 模型管理器 — `model_loader.py`

**文件路径**: `src/model_loader.py`

#### 职责

管理 Qwen3-TTS 模型的加载、生命周期管理和语音生成调用。

#### 类结构

```python
class TTSModelManager:
    _instance = None  # 单例标志

    def __new__(cls)           # 单例构造
    def _load_model(self)      # 模型加载（含自动降级）
    def generate(self, text, voice, language, instructions, streaming)  # 语音生成
```

#### 设计要点

**1. 单例模式实现**

```python
def __new__(cls):
    if cls._instance is None:
        cls._instance = super().__new__(cls)
        cls._instance._load_model()
    return cls._instance
```

- 首次创建时调用 `_load_model()` 加载模型
- 后续复用已加载的实例

**2. 多级加载策略**

加载顺序：
1. 优先加载本地 `models/qwen-0.6b/`（轻量模型，~2-3 GB 显存）
2. 若本地不存在，从 HuggingFace Hub 下载 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
3. 若上述均失败，降级到 `models/qwen-1.7b/`（高质量模型，~4-6 GB 显存）
4. 最后尝试 HuggingFace 的 1.7B 远程模型

可通过环境变量 `QWEN_TTS_MODEL_PATH` 和 `QWEN_TTS_MODEL` 覆盖默认模型路径。

**3. GPU 计算精度自适应**

```python
caps = torch.cuda.get_device_capability()
supports_bf16 = caps[0] >= 8  # Ampere (SM 8.0+) 支持 bfloat16
compute_dtype = torch.bfloat16 if (use_gpu and supports_bf16) else torch.float32
```

- NVIDIA Ampere 架构及以上（RTX 30xx/A100/H100）使用 `bfloat16`
- 旧架构或 CPU 回退到 `float32`

**4. 注意力实现**

固定使用 `attn_implementation="sdpa"`（PyTorch 原生缩放点积注意力），避免对 `flash-attn` 的依赖。

#### 关键 API

```python
def generate(self, text, voice="Ryan", language="English",
             instructions="", streaming=False) -> tuple[list[np.ndarray], int]:
```

- **参数**:
  - `text`: 待合成文本
  - `voice`: 音色 ID（如 "Ono_Anna", "aiden"）
  - `language`: 语言（如 "English", "Chinese"）
  - `instructions`: 情感/风格指令
  - `streaming`: 是否流式模式（对应 `non_streaming_mode` 取反）
- **返回**: `(wavs_list, sample_rate)`，其中 `wavs_list[0]` 为合成音频的 numpy 数组

---

### 4.2 ASR 模型管理器 — `asr_model_loader.py`

**文件路径**: `src/asr_model_loader.py`

#### 职责

管理 Faster-Whisper 模型的加载和音频转录调用。

#### 类结构

```python
class ASRModelManager:
    _instance = None  # 单例标志

    def __new__(cls)           # 单例构造
    def _load_model(self)      # 模型加载
    def transcribe(self, audio_path, language, task, beam_size, word_timestamps)
    def get_supported_models(self)  # 支持的模型列表
```

#### 设计要点

**1. 模型选择控制**

```python
model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
```

- 默认加载 `base` 模型（约 140 MB，适合快速推理）
- 可通过环境变量 `WHISPER_MODEL_SIZE` 切换
- 支持的模型：`tiny`, `base`, `small`, `medium`, `large-v1/v2/v3`, `distil-large-v2`

**2. 计算类型自适应**

```python
compute_type = "float16" if device == "cuda" else "int8"
```

- GPU 上使用 `float16` 加速
- CPU 上使用 `int8` 量化减少内存

#### 转录结果结构

```python
{
    "text": "transcribed text",            # 完整转录文本
    "language": "en",                      # 检测到的语言
    "language_probability": 0.95,          # 语言置信度
    "segments": [                          # 分段结果
        {
            "id": 0,
            "start": 0.0,                  # 开始时间（秒）
            "end": 2.5,                    # 结束时间（秒）
            "text": "hello world",         # 该段文本
            "confidence": -0.12,           # 平均对数概率
            "words": [                     # 单词级时间戳（开启时）
                {"word": "hello", "start": 0.0, "end": 0.8, "probability": 0.98},
                ...
            ]
        }
    ]
}
```

---

### 4.3 发音评价器（MFCC+DTW）— `pronunciation_evaluator.py`

**文件路径**: `src/pronunciation_evaluator.py`

#### 职责

通过声学特征对比来评估发音质量，需要**标准参考音频**作为对照。

#### 算法原理

```
学生音频 ─→ librosa.load() ─→ MFCC 特征提取 ─→ ┐
                                                   ├── fastdtw 对齐 ─→ DTW 距离 ─→ 评分 (0-100)
参考音频 ─→ librosa.load() ─→ MFCC 特征提取 ─→ ┘
```

- **MFCC** (Mel Frequency Cepstral Coefficients): 13 维梅尔频率倒谱系数，是语音识别的经典声学特征
- **DTW** (Dynamic Time Warping): 动态时间规整，解决两个音频长短不一的问题，找到最优对齐路径
- **评分公式**: `score = max(0, 100 - (distance / max_distance) * 100)`

#### 评分等级

| 分数范围 | 等级 | 反馈建议 |
|---------|------|---------|
| 90-100 | excellent | 发音非常标准，继续保持！ |
| 75-89 | good | 发音良好，注意个别音节的准确性 |
| 60-74 | fair | 发音基本正确，需要加强练习 |
| 40-59 | poor | 发音有待改进，建议多听标准发音 |
| 0-39 | very_poor | 发音需要大幅改进，建议从基础音标开始练习 |

#### 局限性

- 需要标准参考音频，使用场景受限
- 基于全局声学特征，无法定位到具体哪个词或音素出错
- 对噪声敏感

---

### 4.4 G2P 引擎 — `g2p_engine.py`

**文件路径**: `src/g2p_engine.py`

#### 职责

将书写形式的文字（Grapheme）转换为发音形式的音素（Phoneme），是音素评价器的核心前置组件。

#### 架构设计

采用**抽象基类 + 策略模式 + 工厂函数**：

```
┌────────────────┐
│   G2PEngine    │  ← 抽象基类 (ABC)
│  (抽象接口)     │
└───────┬────────┘
        │ 继承
   ┌────┴────┐
   │         │
   ▼         ▼
┌────────┐ ┌──────────┐
│English │ │ Chinese  │
│ G2P    │ │  G2P     │
│(g2p-en)│ │(pypinyin)│
└────────┘ └──────────┘

工厂函数: get_g2p_engine(language) → G2PEngine 实例
```

#### 抽象基类方法

| 方法 | 说明 |
|------|------|
| `word_to_phonemes(word)` | 单词语音素列表 |
| `word_to_phoneme_string(word)` | 单词语音素字符串（空格分隔） |
| `text_to_phonemes(text)` | 文本 → 单词音素列表的列表 |
| `text_to_word_phoneme_pairs(text)` | 文本 → [(单词, 音素列表), ...] |
| `_tokenize(text)` | 通用分词（静态方法） |

#### 英文 G2P 实现

- 基于 `g2p-en` 库，底层使用 NLTK 的 POS tagger 和 CMU Pronouncing Dictionary
- 音素集：**ARPAbet**（如 HH, AH, L, OW）
- 自动去除重音标记（AH0 → AH, OW1 → OW）
- 提供 `word_to_phonemes_with_stress()` 保留重音信息

#### 中文 G2P 实现

- 基于 `pypinyin` 库
- 拼音拆分：将带声调的拼音分解为声母 + 韵母
  - `zhuang4` → `('zh', 'uang4')`
  - `ni3` → `('n', 'i3')`
- 声母匹配优先长前缀（zh/ch/sh 优于 z/c/s）

#### 工厂函数

```python
def get_g2p_engine(language: str) -> G2PEngine:
    # "en"/"english"/"eng" → EnglishG2P
    # "zh"/"chinese"/"cn"/"mandarin" → ChineseG2P
    # 其他 → 默认 EnglishG2P（带警告）
```

---

### 4.5 音素发音评价器 — `phoneme_evaluator.py`

**文件路径**: `src/phoneme_evaluator.py`

**这是项目的核心创新模块**，也是发音评价的推荐方案。

#### 职责

基于 G2P + ASR + 音素对齐的发音评价器，**不需要标准参考音频**，只需参考文本即可完成评价。

#### 与 MFCC+DTW 评价器的核心区别

| 维度 | MFCC+DTW 评价器 | 音素评价器（推荐） |
|------|----------------|------------------|
| 参考输入 | 需要标准参考音频 | 仅需参考文本 |
| 评价粒度 | 整体评分 | 逐词、逐音素 |
| 错误类型 | 无 | 可区分替换/遗漏/插入 |
| 语义理解 | 无（纯声学对比） | 有（结合 ASR 文本） |
| 适用场景 | 跟读对比 | 发音练习、口语考试 |

#### 完整工作流程

```
学生录音 (audio_path) + 参考文本 (reference_text)
         │
         ├──▶ 第1步：ASR 转录 ──────────────────▶ 学生说的文字 + 单词时间戳
         │
         ├──▶ 第2步：G2P 转换参考文本 ──────────▶ 期望音素序列
         │
         ├──▶ 第3步：G2P 转换 ASR 文本 ─────────▶ 实际音素序列
         │
         ├──▶ 第4步：词级对齐 (SequenceMatcher) ─▶ 参考词 ↔ ASR 词 映射
         │
         ├──▶ 第5步：逐词音素比对 ──────────────▶ 错误类型检测
         │
         └──▶ 第6步：综合评分 ─────────────────▶ 总评分 + 等级 + 反馈
```

#### 详细步骤

**第1步：ASR 转录**

调用 `ASRModelManager.transcribe()` 获取学生录音的文字内容，**必须开启 `word_timestamps=True`** 以获取单词级时间戳。

**第2-3步：G2P 转换**

使用 `G2PEngine.text_to_word_phoneme_pairs()` 将参考文本和 ASR 输出分别转换为 `[(单词, [音素, ...]), ...]` 结构。

**第4步：词级对齐**

使用 Python 标准库 `difflib.SequenceMatcher` 对齐参考文本的单词序列和 ASR 输出的单词序列：

```python
# get_opcodes() 返回四种操作：
# "equal"    → 一对一匹配
# "replace"  → 替换（最佳配对）
# "delete"   → 参考词缺失（学生没读）
# "insert"   → 多余词（学生多读）
```

**第5步：逐词音素比对**

对每个对齐的词对，使用 `difflib.SequenceMatcher` 进行音素级编辑距离比较：

```python
# 错误类型：
# "substitution" → 音素替换（如 /p/ 读成 /b/）
# "deletion"     → 音素遗漏（如 "and" 读成 "an"）
# "insertion"    → 音素多余（如 "cat" 读成 "cater"）
```

**第6步：综合评分**

```python
base_score = phoneme_accuracy * 100               # 基础分
missing_penalty = (missing_count / ref_word_count) * 20  # 缺失惩罚
extra_penalty = min(extra_count * 2, 10)                   # 多余词惩罚
overall_score = max(0, base_score - missing_penalty - extra_penalty)
```

#### 输出结构

```python
{
    "overall_score": 85.3,          # 综合评分 0-100
    "phoneme_accuracy": 0.875,      # 音素准确率
    "word_count_reference": 5,      # 参考文本单词数
    "word_count_spoken": 5,         # 实际说出的单词数
    "asr_transcript": "hello world",  # ASR 转录结果
    "reference_text": "hello world",  # 参考文本
    "level": "good",                # 等级
    "feedback": "发音良好。需重点练习的词汇：world",  # 反馈建议
    "words": [                      # 逐词评分
        {
            "word": "hello",
            "spoken_word": "hello",
            "start_time": 0.0,
            "end_time": 0.8,
            "score": 100.0,
            "expected_phonemes": ["HH", "AH", "L", "OW"],
            "actual_phonemes": ["HH", "AH", "L", "OW"],
            "phoneme_accuracy": 1.0,
            "errors": [],
            "status": "correct"
        },
        {
            "word": "world",
            "spoken_word": "world",
            "start_time": 1.0,
            "end_time": 2.5,
            "score": 75.0,
            "expected_phonemes": ["W", "ER", "L", "D"],
            "actual_phonemes": ["W", "ER", "L"],
            "phoneme_accuracy": 0.75,
            "errors": [
                {"type": "deletion", "expected": "D", "actual": None, "position": 3}
            ],
            "status": "mispronounced"
        }
    ]
}
```

---

### 4.6 FastAPI 主服务 — `server.py`

**文件路径**: `src/server.py`

#### 职责

提供 HTTP RESTful API 和 WebSocket 接口，协调各模型管理器对外暴露服务。

#### 启动生命周期

```python
@app.on_event("startup")
async def startup_event():
    app.state.model = TTSModelManager()              # 加载 TTS 模型
    app.state.asr_model = ASRModelManager()          # 加载 ASR 模型
    app.state.pronunciation_evaluator = PronunciationEvaluator()  # 初始化发音评价器
    app.state.phoneme_evaluator = PhonemeEvaluator(app.state.asr_model)  # 初始化音素评价器
```

**注意**: `PhonemeEvaluator` 依赖 `ASRModelManager`，通过依赖注入方式组合。

#### 并发控制

```python
app.state.model_lock = asyncio.Lock()

# 所有 TTS 生成操作都需要获取锁
async with app.state.model_lock:
    wavs, sr = model.generate(...)
```

**为什么需要锁？** Qwen3-TTS 模型在 GPU 上推理时，并发调用会导致显存竞争和结果错乱。使用 `asyncio.Lock` 确保同一时刻只有一个 TTS 请求在推理。

#### 关键 API 端点

| 端点 | 类型 | 函数 | 说明 |
|------|------|------|------|
| `/api/v1/tts/synthesize` | POST | `synthesize()` | 文本合成 WAV 音频 |
| `/api/v1/tts/stream` | WebSocket | `websocket_stream()` | 流式合成 PCM16 音频 |
| `/api/v1/tts/voices` | GET | `get_voices()` | 音色列表（预设 + 克隆） |
| `/api/v1/tts/clone` | POST | `clone_voice()` | 音色克隆（模拟） |
| `/api/v1/asr/transcribe` | POST | `transcribe_audio()` | 音频文件转录 |
| `/api/v1/asr/models` | GET | `get_asr_models()` | 支持的 ASR 模型列表 |
| `/api/v1/pronunciation/score` | POST | `score_pronunciation()` | MFCC+DTW 发音评分 |
| `/api/v1/pronunciation/batch-score` | POST | `batch_score_pronunciation()` | 批量 MFCC+DTW 评分 |
| `/api/v1/pronunciation/phoneme-score` | POST | `phoneme_score()` | **音素发音评分（推荐）** |
| `/api/v1/pronunciation/phoneme-score-with-text` | POST | `phoneme_score_with_text()` | 同上（兼容别名） |
| `/api/v1/pronunciation/phoneme-batch-score` | POST | `phoneme_batch_score()` | 批量音素评分 |
| `/api/v1/health` | GET | `health_check()` | 健康检查 |

#### 音频文件处理安全措施

所有接收音频文件的接口都遵循以下安全处理流程：

1. **格式校验**: 检查文件扩展名是否在允许列表中
2. **空文件校验**: 使用 `os.path.getsize()` 检查文件是否为空
3. **音频时长校验**: 音素评分接口检查音频时长 >= 0.3 秒
4. **格式标准化**: 使用 `soundfile` 重编码为 16-bit PCM WAV，确保解码器兼容
5. **临时文件清理**: 使用 `BackgroundTask` 或 `try/finally` 确保临时文件被删除

#### snake_case → camelCase 转换

```python
def _snake_to_camel(data):
    """递归将 dict 的 key 从 snake_case 转为 camelCase"""
```

音素评分接口的输出会自动将蛇形命名法（snake_case）转为驼峰命名法（camelCase），方便 Java/JavaScript 客户端使用。

---

### 4.7 Gradio 调试界面 — `debug_ui.py`

**文件路径**: `src/debug_ui.py`

#### 职责

提供可视化的 TTS 调试界面，方便开发人员快速测试语音合成效果。

#### 功能

- 文本输入框（支持多行）
- 音色下拉选择
- 语言下拉选择
- 情感指令输入
- 音频播放器（展示合成结果）
- 错误信息显示

#### 启动方式

```bash
python src/debug_ui.py
```

运行在 `http://0.0.0.0:7860`

---

## 5. 设计思路与关键决策

### 5.1 为什么选择 G2P+ASR 音素评价而非纯声学评价？

**背景**：最初版本使用 MFCC+DTW 声学特征对比进行发音评价。这种方式虽然经典，但存在以下问题：

| 问题 | MFCC+DTW | G2P+ASR（新方案） |
|------|---------|-----------------|
| 需要参考音频 | ✅ 必需 | ❌ 不需要 |
| 语义理解 | ❌ 无法理解内容 | ✅ 知道学生说了什么 |
| 错误定位 | ❌ 只能给总分 | ✅ 精确到音素 |
| 教育价值 | 低 | 高（知道哪里错了、错在什么类型） |

**决策**：保留两种方案供不同场景使用，但推荐优先使用 G2P+ASR 方案。

### 5.2 为什么 TTS 使用单例 + 锁？

Qwen3-TTS 模型加载到 GPU 显存后，模型对象是有状态的。如果每个请求都创建新的模型实例，会导致：

1. 显存爆炸：每个实例占用 2-6 GB 显存
2. 加载时间：模型加载需要 20-60 秒
3. 并发冲突：GPU 上的并发推理导致结果错乱

**解决方案**：单例模式确保全局只有一个模型实例，`asyncio.Lock` 序列化请求访问。

### 5.3 为什么使用 SequenceMatcher 而非编辑距离？

在音素评价的词级对齐和音素比对中，使用 `difflib.SequenceMatcher` 而非简单的编辑距离算法，因为：

1. SequenceMatcher 天然支持"相等/替换/删除/插入"四类操作
2. 能处理不等长序列的自动对齐
3. 返回的结构化 `opcodes` 可直接映射为评价结果中的错误类型

### 5.4 为什么模型加载失败会自动降级？

```python
# 优先 0.6B → 失败 → 降级 1.7B
```

这个设计考虑了以下实际场景：

- 不同环境的显存限制不同
- 模型文件可能因下载中断而不完整
- HuggingFace Hub 在内网环境中可能无法访问
- 本地模型和远程模型可互为备份

### 5.5 为什么 Python 服务独立部署而非嵌入？

将 TTS/ASR 服务设计为独立进程而非嵌入式库，原因如下：

1. **GPU 资源独占**：深度学习模型在 GPU 上需要稳定的显存和推理时间
2. **技术栈解耦**：主应用可用 Java/Node.js 等开发，通过 REST API 调用
3. **独立扩缩容**：语音服务可按需独立扩展（如增加 GPU 实例）
4. **故障隔离**：模型推理崩溃不影响主应用

---

## 6. 实现流程详解

### 6.1 TTS 语音合成流程

```
客户端请求
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  POST /api/v1/tts/synthesize                                 │
│  { text, voice, language, instructions, output_format }      │
└─────────────────────────────────────────────────────────────┘
    │
    ├── 格式校验：output_format 必须为 "wav"
    │
    ├── 获取 asyncio.Lock
    │
    ├── TTSModelManager.generate(text, voice, language, instructions, streaming=False)
    │   │
    │   └── Qwen3TTSModel.generate_custom_voice(
    │           text=text,
    │           language=language,
    │           speaker=voice,
    │           instruct=instructions,
    │           non_streaming_mode=True
    │       )
    │       │
    │       └── 返回 (wavs_list, sample_rate)
    │
    ├── 释放 Lock
    │
    ├── 写入临时 WAV 文件 (soundfile)
    │
    └── 返回 FileResponse (audio/wav) + BackgroundTask 清理临时文件
```

### 6.2 ASR 语音识别流程

```
客户端请求
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  POST /api/v1/asr/transcribe                                 │
│  multipart: audio + language + task + beam_size + word_timestamps │
└─────────────────────────────────────────────────────────────┘
    │
    ├── 格式校验（扩展名白名单：.wav/.mp3/.flac/.m4a）
    ├── 空文件检查
    ├── 音频标准化：soundfile 重编码为 16-bit PCM WAV
    │
    ├── ASRModelManager.transcribe(audio_path, language, task, beam_size, word_timestamps)
    │   │
    │   ├── WhisperModel.transcribe()
    │   │   │
    │   │   ├── 语言自动检测（language 为空时）
    │   │   ├── CTranslate2 加速推理
    │   │   └── 返回 segments 生成器
    │   │
    │   └── 组装结果 dict（text + segments + words）
    │
    └── 返回 JSON 结果 + 清理临时文件
```

### 6.3 发音评价流程（MFCC+DTW）

```
客户端请求
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  POST /api/v1/pronunciation/score                            │
│  multipart: student_audio + reference_audio                  │
└─────────────────────────────────────────────────────────────┘
    │
    ├── 格式校验 + 保存临时文件
    │
    ├── PronunciationEvaluator.pronunciation_score(student_path, ref_path)
    │   │
    │   ├── 1. extract_mfcc(student_audio)
    │   │       → librosa.load() → MFCC(13维) → 转置
    │   │
    │   ├── 2. extract_mfcc(reference_audio)
    │   │       → librosa.load() → MFCC(13维) → 转置
    │   │
    │   ├── 3. fastdtw(student_mfcc, reference_mfcc)
    │   │       → DTW 距离
    │   │
    │   ├── 4. score = max(0, 100 - (distance / 1000) * 100)
    │   │
    │   └── 5. 返回 {score, distance, level, feedback}
    │
    └── 返回 JSON + 清理临时文件
```

### 6.4 音素发音评价流程（G2P+ASR）

这是**最核心、最推荐**的流程：

```
客户端请求
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  POST /api/v1/pronunciation/phoneme-score                    │
│  multipart: student_audio + reference_text + language        │
└─────────────────────────────────────────────────────────────┘
    │
    ├── 格式校验 + 空文件校验 + 时长校验(>=0.3s)
    ├── 音频标准化（重编码为 16-bit PCM WAV）
    │
    └── PhonemeEvaluator.evaluate(audio_path, reference_text, language)
        │
        ├── Step 1: ASR 转录学生录音
        │   └── asr_model.transcribe(audio_path, word_timestamps=True)
        │       └── 返回 spoken_text + spoken_words_raw（含时间戳）
        │
        ├── Step 2: G2P 转换参考文本
        │   └── g2p.text_to_word_phoneme_pairs(reference_text)
        │       └── ref_word_phonemes = [(word, [phonemes]), ...]
        │
        ├── Step 3: G2P 转换 ASR 文本
        │   └── g2p.text_to_word_phoneme_pairs(spoken_text)
        │       └── asr_word_phonemes = [(word, [phonemes]), ...]
        │
        ├── Step 4: 词级对齐
        │   └── SequenceMatcher(ref_words, asr_words).get_opcodes()
        │       └── alignment = [(ref_idx, asr_idx), ...]
        │
        ├── Step 5: 逐词音素比对
        │   └── 对每个对齐对：
        │       ├── SequenceMatcher(ref_phons, asr_phons).get_opcodes()
        │       ├── 检测 substitution / deletion / insertion
        │       ├── 计算该词音素准确率
        │       └── 综合 status: correct/mispronounced/missing/extra
        │
        └── Step 6: 综合评分
            ├── phoneme_accuracy = correct_phonemes / total_expected
            ├── overall_score = phoneme_accuracy*100 - missing_penalty - extra_penalty
            ├── level = excellent/good/fair/poor/very_poor
            └── feedback = 中文反馈建议 + 需重点练习的词汇
```

---

## 7. 依赖详解

### 7.1 Python 依赖清单

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `fastapi` | 最新 | Web 框架，构建 RESTful API 和 WebSocket |
| `uvicorn` | 最新 | ASGI 服务器，运行 FastAPI 应用 |
| `pydantic` | 最新 | 数据验证和请求模型定义 |
| `starlette` | 最新 | FastAPI 依赖的 ASGI 工具包 |
| `soundfile` | 最新 | 音频文件读写（WAV 格式） |
| `numpy` | 最新 | 数值计算，音频数据处理 |
| `scipy` | 最新 | 科学计算，fastdtw 依赖 |
| `transformers` | ==4.57.3 | HuggingFace Transformers 库 |
| `torch` | ==2.7.0+cu128 | 深度学习框架 |
| `qwen_tts` | 最新 | Qwen3-TTS 模型 Python 接口 |
| `faster_whisper` | 最新 | CTranslate2 加速的 Whisper ASR |
| `librosa` | 最新 | 音频分析（MFCC 特征提取） |
| `fastdtw` | 最新 | 动态时间规整算法 |
| `g2p-en` | 最新 | 英文 Grapheme-to-Phoneme 转换 |
| `pypinyin` | 最新 | 中文拼音转换 |
| `gradio` | 最新 | 调试用可视化 UI |
| `modelscope` | 最新 | 模型下载（备用源） |

### 7.2 依赖关系图

```
server.py
  ├── model_loader.py ─────────── torch, qwen_tts, transformers
  ├── asr_model_loader.py ─────── torch, faster_whisper
  ├── pronunciation_evaluator.py ─ librosa, numpy, fastdtw, scipy
  └── phoneme_evaluator.py
        ├── g2p_engine.py ─────── g2p-en (NLTK), pypinyin
        └── asr_model_loader.py ── (同上)
```

### 7.3 非必需依赖说明

- **flash-attn**: 不是必需的，项目已使用 `attn_implementation="sdpa"` 代替。相关警告可忽略。
- **sox**: 仅影响部分音频处理功能，非必需。若系统未安装 SoX 可忽略相关警告。

---

## 8. 配置与环境变量

### 8.1 环境变量清单

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `QWEN_TTS_MODEL_PATH` | `./models/qwen-0.6b` | 本地 TTS 模型路径 |
| `QWEN_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | HuggingFace TTS 模型 ID |
| `WHISPER_MODEL_SIZE` | `base` | Faster-Whisper 模型大小 |
| `RELOAD` | `0` | 是否启用热重载（`1`/`true` 时启用） |
| `HF_HUB_DISABLE_SSL_VERIFY` | （未设置） | 禁用 SSL 验证（内网环境） |
| `CURL_CA_BUNDLE` | （未设置） | 自定义 CA 证书路径 |

### 8.2 模型路径配置

```
models/
├── qwen-0.6b/           # 默认主模型（轻量，~2-3 GB 显存）
│   ├── config.json
│   ├── configuration.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors      # 模型权重（安全张量格式）
│   ├── preprocessor_config.json
│   ├── README.md
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── speech_tokenizer/
│       ├── config.json
│       ├── configuration.json
│       ├── model.safetensors
│       └── preprocessor_config.json
│
└── qwen-1.7b/           # 降级备选模型（高质量，~4-6 GB 显存）
    └── ... (结构同上)
```

### 8.3 音色克隆数据目录

```
voices/                  # 音色克隆数据（运行时生成）
├── cloned_xxxx.json     # 每个文件代表一个克隆音色
└── ...
```

---

## 9. 部署指南

### 9.1 本地部署

```bash
# 1. 克隆项目
git clone <repo_url>
cd MemoryServerTTS

# 2. 创建并激活虚拟环境
conda create -n memory-tts python=3.12
conda activate memory-tts

# 3. 安装依赖
pip install -r requirements.txt

# 4. 确保模型文件已下载到 models/ 目录

# 5. 启动服务
python main.py
# 或指定热重载模式
set RELOAD=1 && python main.py
```

### 9.2 Docker 部署

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "src/server.py"]
```

```bash
# 构建镜像
docker build -t memory-tts .

# 运行容器（GPU 模式）
docker run -d -p 8000:8000 --gpus all memory-tts

# 运行容器（CPU 模式）
docker run -d -p 8000:8000 memory-tts
```

### 9.3 生产环境建议

1. **GPU 选择**: 推荐 NVIDIA T4 / A10 / A100，至少 8 GB 显存
2. **反向代理**: 使用 Nginx 作为反向代理，提供 SSL 终止和负载均衡
3. **自动重启**: 使用 systemd / supervisor 管理进程，崩溃后自动重启
4. **监控**: 定期调用 `/api/v1/health` 接口检查服务状态
5. **模型预热**: 服务启动后，建议发送一次预热请求，避免第一个请求超时
6. **多实例**: 如需更高并发，可在多张 GPU 上部署多个实例，前加负载均衡

---

## 10. API 接口速查

### 10.1 TTS 文本转语音

| 接口 | 方法 | 请求格式 | 响应格式 |
|------|------|---------|---------|
| `/api/v1/tts/synthesize` | POST | JSON | audio/wav 文件 |
| `/api/v1/tts/stream` | WebSocket | JSON 文本帧 | JSON 音频帧 |
| `/api/v1/tts/voices` | GET | 无 | JSON 音色列表 |
| `/api/v1/tts/clone` | POST | multipart/form-data | JSON |

### 10.2 ASR 语音识别

| 接口 | 方法 | 请求格式 | 响应格式 |
|------|------|---------|---------|
| `/api/v1/asr/transcribe` | POST | multipart/form-data | JSON |
| `/api/v1/asr/models` | GET | 无 | JSON |

### 10.3 发音评价

| 接口 | 方法 | 请求格式 | 响应格式 | 是否需要参考音频 |
|------|------|---------|---------|----------------|
| `/api/v1/pronunciation/score` | POST | multipart/form-data | JSON | ✅ 需要 |
| `/api/v1/pronunciation/batch-score` | POST | JSON 数组 | JSON | ✅ 需要 |
| `/api/v1/pronunciation/phoneme-score` | POST | multipart/form-data | JSON (camelCase) | ❌ **不需要** |
| `/api/v1/pronunciation/phoneme-score-with-text` | POST | multipart/form-data | JSON | ❌ 不需要 |
| `/api/v1/pronunciation/phoneme-batch-score` | POST | JSON 数组 | JSON | ❌ 不需要 |

### 10.4 系统

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 返回服务状态、GPU 信息、显存使用等 |
| `/docs` | GET | FastAPI 自动生成的 Swagger 文档 |
| `/redoc` | GET | ReDoc 格式的 API 文档 |

---

## 11. 开发与调试

### 11.1 调试 UI

```bash
# 启动 Gradio 调试界面（端口 7860）
python src/debug_ui.py
```

### 11.2 官方测试脚本

```bash
# 运行官方接口测试（需 GPU）
python src/test_official.py
```

### 11.3 curl 测试示例

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 获取音色列表
curl http://localhost:8000/api/v1/tts/voices

# 语音合成
curl -X POST http://localhost:8000/api/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "Ono_Anna", "language": "English"}' \
  --output output.wav

# 语音识别
curl -X POST http://localhost:8000/api/v1/asr/transcribe \
  -F "audio=@test.wav" \
  -F "language=en"

# 音素发音评价
curl -X POST http://localhost:8000/api/v1/pronunciation/phoneme-score \
  -F "student_audio=@test.wav" \
  -F "reference_text=hello world" \
  -F "language=en"
```

### 11.4 热重载开发模式

```bash
# Windows
set RELOAD=1 && python main.py

# Linux/macOS
RELOAD=1 python main.py
```

### 11.5 项目代码规范

- **命名规范**: Python 使用 `snake_case`，API 输出支持转换为 `camelCase`
- **类型注解**: 所有函数参数和返回值均使用类型注解
- **文档字符串**: 关键类和方法有中文/英文 docstring
- **错误处理**: 使用 HTTPException 返回统一错误格式 `{"detail": "..."}`

---

## 12. 常见问题与排错

### 12.1 "flash-attn 未安装" 警告

```
UserWarning: flash-attn is not installed, using pytorch sdpa implementation
```

**原因**: `flash-attn` 是可选优化库，可加速注意力计算但非必需。  
**解决**: 忽略此警告，项目已使用 PyTorch 原生 SDPA 实现替代。

### 12.2 "sox 未找到" 警告

**原因**: 某些音频处理库依赖 SoX。  
**解决**: 如需使用可安装 SoX（`apt install sox` 或下载 Windows 版本），否则忽略。

### 12.3 SSL 证书验证失败

```
requests.exceptions.SSLError: HTTPSConnectionPool: ... certificate verify failed
```

**原因**: 内网环境或代理导致 HuggingFace Hub 的 SSL 验证失败。  
**解决**: 

```python
# test_official.py 中已有处理
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
# 或使用本地模型文件绕过下载
```

### 12.4 CUDA 显存不足

```
RuntimeError: CUDA out of memory
```

**解决**:
1. 使用 0.6B 小模型（默认）
2. 尝试 CPU 模式（减速但可用）
3. 减少 `batch_size` 或使用更小的 dtype
4. 关闭其他占用显存的程序

### 12.5 FastAPI multipart 参数丢失

**现象**: `phoneme-score` 接口返回 400 "student_audio and reference_text are required"。  
**根因**: 非文件参数缺少 `Form()` 声明。  
**修复**: 详见 [docs/phoneme-score-fix.md](phoneme-score-fix.md)。  
**已修复**: 当前代码中所有 multipart 接口的非文件参数均已正确声明 `Form()`。

### 12.6 TTS 模型加载失败

**现象**: 服务启动时模型加载失败。  
**原因及解决**:
1. 本地模型文件不存在 → 检查 `models/` 目录
2. HuggingFace 无法访问 → 设置 `HF_HUB_DISABLE_SSL_VERIFY=1`
3. 显存不足 → 使用 0.6B 小模型
4. 如果 0.6B 和 1.7B 都失败 → 检查 PyTorch 版本兼容性

### 12.7 端口占用

```bash
# 查找占用 8000 端口的进程
netstat -ano | findstr :8000
# 在任务管理器中结束对应 PID 的进程
```

---

## 附录 A：项目文件结构

```
MemoryServerTTS/
│
├── main.py                    # 应用主入口（封装 uvicorn 启动）
├── requirements.txt           # Python 依赖清单
├── Dockerfile                 # Docker 容器化配置
├── README.md                  # 项目简介（中文）
│
├── start_server.bat           # Windows 一键启动脚本
├── start_server.sh            # Linux/macOS 一键启动脚本
│
├── src/                       # 核心源代码
│   ├── __init__.py            # 空文件，包标识
│   ├── server.py              # FastAPI 主服务（API 路由 + 启动生命周期）
│   ├── model_loader.py        # TTS 模型管理器（Qwen3-TTS 单例）
│   ├── asr_model_loader.py    # ASR 模型管理器（Faster-Whisper 单例）
│   ├── pronunciation_evaluator.py  # MFCC+DTW 发音评价器
│   ├── phoneme_evaluator.py   # G2P+ASR 音素评价器（核心创新）
│   ├── g2p_engine.py          # G2P 引擎（英文/中文策略模式）
│   ├── debug_ui.py            # Gradio 可视化调试界面
│   └── test_official.py       # 官方接口测试脚本
│
├── models/                    # AI 模型文件
│   ├── qwen-0.6b/             # Qwen3-TTS 0.6B 轻量模型
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   ├── ... (其余配置文件)
│   │   └── speech_tokenizer/  # 语音分词器子模型
│   │       ├── config.json
│   │       └── model.safetensors
│   │
│   └── qwen-1.7b/             # Qwen3-TTS 1.7B 高质量模型
│       └── ... (结构同上)
│
├── voices/                    # 音色克隆数据目录（运行时生成）
│
└── docs/                      # 文档目录
    ├── API_DOCUMENTATION.md         # API 集成文档（面向 SpringBoot 开发者）
    ├── PROJECT_DOCUMENTATION.md     # 本项目技术文档（当前文件）
    ├── phoneme-score-fix.md         # 音素评分接口修复指南
    └── TROUBLESHOOTING_SPRINGBOOT.md # SpringBoot 接入排错指南
```

## 附录 B：音色速查表

| 音色 ID | 语言 | 性别 | 风格描述 |
|---------|------|------|---------|
| `vivian` | 中文 | 女 | 明亮、略带锋芒 |
| `serena` | 中文 | 女 | 温暖、温柔 |
| `uncle_fu` | 中文 | 男 | 低沉柔和，经验丰富 |
| `dylan` | 中文（北京话） | 男 | 清晰自然 |
| `eric` | 中文（四川话） | 男 | 活泼、沙哑明亮 |
| `Ono_Anna` | 英文 | 男 | 充满活力、节奏感强 |
| `aiden` | 英文 | 男 | 阳光、中音清晰 |
| `ono_anna` | 日文 | 女 | 轻盈灵巧 |
| `sohee` | 韩文 | 女 | 情感丰富 |

> **注意**: `Ono_Anna`（英文男声）和 `ono_anna`（日文女声）是大小写不同的两个独立音色。

## 附录 C：发音评分等级对照

| 分数范围 | 等级 | 教学建议 |
|---------|------|---------|
| 90-100 | excellent | 发音非常标准，继续保持！ |
| 75-89 | good | 发音良好，注意个别音节的准确性 |
| 60-74 | fair | 发音基本正确，需要加强练习 |
| 40-59 | poor | 发音有待改进，建议多听标准发音 |
| 0-39 | very_poor | 发音需要大幅改进，建议从基础音标开始练习 |
