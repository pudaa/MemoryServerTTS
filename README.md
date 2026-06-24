# MemoryServerTTS

基于 **Qwen3-TTS** + **Faster-Whisper** + **PaddleOCR** 的 AI 语音与视觉服务平台，为英语学习 APP 提供四大核心能力。

## 快速开始

```powershell
# 1. 环境
conda create -n memory-tts python=3.12 -y
conda activate memory-tts

# 2. 依赖
pip install -r requirements.txt

# 3. 启动
python main.py
```

管理后台: http://localhost:8000/admin | API 文档: http://localhost:8000/docs

## 项目结构

```
MemoryServerTTS/
├── main.py                       # 启动入口
├── requirements.txt              # Python 依赖
├── config/ocr.yaml               # OCR 配置
├── src/
│   ├── server.py                 # FastAPI 主入口
│   ├── common/                   # 公共基础设施
│   │   ├── base_config.py        # 统一配置基类
│   │   └── logging.py            # 统一日志 [MODULE] 前缀
│   ├── tts/                      # TTS 文本转语音
│   ├── asr/                      # ASR 语音识别
│   ├── pronunciation/            # 发音评价
│   ├── ocr/                      # OCR 文字识别
│   └── dashboard/                # 管理后台
├── models/
│   ├── qwen-1.7b/                # TTS 主模型（默认）
│   └── qwen-0.6b/                # TTS 降级方案
├── tests/                        # 测试脚本
└── docs/                         # 详细文档
```

## API 总览

| 方法 | 路径 | 模块 | 说明 |
|------|------|------|------|
| `GET` | `/api/v1/health` | 系统 | 健康检查 |
| `GET` | `/admin` | 面板 | 管理后台 |
| `GET` | `/api/v1/tts/voices` | TTS | 音色列表 |
| `POST` | `/api/v1/tts/synthesize` | TTS | 文本合成语音 |
| `WS` | `/api/v1/tts/stream` | TTS | 流式合成 |
| `POST` | `/api/v1/asr/transcribe` | ASR | 语音转文字 |
| `POST` | `/api/v1/pronunciation/phoneme-score` | 发音 | 音素评价（仅需文本） |
| `POST` | `/api/v1/pronunciation/score` | 发音 | MFCC+DTW 评价 |
| `POST` | `/api/v1/ocr/scan` | OCR | 图片文字提取 |
| `POST` | `/api/v1/ocr/scan-file` | OCR | 文档文字提取 |
| `GET` | `/api/v1/ocr/health` | OCR | OCR 状态 |

## 模型配置

| 模块 | 模型 | 显存 | 配置方式 |
|------|------|------|----------|
| TTS | Qwen3-TTS 1.7B | ~3.9GB | `QWEN_TTS_MODEL_PATH` 环境变量 |
| ASR | Faster-Whisper base | ~1GB | `WHISPER_MODEL_SIZE` 环境变量 |
| OCR | PP-OCRv6 Small | ~0.5GB | `config/ocr.yaml` 或 `OCRCONF_MODEL_TIER` |
| 发音 | G2P + ASR | 复用 ASR | — |

## 硬件要求

| 组件 | 最低 | 推荐（当前） |
|------|------|-------------|
| GPU | 6GB VRAM | 8GB (RTX 4060 Laptop) |
| RAM | 8GB | 16GB |
| Python | 3.10+ | 3.12 |
| CUDA | 11.8+ | 12.x |

## 性能参考

| 操作 | 耗时 | 说明 |
|------|------|------|
| TTS（短句） | ~10–15s | 自回归模型限制，建议异步调用 |
| ASR（1分钟） | ~2–5s | Faster-Whisper base |
| OCR（图片） | ~450ms | PP-OCRv6 Small + GPU |
| 发音评价 | ~1–3s | G2P + ASR 音素对齐 |

## 文档

- [API 集成文档](docs/API_DOCUMENTATION.md)
- [OCR 集成文档](docs/OCR_INTEGRATION.md)
- [项目技术文档](docs/PROJECT_DOCUMENTATION.md)
- [故障排查](docs/TROUBLESHOOTING_SPRINGBOOT.md)
