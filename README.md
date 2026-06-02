# MemoryServerTTS

基于 **Qwen3-TTS** 和 **Faster-Whisper** 的语音服务，提供文本转语音（TTS）、语音识别（ASR）以及发音评价三大核心功能。

## 启动方法

1. 安装依赖（建议使用虚拟环境/conda）：
   ```sh
   pip install -r requirements.txt
   ```
2. 下载模型（已预置在 `models/` 目录下）：
   - `models/qwen-0.6b/` — 默认主模型（轻量，显存约 2-3 GB）
   - `models/qwen-1.7b/` — 降级备选模型（效果更好，显存约 4-6 GB）
3. 启动服务：
   ```sh
   python src/server.py
   # 或
   python main.py
   # 或（热重载模式，开发调试用）
   set RELOAD=1 && python main.py   # Windows
   RELOAD=1 python main.py          # Linux/macOS
   ```

## 目录结构

```
MemoryServerTTS/
├── main.py              # 启动入口
├── requirements.txt     # 依赖列表
├── Dockerfile           # Docker 构建
├── start_server.bat     # Windows 启动脚本
├── start_server.sh      # Linux 启动脚本
├── src/
│   ├── server.py                   # FastAPI 主服务
│   ├── model_loader.py             # TTS 模型管理器（Qwen3-TTS）
│   ├── asr_model_loader.py         # ASR 模型管理器（Faster-Whisper）
│   ├── pronunciation_evaluator.py  # 发音评价（MFCC+DTW，需参考音频）
│   ├── phoneme_evaluator.py        # 🔥 音素评价（G2P+ASR，仅需参考文本）
│   ├── g2p_engine.py               # G2P 引擎（英文/中文）
│   ├── debug_ui.py                 # Gradio 调试界面（端口 7860）
│   └── test_official.py            # 官方接口测试脚本
├── models/qwen-0.6b/   # 0.6B 小模型（默认）
├── models/qwen-1.7b/   # 1.7B 大模型（降级方案）
├── voices/             # 音色克隆数据目录
└── doc/                # 详细文档
```

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/health` | 健康检查 |
| `GET` | `/api/v1/tts/voices` | 获取音色列表 |
| `POST` | `/api/v1/tts/synthesize` | 文本合成语音（WAV） |
| `WebSocket` | `/api/v1/tts/stream` | 流式语音合成 |
| `POST` | `/api/v1/tts/clone` | 音色克隆（模拟） |
| `POST` | `/api/v1/asr/transcribe` | 语音识别（Faster-Whisper） |
| `GET` | `/api/v1/asr/models` | 支持的 ASR 模型列表 |
| `POST` | `/api/v1/pronunciation/score` | 🔇 MFCC+DTW 发音评分（需参考音频） |
| `POST` | `/api/v1/pronunciation/batch-score` | 批量 MFCC+DTW 评分 |
| `POST` | `/api/v1/pronunciation/phoneme-score` | 🔥 音素对齐发音评分（仅需参考文本，推荐） |
| `POST` | `/api/v1/pronunciation/phoneme-score-with-text` | 同上（别名接口） |
| `POST` | `/api/v1/pronunciation/phoneme-batch-score` | 批量音素评分 |

## 常见问题

- `flash-attn` 不是必须，可忽略相关警告。
- Windows 下建议直接用 PyTorch 原生注意力机制（`attn_implementation="sdpa"`）。
- 若报 `sox` 未找到，仅影响部分音频处理功能，可忽略或手动安装 SoX。
- 推荐使用 Python 3.10+，PyTorch 2.0+。
- Gradio 调试 UI 默认运行在 `http://localhost:7860`，通过 `python src/debug_ui.py` 启动。

## 详细文档

请参见 [doc/API_DOCUMENTATION.md](doc/API_DOCUMENTATION.md) 获取完整的 API 使用说明。
