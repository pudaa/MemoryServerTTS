# OCR 模块集成文档

## 概述

MemoryServerTTS 已集成基于 **PaddleOCR (PP-OCRv6)** 的 OCR 文字识别功能，支持图片和 PDF 文档的文字提取，适用于英语学习场景中的作文/听写结果图片转文字。

## 当前状态

| 项目 | 状态 |
|------|------|
| PaddleOCR 3.7.0 | ✅ 已安装 |
| onnxruntime-gpu 1.21.0 | ✅ CUDA 12.x GPU 加速 |
| 默认模型档位 | ✅ **Small** (PP-OCRv6_small) |
| 推理速度 | ✅ ~450ms/张 (GPU) |
| 准确率 | ✅ 96.9% (英文) |
| 预处理 | ✅ orientation + unwarping (essay 预设) |

## 模型档位

| 档位 | 检测模型 | 识别模型 | 速度 | 精度 | 场景 |
|------|----------|----------|------|------|------|
| tiny | PP-OCRv6_tiny_det | PP-OCRv6_tiny_rec | 最快 | 较低 | 移动端/实时 |
| **small** ← 默认 | PP-OCRv6_small_det | PP-OCRv6_small_rec | 快 | 高 | **英语学习推荐** |
| medium | PP-OCRv6_medium_det | PP-OCRv6_medium_rec | 中 | 最高 | 服务端批量 |

切换方式：
```python
from src.ocr_config import OCRConfig
cfg = OCRConfig()
cfg.model_tier = "medium"  # 切换到 Medium
```
或修改 `config/ocr_config.yaml` 中的 `ocr.model_tier` 字段。
或设置环境变量：`$env:OCRCONF_MODEL_TIER = "tiny"`
```python
app.state.ocr_engine = OCREngine(engine="paddle_static", lang="en", device="gpu")
```

> ⚠️ 注意：PaddlePaddle 可能与 PyTorch 存在依赖冲突。如遇到问题，请使用方案 B 或 C。

### 方案 B：Python 3.12 独立环境

```powershell
# 创建 Python 3.12 环境专门用于 OCR
conda create -n ocr-gpu python=3.12 -y
conda activate ocr-gpu
pip install paddleocr onnxruntime-gpu
```

然后修改 `src/server.py` 中的 OCR 引擎配置使用 HTTP 调用独立服务（或直接在 3.12 环境下运行整个项目）。

### 方案 C：Docker 部署（最稳定）

```powershell
docker run -it --rm --gpus all --network host ^
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu
```

## API 接口

### 1. 图片 OCR 扫描

```http
POST /api/v1/ocr/scan
Content-Type: multipart/form-data
```

**参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image | file | 是 | 图片文件 (PNG/JPG/JPEG/BMP/TIFF/WEBP) |
| language | string | 否 | 识别语言 ("en"/"ch"/"Multilingual")，默认 "en" |

**响应示例：**
```json
{
    "success": true,
    "text": "Hello world\nThis is a test",
    "lines": ["Hello world", "This is a test"],
    "boxes": [[10, 20, 200, 50], [10, 60, 200, 90]],
    "confidences": [0.985, 0.972],
    "avg_confidence": 0.9785,
    "language": "en",
    "processing_time_ms": 1874.0
}
```

### 2. 文档 OCR 扫描 (PDF)

```http
POST /api/v1/ocr/scan-file
Content-Type: multipart/form-data
```

**参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | PDF 文件 |
| language | string | 否 | 识别语言 |

**响应示例：**
```json
{
    "success": true,
    "total_pages": 3,
    "pages": [
        {"page": 1, "text": "...", "lines": [...]},
        {"page": 2, "text": "...", "lines": [...]}
    ],
    "full_text": "...",
    "processing_time_ms": 5234.0
}
```

### 3. OCR 健康检查

```http
GET /api/v1/ocr/health
```

**响应示例：**
```json
{
    "ocr_available": true,
    "engine": "onnxruntime",
    "language": "en",
    "device_requested": "gpu",
    "device_actual": "cpu",
    "ocr_version": "PP-OCRv6",
    "supported_image_formats": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"],
    "supported_doc_formats": [".pdf"]
}
```

## 快速测试

### 命令行测试

```powershell
conda activate memory-tts
python src/test_ocr_quick.py
```

### cURL 测试

```bash
# 健康检查
curl http://localhost:8000/api/v1/ocr/health

# OCR 扫描
curl -X POST http://localhost:8000/api/v1/ocr/scan \
  -F "image=@your_essay.png" \
  -F "language=en"
```

### Python 客户端示例

```python
import requests

# OCR 扫描图片
with open("essay.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/scan",
        files={"image": f},
        data={"language": "en"}
    )

result = response.json()
print(f"识别文本:\n{result['text']}")
print(f"置信度: {result['avg_confidence']:.2%}")
```

## 文件结构

```
src/
├── ocr_engine.py       # OCR 引擎封装
├── server.py           # FastAPI 服务（已添加 OCR 端点）
├── test_ocr.py         # 完整验证脚本
└── test_ocr_quick.py   # 快速测试脚本
requirements-ocr.txt    # OCR 专用依赖
```

## 故障排除

### onnxruntime GPU 不可用

**现象：** 日志显示 "GPU 不可用，自动降级到 CPU"

**原因：** Python 3.13 缺少 onnxruntime-gpu 预编译 wheel

**解决：** 参考上方 GPU 加速方案

### PaddleOCR 导入失败

```powershell
pip install -r requirements-ocr.txt
```

### 模型下载失败

```python
import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
```
