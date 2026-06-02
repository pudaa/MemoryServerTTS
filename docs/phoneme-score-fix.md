# MemoryServerTTS — 发音评价接口修复指南

> **问题**：`POST /api/v1/pronunciation/phoneme-score` 始终返回 400 `"student_audio and reference_text are required"`  
> **根因**：FastAPI 的 `multipart/form-data` 处理规则——非文件参数必须显式声明 `Form()` 才能从表单字段中读取

---

## 根因分析

当前代码（`src/server.py`）：

```python
@app.post("/api/v1/pronunciation/phoneme-score")
async def phoneme_score(
    student_audio: UploadFile = File(...),
    reference_text: str = "",           # ❌ 缺少 Form()
    language: str | None = None,        # ❌ 缺少 Form()
):
    if student_audio.filename is None or not reference_text.strip():
        raise HTTPException(status_code=400, detail="student_audio and reference_text are required")
```

在 FastAPI 中，当请求的 `Content-Type` 为 `multipart/form-data` 时：
- `File(...)` 参数 → 从文件字段读取 ✅
- **不带 `Form()` 的参数** → 被当作 **query parameter** 或 **JSON body parameter**，**不会从 form 字段中读取** ❌

因此无论 SpringBoot 端如何正确发送 `reference_text=xxx`，Python 端收到的 `reference_text` 始终是默认值 `""`，`not "".strip()` 为 `True`，触发 400。

## 修复方案

在所有非文件参数前加 `Form()`：

```python
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

@app.post("/api/v1/pronunciation/phoneme-score")
async def phoneme_score(
    student_audio: UploadFile = File(...),
    reference_text: str = Form(...),        # ✅ 从 form 字段读取，必填
    language: str | None = Form(None),      # ✅ 从 form 字段读取，可选
):
    if student_audio.filename is None or not reference_text.strip():
        raise HTTPException(status_code=400, detail="student_audio and reference_text are required")
    if not student_audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(student_audio.filename).suffix) as temp_file:
        content = await student_audio.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        result = app.state.phoneme_evaluator.evaluate(
            audio_path=temp_path,
            reference_text=reference_text,
            language=language,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_file(temp_path)
```

## 同样需要修复的接口

检查项目中所有同时包含 `File(...)` 和字符串参数的端点，确保字符串参数有 `Form()`：

### `phoneme-score-with-text`

```python
@app.post("/api/v1/pronunciation/phoneme-score-with-text")
async def phoneme_score_with_text(
    student_audio: UploadFile = File(...),
    reference_text: str = Form(...),       # 加 Form()
    language: str | None = Form(None),     # 加 Form()
):
```

### ASR 接口（可能也有类似问题）

```python
@app.post("/api/v1/asr/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),     # 加 Form()
    task: str = Form("transcribe"),        # 加 Form()
    beam_size: int = Form(5),              # 加 Form()
    word_timestamps: bool = Form(False),   # 加 Form()
):
```

---

## 验证方法

修复后重启 Python 服务，用 curl 验证：

```bash
# 创建测试音频
python -c "
import wave, struct
with wave.open('test.wav', 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack('<' + 'h'*16000, *[0]*16000))
"

# 测试
curl -X POST http://localhost:8000/api/v1/pronunciation/phoneme-score \
  -F "student_audio=@test.wav" \
  -F "reference_text=hello world" \
  -F "language=en"
```

预期返回 200 + JSON 评分结果。

---

> **总结**：FastAPI multipart 混用 `File` 和字符串时必须全部显式声明类型——`File()` 给文件，`Form()` 给字段。少一个 `Form()` 就收不到值。
