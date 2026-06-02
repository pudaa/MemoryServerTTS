# SpringBoot 接入发音纠正接口排错指南

> **最终根因**（2026-05-22 已修复）：Python 端 `phoneme-score` 接口的 `reference_text` 和 `language` 参数缺少 `Form()` 声明，导致 FastAPI 在 `multipart/form-data` 中无法读取这些非文件字段，始终返回默认值。**已在 `src/server.py` 中修复**，所有 multipart 接口的非文件参数现在都加上了 `Form()`。

> ~~**旧版错误现象**~~：调用 `/api/v1/pronunciation/phoneme-score` 返回 `400 Bad Request`，响应体为：
> ```json
> {"detail": "student_audio and reference_text are required"}
> ```

---

## 根因分析

Python 服务端接口定义为：

```python
@app.post("/api/v1/pronunciation/phoneme-score")
async def phoneme_score(
    student_audio: UploadFile = File(...),     # 必填，文件上传
    reference_text: str = "",                   # 必填，参考文本
    language: str | None = None,                # 可选，语言代码
):
```

FastAPI 使用 **multipart/form-data** 接收请求，要求：

| 字段名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `student_audio` | File | ✅ | 音频文件（.wav/.mp3/.flac/.m4a） |
| `reference_text` | String | ✅ | 参考文本（非空） |
| `language` | String | ❌ | 语言代码，如 "en"、"zh" |

返回 400 的条件（二选一触发）：
1. `student_audio` 字段**缺省**或为**空文件** → `student_audio.filename is None`
2. `reference_text` 为空字符串 → `not reference_text.strip()`

---

## 最可能的错误原因

根据 SpringBoot 调用链路 `multipartPost()` → `evaluatePhonemeScore()`，最常见的问题是：

### ❌ 错误 1：字段名不匹配

```java
// 错误示范 —— 字段名对不上 Python 端的定义
body.add("file", new FileSystemResource(audioFile));    // ❌ 应为 "student_audio"
body.add("text", referenceText);                        // ❌ 应为 "reference_text"
```

### ❌ 错误 2：Content-Type 设置错误

```java
// 错误示范 —— 手动设置 Content-Type 会覆盖 multipart 的 boundary
HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.MULTIPART_FORM_DATA);  // ❌ 缺少 boundary

// 正确方式：让 RestTemplate 自动生成 boundary
// 不手动设置 Content-Type，或使用 FormHttpMessageConverter
```

### ❌ 错误 3：文件作为 byte[] 发送

```java
// 错误示范 —— 以字节数组形式发送，fastapi 无法识别为文件
body.add("student_audio", audioBytes);  // ❌ 没有文件名信息
```

---

## ✅ 正确的 SpringBoot 调用实现

### 方案 A：RestTemplate（推荐，无需额外依赖）

```java
@Service
public class PronunciationServiceImpl {

    private final RestTemplate restTemplate;

    @Value("${tts.service.url:http://localhost:8000}")
    private String baseUrl;

    /**
     * 音素对齐发音评分
     *
     * @param audioFile      学生录音文件（本地临时文件）
     * @param referenceText  参考文本（期望朗读的内容）
     * @param language       语言代码，如 "en"、"zh"（可选）
     */
    public JSONObject evaluatePhonemeScore(File audioFile,
                                           String referenceText,
                                           String language) {
        String url = baseUrl + "/api/v1/pronunciation/phoneme-score";

        // 1. 构建 multipart form 数据
        LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("student_audio", new FileSystemResource(audioFile));
        body.add("reference_text", referenceText);
        if (language != null && !language.isBlank()) {
            body.add("language", language);
        }

        // 2. 构建请求（不要手动设置 Content-Type，让 RestTemplate 自动生成）
        HttpEntity<LinkedMultiValueMap<String, Object>> requestEntity =
                new HttpEntity<>(body);

        // 3. 发送 POST 请求
        ResponseEntity<String> response = restTemplate.postForEntity(
                url, requestEntity, String.class);

        if (response.getStatusCode().is2xxSuccessful()) {
            return JSON.parseObject(response.getBody());
        } else {
            throw new RuntimeException(
                    "Python service returned " + response.getStatusCode()
                    + ": " + response.getBody());
        }
    }
}
```

### RestTemplate Bean 配置

```java
@Configuration
public class RestTemplateConfig {

    @Bean
    public RestTemplate restTemplate() {
        RestTemplate restTemplate = new RestTemplate();

        // 确保 multipart 消息转换器已注册
        restTemplate.getMessageConverters().add(
                new FormHttpMessageConverter());

        // 可选：增加超时配置
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(30_000);
        factory.setReadTimeout(60_000);  // TTS 推理可能较慢
        restTemplate.setRequestFactory(factory);

        return restTemplate;
    }
}
```

### 方案 B：WebClient（响应式）

```java
@Service
public class PronunciationServiceWebClient {

    private final WebClient webClient;

    public PronunciationServiceWebClient(
            @Value("${tts.service.url:http://localhost:8000}") String baseUrl) {
        this.webClient = WebClient.create(baseUrl);
    }

    public Mono<JSONObject> evaluatePhonemeScore(
            File audioFile, String referenceText, String language) {

        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("student_audio", new FileSystemResource(audioFile));
        builder.part("reference_text", referenceText);
        if (language != null) {
            builder.part("language", language);
        }

        return webClient.post()
                .uri("/api/v1/pronunciation/phoneme-score")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .bodyValue(builder.build())
                .retrieve()
                .bodyToMono(String.class)
                .map(JSON::parseObject);
    }
}
```

### 方案 C：Apache HttpClient（底层控制）

```java
public JSONObject evaluatePhonemeScoreHttpClient(File audioFile,
                                                  String referenceText,
                                                  String language) throws Exception {
    try (CloseableHttpClient client = HttpClients.createDefault()) {
        HttpPost post = new HttpPost(baseUrl + "/api/v1/pronunciation/phoneme-score");

        MultipartEntityBuilder builder = MultipartEntityBuilder.create();
        builder.addBinaryBody("student_audio", audioFile,
                ContentType.create("audio/wav"), audioFile.getName());
        builder.addTextBody("reference_text", referenceText,
                ContentType.TEXT_PLAIN);
        if (language != null) {
            builder.addTextBody("language", language, ContentType.TEXT_PLAIN);
        }

        post.setEntity(builder.build());

        try (CloseableHttpResponse response = client.execute(post)) {
            String body = EntityUtils.toString(response.getEntity());
            int status = response.getStatusLine().getStatusCode();
            if (status >= 200 && status < 300) {
                return JSON.parseObject(body);
            } else {
                throw new RuntimeException("HTTP " + status + ": " + body);
            }
        }
    }
}
```

---

## 自测方法

在 SpringBoot 中增加一个集成测试来验证字段名是否正确：

```java
@Test
void testPhonemeScoreFields() throws Exception {
    // 创建一个临时 WAV 文件用于测试
    File testAudio = new File("test_silence.wav");
    // ... 确保文件存在 ...

    LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
    body.add("student_audio", new FileSystemResource(testAudio));
    body.add("reference_text", "hello world");
    body.add("language", "en");

    // 打印实际请求体，检查字段名
    System.out.println("Request fields: " + body.keySet());
    // 预期输出: [student_audio, reference_text, language]

    HttpEntity<LinkedMultiValueMap<String, Object>> request = new HttpEntity<>(body);
    ResponseEntity<String> response = restTemplate.postForEntity(
            "http://localhost:8000/api/v1/pronunciation/phoneme-score",
            request, String.class);

    assertEquals(200, response.getStatusCodeValue(),
            "Response: " + response.getBody());
}
```

也可以直接用 curl 验证字段名是否正确：

```bash
# 验证正确请求（应返回 200）
curl -X POST http://localhost:8000/api/v1/pronunciation/phoneme-score \
  -F "student_audio=@test.wav" \
  -F "reference_text=hello world" \
  -F "language=en"

# 验证错误字段名（应返回 400）
curl -X POST http://localhost:8000/api/v1/pronunciation/phoneme-score \
  -F "file=@test.wav" \
  -F "text=hello world"
```

---

## 字段名对照速查表

| SpringBoot 端必须使用的字段名 | 说明 |
|-------------------------------|------|
| `student_audio` | 学生录音文件（**不要**写成 `file`、`audio`、`audioFile`） |
| `reference_text` | 参考文本（**不要**写成 `text`、`referenceText`、`content`） |
| `language` | 语言代码（可选，**不要**写成 `lang`、`languageCode`） |

---

> **总结**：99% 的情况是 SpringBoot 端 `multipartPost()` 中 `body.add("file", ...)` 或 `body.add("text", ...)` 的字段名与 Python 端 `student_audio` / `reference_text` 不匹配导致。请逐一核对。
