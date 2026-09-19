# .backport — faster-qwen3-tts 的 transformers 4.x 兼容副本

**来源**：[andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) **v0.4.0**，MIT License。
**目的**：让 CUDA Graph 加速路径能在本项目现有的 `transformers 4.57.3` 上运行，
避免把 `transformers` 升到 5.x（那会牵动 ASR / OCR / 发音模块，且 `qwen-tts-hf`
会覆盖 `qwen_tts` 模块）。

验证结论与完整数据见 [docs/FASTER_TTS_VALIDATION.md](../docs/FASTER_TTS_VALIDATION.md)。

## 改了什么（共 2 处，均为向后兼容的双分支）

| 文件 | 补丁 | 原因 |
|---|---|---|
| `faster_qwen3_tts/predictor_graph.py` | `lazy_initialization(k, v)` 失败则退回 `(k)` | transformers 4.x 的 `StaticLayer.lazy_initialization` 只接受 1 个参数 |
| `faster_qwen3_tts/talker_graph.py` | 同上 | 同上 |
| `faster_qwen3_tts/talker_graph.py` | 取 K/V 时优先 `.layers[i].keys/.values`，`AttributeError` 则退回 `cache[i]` | 4.57 的 `DynamicCache.__getitem__` 返回 `(k, v)`；5.x 改为 `.keys/.values` |

另有两处非功能性调整：
- `__init__.py` 不再导出 `GGMLQwen3TTS`（GGML 后端需额外原生 wheel，未声明依赖）。
- 删除 `cli.py`（本副本仅供程序调用，不用 CLI）。

`model.py`、`generate.py`、`streaming.py`、`sampling.py`、`utils.py` **未作任何修改**。
所有改动点都带 `# backport:` 注释，便于上游修复后比对/移除。

> 上游对应 PR #65 是**反向**的（把 4.x 写法适配到 5.x）；本副本是把它移植回 4.x。

## 安装与运行

```powershell
# 克隆环境（不要动 memory-tts）
conda create -n memory-tts-bp --clone memory-tts -y

# 关键：--no-deps，否则 transformers 会被抬到 5.x
& "D:\DevVmEnv\Anaconda_envs\envs\memory-tts-bp\python.exe" -m pip install --no-deps "faster-qwen3-tts==0.4.0"

# 运行（脚本内部把本目录插入 sys.path 优先于 site-packages）
$env:NUMBA_CACHE_DIR = "$PWD\.numba-cache"
& "D:\DevVmEnv\Anaconda_envs\envs\memory-tts-bp\python.exe" .\.backport\bench_backport.py --model .\models\qwen-1.7b
```

## 脚本

| 文件 | 用途 |
|---|---|
| `bench_backport.py` | 非流式 / 流式 / 单词语音三段基准，打印 TTFA 与显存 |
| `compare_audio.py` | 对各方案的 wav 做波形客观量 + ASR 回读比对（用主环境跑） |

## 维护提示

- 上游若发布修复了 4.x 兼容的新版本，**优先换回官方包**并删除本目录。
- 上游 `transformers>=5.15.1` 约束没有上界，pip 会解析到 5.17 并触发
  `MimiConfig.rope_theta` 报错；如需在 5.x 上跑，锁 `transformers==5.15.1`。
- 本副本是**原样 vendor**，不做二次开发；业务接入代码不应 import 本目录的私有模块。
