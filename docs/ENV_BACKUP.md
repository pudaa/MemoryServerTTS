# 环境备份与恢复（memory-tts / conda）

> 修改 TTS 性能路径可能带来破坏性影响（依赖升级、换引擎、改 CUDA Graph 等）。
> 本文记录**恢复点在哪、怎么恢复、哪些版本绝对不能漂移**。

## 1. 备份位置

备份放在**仓库之外**，避免污染项目、也避免被 git 操作误删：

```
D:\DevVmEnv\conda-backups\memory-tts\<YYYYMMDD-HHmmss>\
├── environment.yml     # conda env export --no-builds（含 pip 段，144 行）
├── pip-freeze.txt      # pip freeze 精确版本（155 行）
└── conda-list.txt      # conda list 明文（138 行，便于肉眼对比）
```

环境本体位于 `D:\DevVmEnv\Anaconda_envs\envs\memory-tts`（**13.07 GB**，因为含
`torch 2.7.0+cu128` 的 CUDA 运行时；实际恢复耗时主要在这里）。

## 2. 恢复方式

### 方式 A：从规格重建（推荐，占用小，可用于跨机）

```powershell
# 全新环境（不会覆盖现有 memory-tts）
conda env create -f D:\DevVmEnv\conda-backups\memory-tts\<stamp>\environment.yml

# 或者：原环境被改坏后，先删再建
conda deactivate
conda env remove -n memory-tts -y
conda env create -f D:\DevVmEnv\conda-backups\memory-tts\<stamp>\environment.yml
```

⚠️ `environment.yml` 里 `pip` 段的 torch 会带 `+cu128` 本地版本号，`conda env create`
可能拉不到。若失败，用方式 B 的 pip 行单独装 torch。

### 方式 B：先 conda 基础环境，再精确 pip 还原

```powershell
conda create -n memory-tts python=3.12.13 -y
conda activate memory-tts

# 关键：torch 必须锁 CUDA 12.8 轮子（driver 610.47 / CUDA UMD 13.3，向下兼容 cu128）
pip install "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r D:\DevVmEnv\conda-backups\memory-tts\<stamp>\pip-freeze.txt
```

### 方式 C：整目录快照（最彻底，恢复最快）

规格重建**不能**还原本机手工放置的东西（例如 `%USERPROFILE%` 下的 HF/ModelScope 缓存、
手工放的 ffmpeg）。若要"随时完美恢复"，直接复制整个环境目录：

```powershell
# 备份（在服务停止时做，13 GB）
robocopy "D:\DevVmEnv\Anaconda_envs\envs\memory-tts" `
         "D:\DevVmEnv\conda-backups\memory-tts\<stamp>-dirclone" /MIR /R:1 /W:1 /NFL /NDL

# 恢复
robocopy "D:\DevVmEnv\conda-backups\memory-tts\<stamp>-dirclone" `
         "D:\DevVmEnv\Anaconda_envs\envs\memory-tts" /MIR /R:1 /W:1
```

> 目录快照有个已知坑：conda 环境路径写在部分脚本/`conda-meta` 里。如果想克隆成**不同名字**的
> 环境，用 `conda create -n memory-tts-test --clone memory-tts` 更安全。

## 3. 绝对不能漂移的版本（硬约束）

| 包 | 锁定值 | 为什么 |
|----|--------|--------|
| `torch` | `2.7.0+cu128` | driver 610.47 支持 cu128。CUDA Graph 方案要求 ≥2.5.1，2.7.0 满足 |
| `transformers` | `4.57.3` | `qwen-tts 0.1.1` 硬钉 `transformers==4.57.3`。升到 5.x 会破坏 qwen_tts |
| `qwen-tts` | `0.1.1` | 上游最新版（2026-02-06），无更新可升 |
| `onnxruntime-gpu` | `1.21.0`（`<1.27.0`） | ≥1.27 需要 CUDA 13.x，会打断 OCR |
| `faster-whisper` | `1.2.1` | ASR + 发音评价复用 |
| `numpy` | `2.3.5` | librosa/paddleocr 兼容面 |
| `python` | `3.12.13` | — |

### ⚠️ 引入 `faster-qwen3-tts` 的冲突（调研结论，勿盲装）

`faster-qwen3-tts` 走 CUDA Graph + 静态 KV cache，实测能大幅提速；但它依赖
**`qwen-tts-hf`**（`0.1.1.post1`），后者与上游 `qwen-tts` **提供同一个 `qwen_tts` 模块**，
且要求 `transformers>=5.15.1,<6`：

- **两个发行版不能同时装在一个环境里**（会互相覆盖 `qwen_tts/`）。
- 装它就意味着把 `transformers` 从 4.57.3 抬到 5.x，**可能影响 ASR / OCR / 发音模块**。
- 因此**必须在新环境或克隆环境里试验**，绝不要直接在 `memory-tts` 上原地升级：

```powershell
conda create -n memory-tts-perf --clone memory-tts -y
conda activate memory-tts-perf
# 到这里再评估 faster-qwen3-tts；当前环境保持不动
```

回滚：`conda env remove -n memory-tts-perf -y`，`memory-tts` 全程未受影响。

## 4. 模型权重备份

模型不在 env 里，也已被 `.gitignore` 忽略，需单独注意：

```
models\qwen-1.7b\   (model.safetensors 3.83 GB + speech_tokenizer 0.68 GB)
models\qwen-0.6b\   (model.safetensors 1.81 GB + speech_tokenizer 0.68 GB)
```

约 7 GB。若发生误删，可用 `modelscope` 或 HF 重新下载
（`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`、`.../0.6B-CustomVoice`）。
也可以用同一套 robocopy 方式快照到 `D:\DevVmEnv\conda-backups\models\`。

## 5. 代码恢复点

调研开始时的 git 状态：

- 分支 `master`，HEAD `708924d`（"完善短文本音频生成的自校验过程，以及预生成音频机制"），
  与 `origin/master` 一致，无未推送提交，工作区干净。
- 本话轮新增（未改动任何运行时代码）：
  `a439b8b` — `docs(research): Qwen3-TTS 性能调研报告 + 分阶段基准工具`

回滚调研提交（如需）：

```powershell
git revert a439b8b          # 保留历史
# 或
git reset --hard 708924d    # 彻底回到调研前（会丢弃之后的提交）
```

> 注意 `.gitignore` 里 `tests/`、`.github/`、`.vscode/`、`.workbuddy/`、`models/` 均被忽略，
> 这些内容**不在 git 备份范围内**（`tests/` 里有 50 个单测，别以为 git 能救回来）。
