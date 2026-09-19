# Qwen3-TTS — Current State Report (compiled late 2026)

Scope: the `qwen-tts` PyPI package, the `QwenLM/Qwen3-TTS` GitHub repo, the Hugging Face `Qwen/qwen3-tts`
collection, ModelScope, the vLLM-Omni serving stack, and community work — with emphasis on **generation
latency on an RTX 4060 Laptop 8 GB** running `Qwen3-TTS-12Hz-1.7B-CustomVoice` / `12Hz-0.6B-CustomVoice`
with `qwen-tts 0.1.1`, `transformers 4.57.3`, `torch 2.7.0+cu128`.

**Verification rule used throughout:** every claim below is tagged with the URL it came from. Claims I could
not verify are explicitly labelled "could not verify". Nothing here is inferred from memory.

---

## 0. Headline (what actually matters for you)

1. **`qwen-tts 0.1.1` IS still the latest PyPI release** (uploaded 2026-02-06). There is nothing newer to
   upgrade to. The upstream repo's **last commit is 2026-03-17**, and it has **no GitHub releases and no tags**
   — upstream is effectively dormant.
   Sources: <https://pypi.org/pypi/qwen-tts/json>, <https://pypi.org/simple/qwen-tts/>,
   <https://api.github.com/repos/QwenLM/Qwen3-TTS/releases> (`[]`),
   <https://api.github.com/repos/QwenLM/Qwen3-TTS/tags> (`[]`),
   <https://api.github.com/repos/QwenLM/Qwen3-TTS/commits?per_page=10> (newest `022e286`, 2026-03-17).

2. **The root cause is Python/kernel-launch overhead in the autoregressive decode loop — NOT codec FLOPs, and
   (measured on this machine) NOT the waveform decoder at all.** Community PR #191 profiled the codec decoder
   at **9.8 s CPU vs 3.5 s CUDA** per generation (their number is for a **batch of 12** on ROCm, do not
   extrapolate it to single-request local inference) and states the decoder contains "136+ attention modules
   whose Python dispatch overhead dominates waveform decoding time". The vLLM-Omni engineering blog
   independently measured **~14 % Stage-0 and ~6 % Stage-1 GPU utilization at concurrency 64** — "the GPU is
   waiting on Python scheduling, small tensor allocations and kernel-launch overhead rather than compute."
   Sources: <https://github.com/QwenLM/Qwen3-TTS/pull/191>,
   <https://blog.vllm.com.cn/2026/06/23/vllm-omni-tts.html>.

   > **Correction from local measurement (see §7).** On this RTX 4060 Laptop, single-request, I split the two
   > stages and timed them separately. The **codec / waveform decode is 0.7–0.04 s and is 1–4 % of total time**;
   > the **autoregressive code generation (Talker + Code Predictor) is 96–99 %**. So adopting PR #191's
   > `compile_codec` would optimize the part that is already fast and would change almost nothing here. The
   > ~500-kernel-launch diagnosis is correct, but the loop that needs capturing/replaying is the **16-codebook
   > decode step** (1 Talker pass + 15 sequential Code-Predictor passes per 80 ms audio frame), not the vocoder.

3. **There is a ready-made fix that publishes RTX 4060 numbers matching your symptom exactly:**
   `andimarafioti/faster-qwen3-tts` (MIT, also on PyPI as `faster-qwen3-tts`) uses a **static KV cache +
   `torch.cuda.CUDAGraph`**. Its README table for **RTX 4060 (Windows)**:

   | Model | Baseline RTF | Baseline TTFA | CUDA-graph RTF | CUDA-graph TTFA | Speedup |
   |---|---|---|---|---|---|
   | 0.6B | 0.23 | 2 697 ms | **2.26** | **413 ms** | 9.8× / 6.5× |
   | 1.7B | 0.23 | 2 905 ms | **1.83** | **460 ms** | 7.9× / 6.3× |

   Baseline RTF 0.23 means ~4.3 s of compute per 1 s of audio — **your "10–15 s per short sentence" is exactly
   this number**. Source: <https://github.com/andimarafioti/faster-qwen3-tts> (raw README).

4. **No official Flash/Turbo/Realtime/Streaming variant exists, and the 25 Hz models are announced but
   unreleased.** The whole official family is still the six 12 Hz artifacts from 2026-01-22/29.

---

## 1. Version history of the `qwen-tts` PyPI package

### 1.1 Exact released versions and dates

From <https://pypi.org/pypi/qwen-tts/json> and cross-checked against the simple index
<https://pypi.org/simple/qwen-tts/> (wheel upload timestamps, UTC):

| Version | Wheel uploaded (UTC) | sdist uploaded (UTC) | Wheel size |
|---|---|---|---|
| 0.0.2 | 2026-01-22T07:53:27 | 2026-01-22T07:53:30 | 100,680 B |
| 0.0.3 | 2026-01-22T08:35:26 | 2026-01-22T08:35:28 | 100,689 B |
| 0.0.4 | 2026-01-22T15:33:35 | 2026-01-22T15:33:36 | 113,186 B |
| 0.0.5 | 2026-01-23T06:09:54 | 2026-01-23T06:09:56 | 113,320 B |
| 0.1.0 | 2026-02-05T06:31:28 | 2026-02-05T06:31:30 | 113,501 B |
| **0.1.1** | **2026-02-06T04:10:51** | 2026-02-06T04:10:53 | 113,529 B |

**Latest version now: `0.1.1`.** No version has been yanked. There is no 0.1.2, 0.2.x or 1.x.
The PyPI JSON API's `info.version` is `0.1.1` and `release_url` is
<https://pypi.org/project/qwen-tts/0.1.1/>. PyPI reports the project's "Key dates → Released: Feb 6, 2026"
(<https://pypi.org/project/qwen-tts/#history>).

### 1.2 Package metadata for 0.1.1 (i.e. what you have pinned)

From <https://pypi.org/pypi/qwen-tts/json>:

- `requires_dist`: `transformers==4.57.3`, `accelerate==1.12.0`, `gradio`, `librosa`, `torchaudio`,
  `soundfile`, `sox`, `onnxruntime`, `einops`
- `requires_python`: `>=3.9`; classifiers list 3.9–3.13
- `license`: Apache-2.0
- PyPI **owner is the user `bakerbunker`**, not an Alibaba/Qwen org account — worth knowing, but the package
  description is the official Qwen README and `author` is "Alibaba Qwen Team".
- CLI entry point: `qwen-tts-demo` (documented at
  <https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/changelog>).

Note: the hard pin `transformers==4.57.3` is the reason you cannot move to Transformers 5.x without a patch
(see §1.5).

### 1.3 What changed in each release (changelog / release notes / commit log)

**No per-release changelog and no release notes exist.** Verified absences:

- <https://api.github.com/repos/QwenLM/Qwen3-TTS/releases> → `[]`
- <https://api.github.com/repos/QwenLM/Qwen3-TTS/tags> → `[]`
- The Mintlify docs changelog (<https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/changelog>) jumps straight
  from "January 22, 2026 – Qwen3-TTS Initial Release" to a "Version History → qwen-tts Package → v0.1.1 –
  Current Release" entry that only lists dependencies and Python support. **It does not list 0.0.2–0.1.0 at
  all, and describes no behaviour changes for 0.1.1.**

So the only real evidence for what changed is the repo commit log. The complete commit list for the period
covering the 0.1.x releases (from
<https://api.github.com/repos/QwenLM/Qwen3-TTS/commits?per_page=10>) is:

| Commit | Date (UTC) | Message |
|---|---|---|
| `1ab0dd7` | 2026-01-25 | `update: update readme` |
| `680d4e9` | 2026-02-04 | `fix: adjust finetune script for stable training.` |
| `5f8581d` | 2026-02-05 | `fix: padding bugs in 12Hz tokenizer decode.` |
| `6cafe55` | 2026-02-06 | `fix: padding value process bug in tokenizer decode` |
| `022e286` | 2026-03-17 | `fix finetuning bug` |

**Inference (clearly labelled as inference, not a quoted changelog):** the 0.1.0 wheel (2026-02-05T06:31)
lands within a minute of commit `5f8581d` (2026-02-05T06:32), and the 0.1.1 wheel (2026-02-06T04:10) lands
within a minute of `6cafe55` (2026-02-06T04:11). So 0.1.0 ≈ "12 Hz tokenizer decode padding fix" and
0.1.1 ≈ "tokenizer decode padding-value fix". Both are **correctness fixes to the codec decode path**, not
performance fixes. Commit `022e286` (2026-03-17, `fix finetuning bug`) is **in the repo but in no PyPI
release** — the repo is ahead of PyPI.

### 1.4 Is the upstream repo maintained?

Evidence says no meaningful activity since 2026-03:

- Last commit: `022e286` on 2026-03-17T06:38:41Z
  (<https://api.github.com/repos/QwenLM/Qwen3-TTS/commits?per_page=10>).
- Repo metadata (<https://api.github.com/repos/QwenLM/Qwen3-TTS>): `pushed_at: 2026-03-17T06:38:41Z`,
  `stargazers_count: 13450`, `forks_count: 1740`, `open_issues_count: 55`, `archived: false`.
- Meanwhile issues are still being filed in **September 2026** (e.g. #371 on 2026-09-16, #369/#370 on
  2026-09-13, #365 on 2026-09-03) and are essentially unaddressed (issue list via
  <https://api.github.com/repos/QwenLM/Qwen3-TTS/issues?state=all&per_page=100&sort=created&direction=desc>).
- A GitHub Actions bot auto-labels stale issues `inactive` (e.g. comments on issues #171, #347).

### 1.5 Third-party distribution: `qwen-tts-hf`

There **is** a newer, unofficial distribution: **`qwen-tts-hf 0.1.1.post1`**, uploaded **2026-08-25**.
Source: <https://pypi.org/pypi/qwen-tts-hf/json>.

- Self-described as "an unofficial compatibility distribution of Qwen-TTS for Transformers 5. It is built
  from the patch proposed in [QwenLM/Qwen3-TTS#360] and is intended as a temporary dependency until upstream
  publishes equivalent support. It provides the same `qwen_tts` Python package as upstream `qwen-tts`; **do
  not install both distributions in the same environment**."
- `requires_dist`: `transformers<6,>=5.15.1`, `accelerate==1.12.0`, gradio, librosa, torchaudio, soundfile,
  sox, onnxruntime, einops. `requires_python >=3.10`. License expression Apache-2.0.
- Maintainer metadata: `maintainer: "Andres Marafioti"`; PyPI owner role is user `filsino`. Project URL for
  the patch: <https://github.com/andimarafioti/Qwen3-TTS/tree/distribution/qwen-tts-hf>.
- Upstream issue **#360 "Support Transformers 5"** is **open** (filed 2026-08-24 by the issue list at
  <https://api.github.com/repos/QwenLM/Qwen3-TTS/issues?state=all&per_page=100&sort=created&direction=desc>).
  A related report, issue #237, is titled "[Compatibility] qwen-tts 0.1.1 requires transformers==4.57.3 but
  qwen3_tts needs transformers >= 5.x"
  (title captured from <https://api.github.com/search/issues?q=repo%3AQwenLM%2FQwen3-TTS+latency+OR+slow+OR+performance+OR+RTF+OR+%22real-time%22>).

---

## 2. Model variants released since the initial 12Hz-1.7B/0.6B CustomVoice models

### 2.1 The official line-up is unchanged — six artifacts, all 12 Hz

Hugging Face collection API response, `lastUpdated: 2026-01-22T13:00:57Z`, 6 items
(<https://huggingface.co/api/collections/Qwen/qwen3-tts>). Download/like figures are from that response
plus <https://huggingface.co/api/models?author=Qwen&search=TTS&limit=100>:

| HF repo ID | Params (`numParameters`) | Downloads | Likes | Last modified |
|---|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1,916,676,352 | 2,590,596 | 1,977 | 2026-01-29 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1,928,677,440 | 3,821,422 | 522 | 2026-01-23 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1,916,676,352 | 278,139 | 420 | 2026-01-29 |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 905,788,672 | 1,002,045 | 188 | 2026-01-29 |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 914,643,008 | 643,502 | 296 | 2026-01-29 |
| `Qwen/Qwen3-TTS-Tokenizer-12Hz` (the codec) | 170,557,441 | 292,891 | 90 | 2026-01-29 |

Plus one Space: `Qwen/Qwen3-TTS` (gradio, `zero-a10g`, last modified 2026-06-09).

Model card table (from the package README, <https://pypi.org/project/qwen-tts/>): VoiceDesign (1.7B only)
and 1.7B-CustomVoice have **Streaming ✅ and Instruction Control ✅**; the two Base models and
0.6B-CustomVoice have **Streaming ✅ but no Instruction Control**. All five are claimed to support
zh/en/ja/ko/de/fr/ru/pt/es/it.

**The README's own "Released Models" table lists only 12 Hz models.** It adds: "Other models mentioned in
the technical report will be released in the near future."

### 2.2 25 Hz variants: documented, benchmarked — and NOT released

This is a real trap. 25 Hz models appear **in the benchmark tables** of the README and the official docs
(`Qwen3-TTS-25Hz-0.6B-Base`, `Qwen3-TTS-25Hz-1.7B-Base`, `Qwen3-TTS-25Hz-0.6B-CustomVoice`,
`Qwen3-TTS-25Hz-1.7B-CustomVoice`, plus `Qwen-TTS-Tokenizer-25Hz` Stage 1/Stage 2), and the official FAQ
describes them: "**25Hz Tokenizer:** Faster inference, Slightly lower quality. Models:
Qwen3-TTS-25Hz-(0.6B/1.7B)-(CustomVoice/Base)"
(<https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/faq>).

But **no 25 Hz checkpoint exists on HF or ModelScope.** Verified three ways:

- <https://huggingface.co/api/models?search=Qwen3-TTS-25Hz&limit=50> → `[]`
- <https://huggingface.co/api/models/Qwen/Qwen3-TTS-25Hz-1.7B-Base> → HTTP 401 `Invalid username or password`
  (i.e. the repo does not exist publicly)
- <https://modelscope.cn/api/v1/models/Qwen/Qwen3-TTS-25Hz-1.7B-Base> → `{"Code":10010205001,"Message":
  "获取模型信息失败，信息：record not found"}`

Issue **#34 "Release Qwen3-TTS 25Hz models and tokenizer on Hugging Face"** (filed by Hugging Face's
NielsRogge, 2026-01-23) was **closed 2026-01-26** with `state_reason: "completed"` and 8 reactions, but the
weights never appeared (12 comments; <https://api.github.com/repos/QwenLM/Qwen3-TTS/issues/34>).
A later issue (#369, 2026-09-13) also states plainly: "**The 25 Hz weights are not released yet (#34)**"
(<https://api.github.com/repos/QwenLM/Qwen3-TTS/issues/369>).

The repo *does* contain 25 Hz tokenizer **code** (`qwen_tts/core/tokenizer_25hz/...`, referenced in issues
#369/#370/#356), but no weights.

Since 25 Hz is documented as "faster inference", this is the single biggest missing official latency lever.

### 2.3 Flash / Turbo / Streaming / Realtime / InstructVoice / VoiceDesign / Base / Finetune variants

- **VoiceDesign exists** — `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (1.7B only), released 2026-01-21.
- **Base (finetune-able) exists** — `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base`. Fine-tuning instructions are in
  the repo's `finetuning/` directory per the README.
- **No** `Flash`, `Turbo`, `Realtime`, `Streaming`, `InstructVoice`, or `VoiceEditing` repo exists under the
  `Qwen` org. Evidence: exhaustive HF API queries
  <https://huggingface.co/api/models?author=Qwen&search=TTS&limit=100> (returns exactly the 6 items above),
  <https://huggingface.co/api/models?search=Qwen3-TTS&limit=100&sort=downloads&direction=-1> (returns the 6
  official repos plus community quantizations/finetunes — no official new variant), and
  <https://huggingface.co/api/models?search=Qwen3-TTS-25Hz&limit=50> → `[]`.
  Note "Realtime" exists only as the **DashScope cloud API**, not as local weights:
  <https://help.aliyun.com/zh/model-studio/qwen-tts-realtime>.
- **No official quantized checkpoints** (FP8/INT8/AWQ/GPTQ), **no official ONNX/TensorRT exports**. The
  official FAQ says under "Can I run Qwen3-TTS on CPU?": "Consider using quantized models (**coming soon**)"
  (<https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/faq>).

### 2.4 Community / third-party checkpoints (verified HF repo IDs)

**GGUF (llama.cpp / ggml / crispasr ecosystem)** — most popular first:

| Repo ID | Notes | Source |
|---|---|---|
| `Serveurperso/Qwen3-TTS-GGUF` | 364,062 downloads, 45 likes, apache-2.0, created 2026-05-11; converts all six base models; includes tokenizer GGUF | <https://huggingface.co/api/models/Serveurperso/Qwen3-TTS-GGUF?blobs=true> |
| `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF` | 27,886 dl, created 2026-08-03 | <https://huggingface.co/api/models?search=Qwen3-TTS&limit=100&sort=downloads&direction=-1> |
| `cstr/qwen3-tts-1.7b-base-GGUF`, `…-1.7b-customvoice-GGUF`, `…-0.6b-base-GGUF`, `…-0.6b-customvoice-GGUF`, `…-1.7b-voicedesign-GGUF`, `…-tokenizer-12hz-GGUF`, `cstr/qwen3-tts-voices-GGUF` | `library_name: ggml`, `crispasr` tag; created 2026-04-29 … 2026-06-17 | same HF query |
| `khimaros/Qwen3-TTS-12Hz-*-GGUF`, `khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF` | created 2026-04-16 | same |
| `mradermacher/Qwen3-TTS-12Hz-1.7B-Base-GGUF`, `-i1-GGUF`, `…-0.6B-Base-GGUF`, `-i1-GGUF` | created 2026-08-08/09 | same |
| `Jahaz/Qwen3-tts-0.6b-gguf-for-koboldcpp`, `BricksDisplay/Qwen3-TTS-12Hz-0.6B-GGUF`, `Mouserat/qwen3-tts-0.6b-base-gguf`, `hans00/…`, `lookbe/…`, `CC-TM/…`, `rabindra1x/…`, `digibuzz24/…`, `FindaDeath/…`, `badlogicgames/qwen3-tts-0.6b-q8_0-gguf`, `thewh1teagle/qwen3-tts-gguf` | assorted | same |

**Exact GGUF file sizes** from `Serveurperso/Qwen3-TTS-GGUF` `<blobs=true>` (useful for 8 GB planning):

| File | Bytes | ≈ |
|---|---|---|
| `qwen-talker-0.6b-customvoice-Q4_K_M.gguf` | 604,878,080 | 0.56 GiB |
| `qwen-talker-0.6b-customvoice-Q8_0.gguf` | 968,588,544 | 0.90 GiB |
| `qwen-talker-0.6b-customvoice-BF16.gguf` | 1,817,689,344 | 1.69 GiB |
| `qwen-talker-1.7b-customvoice-Q4_K_M.gguf` | 1,182,631,296 | 1.10 GiB |
| `qwen-talker-1.7b-customvoice-Q8_0.gguf` | 2,042,834,304 | 1.90 GiB |
| `qwen-talker-1.7b-customvoice-BF16.gguf` | 3,839,585,664 | 3.58 GiB |
| `qwen-talker-1.7b-customvoice-F32.gguf` | 7,672,655,136 | 7.15 GiB |
| `qwen-tokenizer-12hz-Q4_K_M.gguf` | 254,974,752 | 0.24 GiB |
| `qwen-tokenizer-12hz-Q8_0.gguf` | 291,150,624 | 0.27 GiB |
| `qwen-tokenizer-12hz-BF16.gguf` | 358,980,384 | 0.33 GiB |

(Total repo storage: 60,071,360,480 B.)

**MLX / Apple Silicon** (created 2026-01-22, `library_name: mlx-audio`): `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-{bf16,8bit,6bit,4bit}`,
`…-1.7B-CustomVoice-{bf16,8bit,6bit,4bit}`, `…-1.7B-VoiceDesign-{bf16,8bit,6bit,4bit}`,
`…-0.6B-Base-{bf16,8bit,6bit,4bit}`, `…-0.6B-CustomVoice-{bf16,8bit,6bit,4bit}`; plus
`theoracleguy/*`, `aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit`, `PowerBeef02/*` (tagged `vocello`).
Source: <https://huggingface.co/api/models?search=Qwen3-TTS&limit=100&sort=downloads&direction=-1>.

**ONNX / mobile / other accelerators:**

- `speed-brain-ai/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX` — `library_name: onnxruntime`, tags include
  **`streaming`**, created **2026-09-14** (226 dl). <https://huggingface.co/api/models?search=Qwen3-TTS&limit=100&sort=downloads&direction=-1>
- `kautism/qwen3-tts-rk3588` — ONNX + GGUF for Rockchip RK3588, created 2026-02-27.
- `Arm/qwen3-tts-0-6b-custom-voice-mix-precision` and `Arm/qwen3-tts-0-6b-base-mix-precision` — tags
  include `litert`, `tflite`, `onnx`, `gguf`, `quantized`, **`streaming`**; created 2026-08-18.
- `litert-community/Qwen3-TTS-12Hz-0.6B-Base` — LiteRT/TFLite, `on-device`, created 2026-07-07.
- `UrocyonF/Qwen3-TTS-12Hz-1.7B-NVFP4` — NVFP4 / `compressed-tensors`, `library_name: transformers`,
  base `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, created 2026-07-22.
- `owlninjam/qwen3-tts-1.7b-customvoice-fp16` (2026-03-14), `mysanit/qwen3-tts-gguf` (onnx+gguf),
  `Solidaxel/Qwen3-TTS-GGUF` (gguf+onnx), `cgisky/qwen3-tts-custom-gguf`,
  `caffeinism/mimi-qwen3-tts-encoder`.
- `takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz` — a 48 kHz **finetune of the codec**, created 2026-03-04.

**Not verified in this pass:** none of these community quantizations were checked for *latency* claims; I only
verified that the repos and their tags/sizes exist. The only quantization source with published RTF numbers I
found is the pure-C engine (§5), which quantizes itself rather than consuming the GGUFs above.

---

## 3. Official and community fixes for generation latency

### 3.1 Upstream repo: no latency fix has been merged

- Current open PRs include **#191** (below), and nothing perf-related is merged. The commit log for
  2026-02-01 → present contains only tokenizer-decode padding fixes and finetune fixes (§1.3).
- The GitHub Actions bot labelling old issues `inactive` (e.g. #171 on 2026-04-10) shows the tracker is not
  actively triaged.

### 3.2 PR #191 — `torch.compile` on the codec decoder (OPEN, unmerged) — the most directly relevant item

<https://github.com/QwenLM/Qwen3-TTS/pull/191> — "[Perf] Add `compile_codec` option for 3-4x faster batch
decoding", opened **2026-02-07** by `Finrandojin`, `state: "open"`, `merged_at: null`, 1 comment.

The PR body's profiling table (author's measurement, AMD RX 7900 XTX / ROCm 6.3):

| Component | CPU time | CUDA time |
|---|---|---|
| Codec decoder (single generation) | 9.8 s | 3.5 s |
| Share of total batch time | — | ~85 % |

and its claim: "The codec decoder contains **136+ attention modules whose Python dispatch overhead dominates
waveform decoding time**." After `torch.compile(mode="max-autotune", dynamic=True)` on the codec decoder:

| Metric | Before | After |
|---|---|---|
| Single generation | ~14 s | ~9 s |
| Batch = 12 throughput | 1.3× real-time | **4.3× real-time** |

Trade-off explicitly documented: "First call incurs ~30–60 s warmup". API added:
`Qwen3TTSModel.from_pretrained(..., compile_codec=True)` or `model._compile_codec()`.

Comment (2026-03-08, `Iamgoofball`): "Set this up locally on top of that Faster Qwen3-TTS fork that's using
CUDAGraphs and I'm getting massive speedups with no change in generation quality. PR works 👍"
(<https://api.github.com/repos/QwenLM/Qwen3-TTS/issues/191/comments>).

**This PR is a drop-in, ~40-line-ish change you can apply to your pinned 0.1.1 tree yourself.**

### 3.3 Issue #89 — the main community latency thread (and the fix everyone actually uses)

Issue #89 is where the community converged. The key comment is
<https://github.com/QwenLM/Qwen3-TTS/issues/89#issuecomment-3799395212> by `dffdeeq` (2026-01-26,
edited 2026-02-16; **36 reactions**):

> "i got ~6x speedup on regular inference and about the same with streaming which I also implemented
> (**ttfb ~0.08s on a 5090, RTF < 0.25**) by using **torch.compile + cuda graphs for the decoder,
> torch.compile for CodePredictor and Talker, fast codebook and `torch.set_float32_matmul_precision('high')`**"

A later comment from the same author lists the mechanism more completely:
"torch.compile + cuda graphs for the decoder, torch.compile for CodePredictor, also **tweaking the eos check
to avoid cpu sync**, and `torch.set_float32_matmul_precision('high')`", with the optimization list printed by
his benchmark script:
"1. `torch.set_float32_matmul_precision('high')` – TensorFloat32 on Ampere+ GPUs; 2. bfloat16 dtype;
3. flash_attention_2; 4. **`use_fast_codebook=True` – bypasses HuggingFace `generate()` for 2-3x speedup**;
5. torch.compile with max-autotune mode for decoder; 6. `compile_codebook_predictor=True` – compiled code
predictor" (<https://api.github.com/repos/QwenLM/Qwen3-TTS/issues/89/comments>, comments 3799395212 and
3800982533).

Reported before/after numbers from that thread (all user-reported, hardware as stated):

| Reporter / hardware | Configuration | Baseline RTF | Optimized RTF | Speedup |
|---|---|---|---|---|
| `dur-randir`, CUDA 13.0 + torch 2.10 + FA 2.8.3 (Windows) | 1.7B | 9.40 | 2.91 | 2.67–3.20× |
| same box, **after switching Windows → Linux** (torch 2.9.1+cu130) | 1.7B | 3.28 | **1.02** | 3.94× |
| `mame82`, **RTX 4070 mobile**, 0.6B base | — | 2.82 | **0.86–0.98** | 2.46–2.71× |
| `RyrieNorth`, RTX 5090, CUDA 13.1, torch 2.10.0+cu130, FA 2.8.3 | 1.7B | 1.57 | **0.50–0.51** | 3.00× |
| `RyrieNorth`, same 5090, streaming | 1.7B | 1.36 | **0.55** | 2.58× |
| `nfranke` (anonymous GPU) | 1.7B | 0.78–0.82 | 0.35–0.36 | ~2.0× |

Two important incidental findings from the same thread:

- **Windows → Linux gave a 3× speedup on identical hardware** (`dur-randir`, comment 3810737385): RTF 2.91
  → 1.02. He attributes it to CPU contention. This is directly relevant to a Windows laptop.
- **GPU utilization is the tell.** Multiple users report `nvidia-smi` showing **10–16 % GPU utilization**
  (comments 3801756963, and the HF discussion in §3.7). "From my usage, the GPU utilization rate is very low,
  with a maximum of 16%."
- A user on 2080 Ti reported "生成3秒音频需要15秒左右" (≈15 s for 3 s of audio) **while "Index TTS2 / GPT-SoVits
  都很快（2秒左右）"** (comment 3811768740) — a direct community comparison to alternative engines.
- A user noted the architecture criticism: "even with all optimizations, RTF of 0.5 is ridiculous for what is
  fundamentally a 1.7B LM with a small MTP module on top… I get this speed on CPU alone with Qwen3-1.7B"
  (comment 3800008287).

### 3.4 vLLM-Omni — the official serving path, with real merged perf work

The Qwen README states: "vLLM officially provides day-0 support for Qwen3-TTS… Now only offline inference is
supported. Online serving will be supported later, and **vLLM-Omni will continue to offer support and
optimization for Qwen3-TTS in areas such as inference speed and streaming capabilities**"
(<https://pypi.org/project/qwen-tts/>).

**Merged: PR #3485 "[Perf] Fix Qwen3-TTS latency regression"**, merged **2026-05-10**
(<https://api.github.com/repos/vllm-project/vllm-omni/pulls/3485>). Mechanism:

- Decouple connector streaming chunking (`codec_chunk_frames` / `codec_left_context_frames`) from Code2Wav's
  internal decode window (`decode_chunk_frames=300`, `decode_left_context_frames=25`).
- Add `initial_codec_chunk_frames` so only the **first** emitted codec chunk is small (lowers first-audio
  latency without making Code2Wav repeatedly decode tiny overlapping windows).
- Align Stage-1 `max_num_seqs` to 10 and "document avoiding `max_num_seqs: 1` for latency-sensitive serving".
- **Wire custom Code2Wav decode-chunk overrides into CUDA-Graph warmup buckets** so configs stop silently
  missing graph capture and falling back to eager.
- Clean `_cached_ic` in `OmniConnectorModelRunnerMixin` request cleanup.

Benchmark from the PR description (H20, Qwen3-TTS CustomVoice "Vivian", streaming PCM, c=8, 20 measured
requests, first 2 excluded as warmup):

| Case | TTFT/TTFA median | E2EL median | RTF median |
|---|---:|---:|---:|
| v0.20 after #3203 regression point | 878.5 ms | 2017.4 ms | 0.389 |
| This PR, run 1 | 130.6 ms | 1393.8 ms | 0.2565 |
| This PR, run 2 | 145.3 ms | 1270.7 ms | **0.2591** |

**Issue/RFC #3163 "[RFC]: Cross-request batching for Qwen3-TTS Code2Wav stage to fix TTFB scaling under
concurrency"**, created 2026-04-26, **closed 2026-05-29** as `completed`
(<https://api.github.com/repos/vllm-project/vllm-omni/issues/3163>). This is the most detailed public
analysis of the Code2Wav bottleneck I found:

- Measured on a **single RTX 4090**, `qwen3-tts-1.7b-base`: TTFB **99 ms at QPS=1 → 275 ms at QPS=2 → 515 ms
  at QPS=3 → 1160 ms at QPS=4** (11.7× scaling).
- Root cause: PR #1617 introduced a `CUDAGraphDecoderWrapper` captured **at bs=1 only**, which "enforced bs=1
  in `forward()` via a per-request for-loop, undoing #1426['s batched decoding]".
- Quantitative saturation statement: "at QPS=4 on a single RTX 4090, the Code2Wav stage is approaching GPU
  saturation (**input chunk rate ~120 chunks/s vs single-request capacity ~33 chunks/s**). Batching expands
  per-step throughput by ~3-4×."
- Rejected alternative, with reason: engine-level CUDA Graph for Stage 1 is "infeasible per the existing
  `qwen3_tts.yaml` comment ('its codec generation loop is not cudagraph-compatible')" — hence the
  decoder-internal graph wrapper.

### 3.5 vLLM-Omni engineering blog — the clearest official statement of *where the time goes*

"Engineering TTS Inference in vLLM-Omni", 2026-06-23, Chinese:
<https://blog.vllm.com.cn/2026/06/23/vllm-omni-tts.html>. Key Qwen3-TTS statements:

- Architecture: "Talker → connector → Code2Wav"; Code2Wav is a light non-DiT decoder with no iterative
  denoising loop.
- At **c=64 on H20×2**, measured average GPU utilization was **~14 % for Stage 0 and ~6 % for Stage 1**: the
  GPU "waits for Python scheduling, small tensor allocations and kernel launch overhead, not compute."
  It contrasts this with VoxCPM2 (whose `torch.compile` of the whole forward cut RTF ~0.21 → ~0.13, the single
  biggest win there).
- Per-request speaker-embedding preparation originally computed **mel/STFT on CPU per request** and copied it
  to GPU; they cached mel basis/window buffers on GPU and batched the mel/STFT on GPU, eliminating repeated
  CPU work and H2D transfers.
- `trailing_text` sliding window originally used tensor slicing+concat every decode step; replaced with an
  offset + compaction only past `_TRAILING_TEXT_COMPACT_MIN_FRAMES = 64`.
- `req_id_to_index` was an O(N²) `list.index()` scan per decode step; replaced with a dict.
- Code2Wav CUDA-graph capture is keyed by `(batch_size, frames)` with `bisect_left` bucket selection; measured
  **hit rate started at 88 % and stabilised ~81 %** over five c=16 rounds, with fallbacks mostly at batch>1
  shapes like `(2, 98, 169)` and `(8, 73, 73)`, and `stream_capture_fallbacks=0`.
- Precision note: the Talker **code predictor** is precision-sensitive (very short sequences, repeated
  prefill); they keep RMSNorm variance, RoPE cos/sin, attention and QKV projections in **fp32** to match the
  reference bit-wise, because bf16 fused kernels drift and hurt audio after dozens of steps.

Results (Qwen3-TTS, c=64, p=512, H20×2, voice cloning):

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Audio throughput | 26.55 audio-s/s | 42.88 audio-s/s | +61.5 % |
| Median E2EL | 9654 ms | 5699 ms | −41.0 % |
| P99 E2EL | 17686 ms | 8956 ms | −49.4 % |
| P99 TTFP | 7558 ms | 5563 ms | −26.4 % |

Concurrency sweep (H20×2, voice clone, streaming): TTFP 70.61 ms at c=1, 268.75 ms at c=8, 451.32 ms at c=16,
637.43 ms at c=32, **1127.93 ms at c=64**.

### 3.6 `sdpa` vs `flash-attn`, dtype

- **flash-attn is not the lever.** In issue #89, `dffdeeq` analysed `nfranke`'s numbers and concluded: "in the
  compiled path, flash-attn vs eager makes no real difference" (comment 3811026425). Separately, in HF
  discussion #18 an RTX 4090 user says: "Same on my RTX 4090. Infact I am getting same (or somewhat better)
  speed with `attn_implementation = "eager"`"
  (<https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/discussions/18>).
- Official docs advice is nonetheless FA2 + bf16 + 0.6B + batching
  (<https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/faq>, "How can I speed up inference?").
- **fp16 caution:** open PR **#355** "fix(modeling): cast logits to float32 to prevent FP16 sampling overflow"
  (2026-08-07) exists in the repo's issue/PR list. The official FAQ's OOM advice to "use float16 instead of
  bfloat16" therefore carries a known risk of sampling overflow.
- **fp32 caution on the 25 Hz tokenizer path:** issue **#369** reports that `qkv_attention_manual` in
  `qwen_tts/core/tokenizer_25hz/vq/whisper_encoder.py` does `masked_fill` on a **bool** mask, which casts
  `-finfo.max` to `True`, so padded keys are never masked. It is reached "whenever `flash_attn` is not
  importable (any CPU/macOS install), **and also on GPU whenever `q.dtype` is not fp16/bf16`** — i.e. loading
  without `dtype=` gives fp32 under the pinned transformers 4.57.3, and "the README tokenizer example passes
  no dtype, so a default GPU install hits it as well". Scope note in the issue: "only the 25 Hz tokenizer
  encoder uses this class. The 12 Hz tokenizer and the TTS model do not import it, so the published 12 Hz
  checkpoints are unaffected." Fix PR #370 is open. This matters for correctness, not speed, and only if you
  touch the 25 Hz path.

### 3.7 Batching and concurrency

- **Batching works and helps throughput, not single-request latency.** PR #191: batch=12 went 1.3× → 4.3×
  real-time. The pure-C engine reports ~3.35× per-step at B=8 and ~3× end-to-end (§5).
- **Do not call one model instance from multiple Python threads.** Issue **#350 "Multi-concurrency inference
  not supported"** (open, 2026-07-24, labelled `inactive`): "When I load a single `Qwen3TTSModel` instance and
  call `generate_custom_voice()` from multiple Python threads at the same time, inference becomes extremely
  slow — much slower than sequential calls, and much slower than passing a list of texts for batch inference.
  Increasing the thread count does not improve throughput; total latency just keeps growing."
  Recommended pattern in the issue text: queue + single worker, dynamic batching, multiple instances, or
  vLLM-Omni. <https://api.github.com/repos/QwenLM/Qwen3-TTS/issues/350>
- The official FAQ recommends batching by passing a list of texts to one `generate_*` call.

### 3.8 KV cache / static cache / CUDA graphs / torch.compile / codec-on-CPU

Summary of what actually exists:

| Technique | Status | Evidence |
|---|---|---|
| `torch.compile` on codec decoder | PR #191 **open, unmerged**; confirmed working by a second user | <https://github.com/QwenLM/Qwen3-TTS/pull/191> |
| `torch.compile` on Talker + CodePredictor | Implemented in community fork `dffdeeq/Qwen3-TTS-streaming` | issue #89 comment 3799395212 |
| CUDA graphs (decoder + predictors) | Community forks; **vLLM-Omni has a `CUDAGraphDecoderWrapper` captured at bs=1 only**, being extended to bs∈{2,4,8} | issue #89; vLLM-Omni #3163; vLLM blog |
| Static KV cache | Implemented in `faster-qwen3-tts` (static KV + padded attention mask) | <https://github.com/andimarafioti/faster-qwen3-tts> |
| `torch.set_float32_matmul_precision('high')` | Community; one line | issue #89 comment 3800982533 |
| Avoiding CPU sync in the EOS check | Community | issue #89 comment 3800982533 |
| `use_fast_codebook=True` (bypass HF `generate()`) | Community; claimed 2–3× | issue #89 comment 3800982533 |
| Chunked / streaming vocoder decode | Exists in upstream code as `chunked_decode` with 25-frame left context; vLLM-Omni exposes `decode_chunk_frames=300` / `decode_left_context_frames=25`; two-phase first-chunk trick in `kunzite-app/Qwen3-TTS-streaming` | vLLM blog; <https://github.com/kunzite-app/Qwen3-TTS-streaming> |
| Codec running on CPU unnecessarily | **Confirmed pattern in the upstream hot path**: vLLM-Omni found per-request mel/STFT computed on CPU and copied to GPU, and fixed it by GPU-resident batched mel/STFT | vLLM blog |
| `sdpa` vs `flash-attn` | No material difference once compiled | issue #89 comment 3811026425; HF discussion #18 |
| Batching | Effective for throughput (3–4×), not single-request latency | PR #191; vLLM-Omni #3163 |
| KV-cache reuse across requests (prefix caching) | **Could not verify any implementation** | — |
| Official ONNX / TensorRT export | **Does not exist** | §2.3 |

---

## 4. Official benchmark numbers for RTF / latency, and streaming-latency claims

### 4.1 What Qwen officially publishes

The official evaluation section publishes **only quality metrics** — WER, SIM, APS/DSD/RP — no RTF and no
latency table. From <https://pypi.org/project/qwen-tts/> and
<https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/benchmarks>:

- Evaluation config: `dtype=torch.bfloat16`, `max_new_tokens=2048`, per-checkpoint `generate_config.json`
  defaults.
- Seed-TTS test set WER: `Qwen3-TTS-12Hz-1.7B-Base` zh **0.77**, en **1.24** (stated as "best-in-class
  English"); `12Hz-0.6B-Base` zh 0.92, en 1.32; `25Hz-1.7B-Base` zh 1.10, en 1.49; `25Hz-0.6B-Base` zh 1.18,
  en 1.64. CosyVoice 3 is zh **0.71** / en 1.45.
- Tokenizer table: `Qwen-TTS-Tokenizer-12Hz` is 16 codebooks, size 2048, **12.5 FPS**, PESQ_WB 3.21,
  PESQ_NB 3.68, STOI 0.96, UTMOS 4.16, SIM 0.95.

### 4.2 The only official *latency* claims are marketing for the cloud API

- "**Extreme Low-Latency Streaming Generation** … It can output the first audio packet immediately after a
  single character is input, with **end-to-end synthesis latency as low as 97ms**, meeting the rigorous
  demands of real-time interactive scenarios."
  Sources: <https://pypi.org/project/qwen-tts/>,
  <https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/changelog>.
- The FAQ repeats the 97 ms figure and clarifies the scope: streaming for **local** Python use is incomplete —
  "For streaming API usage, see the DashScope API documentation. **Python package streaming support is coming
  soon.**" (<https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/faq>)
  **So the 97 ms number is a DashScope cloud-API figure, not a local-inference figure, and no hardware or
  batch size is given.** Treat it as unverifiable for your hardware.
- Also from the FAQ: "**How long does voice cloning take?** First generation with new voice: 2–4 seconds
  (includes prompt extraction). Subsequent generations with cached prompt: **< 1 second**." (No GPU stated.)

### 4.3 Benchmark numbers from the vLLM-Omni stack (official-adjacent, hardware stated)

Already listed in §3.4/§3.5. Consolidated:

| Source | Hardware | Config | Result |
|---|---|---|---|
| vLLM-Omni PR #3485 (merged 2026-05-10) | H20 | c=8 streaming, Vivian | RTF **0.2565–0.2591**, TTFA 130.6–145.3 ms, E2EL 1270.7–1393.8 ms (vs pre-fix RTF 0.389, TTFA 878.5 ms) |
| vLLM-Omni #3163 | 1× RTX 4090 | QPS sweep | TTFB 99 ms (QPS 1) → 1160 ms (QPS 4) |
| vLLM-Omni #3163 (PR #1617 data) | RTX 5090 | c=1/4/10 | TTFP 365 ms → 3211 ms → 8655 ms |
| vLLM-Omni blog | H20×2 | c=1…64 | TTFP 70.61 → 1127.93 ms; throughput 42.88 audio-s/s |
| vLLM-Omni blog (perf doc, cited) | H200 | c=1/4/10 | TTFP 64 → 119 → 425 ms |

### 4.4 Community RTF numbers on consumer GPUs (user-reported)

See the table in §3.3. The most relevant ones for you:

- **RTX 4070 mobile, 0.6B base, optimized: RTF 0.86–0.98** (real-time) — issue #89 comment 3812963804.
- **RTX 4060 (Windows), 0.6B: baseline RTF 0.23 → 2.26 with CUDA graphs; 1.7B: 0.23 → 1.83** —
  <https://github.com/andimarafioti/faster-qwen3-tts>.
- **RTX 4060-class (~270 GB/s), 1.7B, resident fused CUDA, `--quant-mixed`: RTF 0.44** —
  <https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/docs/cuda-performance.md>.
- **RTX 3090: RTF ≈ 3.0** unoptimized; **RTX 3080 mobile: RTF 4.0**; **RTX 5090: RTF 1.57 → 0.50 optimized**
  — HF discussion #18 and issue #89.

---

## 5. Community solutions

### 5.1 Qwen3-TTS-specific speed forks/projects (verified repo metadata)

| Project | Stars | License | What it does | Published speed claims |
|---|---|---|---|---|
| [`dffdeeq/Qwen3-TTS-streaming`](https://github.com/dffdeeq/Qwen3-TTS-streaming) | 269 | Apache-2.0 | Fork "with streaming inference support + **~6× faster inference**". `torch.compile` + CUDA graphs on the decoder, `torch.compile` on CodePredictor and Talker, fast codebook, TF32, EOS sync removal, crossfade overlap | "~6x speedup", "ttfb ~0.08s on a 5090, RTF < 0.25" (author); user-measured 0.50–1.02 RTF on 5090/4070-mobile. Repo API: <https://api.github.com/repos/dffdeeq/Qwen3-TTS-streaming> |
| [`kunzite-app/Qwen3-TTS-streaming`](https://github.com/kunzite-app/Qwen3-TTS-streaming) | — | — | Fork of the above adding **two-phase streaming** (phase 1: `emit_every=5`, `decode_window=48`, optimized OFF; phase 2: `emit_every=12`, `decode_window=80`, optimized ON) | 1st chunk **570 ms → 208 ms** (2.75×); total 3.16 s → 2.58 s; RTF 0.56 → 0.39. Also `dsugisawa-mixi/Qwen3-TTS-streaming` mirrors it |
| [`andimarafioti/faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) | — | MIT | "Real-time Qwen3-TTS inference using CUDA graph capture. No Flash Attention, no vLLM, no Triton. Just `torch.cuda.CUDAGraph`." Static KV cache + padded attention; both streaming and non-streaming. PyPI `faster-qwen3-tts`. Needs torch ≥ 2.5.1 (Blackwell/RTX 50xx needs cu128 wheels). Extra GGML backend via `qwentts-cpp-python>=0.3.1` | **RTX 4060 (Windows)** 0.6B 0.23→**2.26** RTF, 2697→**413 ms** TTFA; 1.7B 0.23→**1.83** RTF, 2905→**460 ms** TTFA. RTX 4090 0.6B 0.82→**4.78**; H100 0.6B 0.435→**3.884**. Per-step on Jetson (0.6B): Talker 75→12 ms, Predictor 190→26 ms, overhead 65→16 ms, **total 330→54 ms** |
| [`tsdocode/nano-qwen3tts-vllm`](https://github.com/tsdocode/nano-qwen3tts-vllm) | 138 | *(no license field)* | "Qwen3-TTS with nano vLLM-style optimizations… **Achieved 3x faster**"; credited by `faster-qwen3-tts` for CUDA-graph inspiration | 3× (README description) |
| [`gabriele-mastrapasqua/qwen3-tts`](https://github.com/gabriele-mastrapasqua/qwen3-tts) | 100 | MIT | **Pure C inference engine** (no Python/PyTorch/ONNX — C + BLAS), mmap'd BF16 safetensors, INT8/INT4 quantization, optional **CUDA** and **Metal** resident-fused backends, HTTP server, `--serve --batch-size N` | CUDA 1.7B `--quant-mixed` **RTF 0.44 on RTX 4060-class**; A100 0.6B 0.39 / 1.7B 0.50. CPU M1: 0.6B `--int8` **0.69**, `--int4` **0.52**. Batching ~3.35× per-step at B=8, ~3× e2e. See §5.3 |
| [`bon5co/qwen3-tts`](https://github.com/bon5co/qwen3-tts) | 0 (fork of the above) | MIT | Fork carrying the detailed `docs/cuda-performance.md` | same numbers |
| `cgisky1980/qwen3-tts-rust` | — | — | Rust port, recommended in issue #171 by its author ("试试 https://github.com/cgisky1980/qwen3-tts-rust") | no numbers found |
| [`predict-woo/qwen3-tts.cpp`](https://github.com/predict-woo/qwen3-tts.cpp), `Pascal`/`qwentts.cpp` | — | — | C++ runtime; `faster-qwen3-tts` has an experimental GGML adapter for "Pascal's `qwentts.cpp` runtime" | — |
| [`jamiepine/voicebox`](https://github.com/jamiepine/voicebox) | **55,116** | MIT | "The open-source AI voice studio. Clone, dictate, create." Topics include `qwen3-tts`, `mlx`, `cuda`. Not a speed patch itself, but by far the largest consumer of Qwen3-TTS | — |
| ComfyUI nodes: [`flybirdxx/ComfyUI-Qwen-TTS`](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) (1,909★), [`DarioFT/ComfyUI-Qwen3-TTS`](https://github.com/DarioFT/ComfyUI-Qwen3-TTS) (302★), [`1038lab/ComfyUI-QwenTTS`](https://github.com/1038lab/ComfyUI-QwenTTS) (254★) | — | GPL-3.0 for 1038lab | UI wrappers | — |
| [`Xerophayze/TTS-Story`](https://github.com/Xerophayze/TTS-Story) | 336 | other | Audiobook tool with Qwen3-TTS + Kokoro + Chatterbox + VOX CPM + IndexTTS-2 backends | — |
| `LostRuins/koboldcpp` issue #2173 "Faster Qwen3-TTS iq4xs" | — | — | GGUF/iq4_xs quantization in koboldcpp | <https://github.com/LostRuins/koboldcpp/issues/2173> |
| `andimarafioti/Qwen3-TTS` branch `distribution/qwen-tts-hf` | — | — | The Transformers-5 compat patch published as `qwen-tts-hf` | <https://github.com/andimarafioti/Qwen3-TTS/tree/distribution/qwen-tts-hf> |

Forum/community discussions found (titles verified, contents not all read):

- HF discussion **#18** on `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`: "**Low generation speed and low GPU
  utilization (~12%) during inference**" — <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/discussions/18>
  (58 community threads on that repo). RTF ~3.0 on RTX 3090, 14–16 % GPU util on RTX 5090, RTF 4.0 on 3080
  mobile, plus a user noting sentence-level outputs "are significantly different from each other even with the
  same configuration".
- Issue #89 (§3.3) is the de-facto community performance thread.
- Chinese-language: the vLLM blog post (2026-06-23) is the highest-quality Chinese source I verified. A
  Bilibili-linked GPU-cloud image listing for "Faster-Qwen3-TTS" surfaced in search
  (<https://www.compshare.cn/images/69wfjOHr63Mw>) but I did not verify its content.
- The `bon5co/qwen3-tts` `docs/cuda-performance.md` doc is an English performance write-up with explicit
  RTX 4060-class numbers (§5.3).

**Could not verify:** I did not find a verifiable Zhihu / CSDN / Juejin / Reddit / HN thread with concrete
Qwen3-TTS latency measurements. `web_search` returned source lists rather than summaries, and the Chinese
queries I ran (`Qwen3-TTS 推理速度 慢 加速 显存 4060`, `Qwen3-TTS 8GB VRAM RTX 4060 laptop`) did not surface
readable pages I could confirm. Treat "there are Chinese community write-ups" as unproven here.

### 5.2 Details of `faster-qwen3-tts` (most relevant to your hardware)

From the raw README (<https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main/README.md>):

- Mechanism: "Qwen3-TTS runs two autoregressive transformers per decode step: **Talker (28 layers)** …
  **Code Predictor (5 layers)**, generating 15 additional codebook tokens. A single step involves **~500 small
  CUDA kernel launches** with Python overhead between them. The GPU spends more time waiting for the next
  kernel than computing." Fix = static KV cache + `torch.cuda.CUDAGraph` capture of both graphs.
- Streaming chunk-size trade-off (Jetson AGX Orin, 0.6B): chunk_size 1 → TTFA 240 ms / RTF 0.750; 2 → 266 ms /
  1.042; 4 → 362 ms / 1.251; 8 → 556 ms / 1.384; 12 → 753 ms / 1.449; non-streaming → RTF 1.57.
- Voice-clone mode speed is identical across modes on 0.6B at chunk_size=8: xvec TTFA 152 ms / RTF 5.470,
  full ICL 149 ms / 5.497, CustomVoice 148 ms / 5.537.
- `non_streaming_mode` has **no measurable perf impact**: on RTX 4090, 1.7B, ICL, chunk_size=8, TTFA ≈159 ms
  either way and RTF 4.87 (nsm=False) vs 4.85 (nsm=True).
- It implements the codec left-context correctly: "the model wrapper decodes each chunk to audio using a
  sliding window with **25-frame left context** (matching the upstream codec's `chunked_decode` pattern) to
  avoid boundary artifacts."
- It **appends 0.5 s of silence to the reference audio** by default to fix an ICL phoneme-bleed artifact
  (`append_silence=False` to match upstream).
- Parity caveat, stated honestly: static cache and dynamic cache are not bit-identical because SDPA kernel
  selection differs; side-by-side WAVs are shipped for judging. A slow dynamic-cache parity path exists for
  validation only (~0.77 s TTFA on RTX 4090 vs ~0.16–0.18 s fast path).
- Acknowledgements name `dffdeeq/Qwen3-TTS-streaming` and `tsdocode/nano-qwen3tts-vllm`.
- CLI: `faster-qwen3-tts clone|custom|design|serve`, `--streaming`, prints RTF; demo UI shows live TTFA/RTF;
  OpenAI-compatible `/v1/audio/speech` server.

### 5.3 Details of the pure-C engine (best non-Python option)

From <https://raw.githubusercontent.com/bon5co/qwen3-tts/main/docs/cuda-performance.md> and
<https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/README.md>:

- Architecture facts worth noting: **Talker = 28-layer Qwen3 transformer with GQA/RoPE/SwiGLU; Code Predictor
  = 5 layers running 15 sequential passes per frame; Speech Decoder = causal ConvNet, 16-codebook RVQ
  dequantization, 480× upsampling → 24 kHz**. "Code Predictor: 1024 hidden, 5 layers (+2048→1024 projection
  for 1.7B)". Memory: 0.6B ~3 GB, 1.7B ~8 GB (BF16 mmap).
- CUDA RTF table (measured on a "mainstream ~270 GB/s NVIDIA GPU (**RTX 4060-class**)", 1.7B):
  naive per-op GPU offload 1.47 → resident fused int8 **0.55** → resident fused `--quant-mixed` (int4 Talker +
  int8 CP) **0.44**.
- Bandwidth-scaled estimate table (extrapolated, flagged as such in the source): RTX 3060 ~0.33, RTX 4070
  ~0.24, RTX 3090/4080 ~0.13–0.17, RTX 4090 ~0.12.
- Batching (`QWEN_CUDA_BATCH=1`, `--batch-size N` ≤ 8): 3.35× per-step at B=8 (Talker 4.1×, CP 2.7×), ~3×
  end-to-end; "a request's audio is **byte-identical** whether it runs solo or inside a batch of 8 (md5 match)".
- Honest scaling limit stated: "decode is bandwidth-bound only up to a point — past it, single-stream becomes
  **launch-latency-bound** (the A100's 5–6× bandwidth did not translate to 5× RTF), so big cards pay off in
  **batching**, not single-stream."
- CPU/Metal numbers: M1 CPU 0.6B `--int8` 0.69 / `--int4` 0.52; M2 Pro Metal 0.6B 0.36–0.39, 1.7B 0.48–0.53;
  M4 Metal 0.6B 0.28 / 1.7B 0.41 (int4); streaming TTFA 314 ms (M2 Pro 0.6B), 469 ms (M1 0.6B).
- CUDA streaming **server** is explicitly "work in progress… never met a serving KPI target"; the CPU server
  is the qualified path (30-minute soaks).
- Opinionated and important: "**a stream can average twice real time and still hiccup for two seconds, and it is
  the hiccup a listener hears. One measured case had RTF 0.90 while 35 % of streams stalled** against a
  one-second buffer."

### 5.4 Alternative fast TTS engines (verified existence/licenses; latency mostly unverified)

Verified via HF API in this pass:

| Repo ID | License | Languages | Downloads | Created | Notes |
|---|---|---|---|---|---|
| `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | **apache-2.0** | zh, en, fr, es, ja, ko, it, ru, de | 187,753 | 2025-12-11 | 0.5B; 645 likes; onnx+safetensors |
| `FunAudioLLM/CosyVoice2-0.5B` | apache-2.0 | same 9 | 4,514 | 2025-07-29 | also `CosyVoice-300M`, `-300M-SFT`, `-300M-Instruct` |
| `IndexTeam/IndexTTS-2` | **license:other** | en, zh | 9,994 | 2025-06-18 | 783 likes; arXiv 2506.21619 |
| `IndexTeam/IndexTTS-2.5` | **license:other** | zh, en, ja, es, ar | 13,993 | **2026-08-10** | tags: `emotion-controllable`, `cross-lingual`, `zero-shot`; arXiv 2601.03888 |
| `hexgrad/Kokoro-82M` | **apache-2.0** | en | **11,687,399** | 2024-12-26 | 82M params; `hexgrad/Kokoro-82M-v1.1-zh` (apache-2.0, 230 likes) adds Chinese; ONNX/MLX/ExecuTorch variants exist |
| `microsoft/VibeVoice-1.5B` | **MIT** | en, zh | 474,084 | 2025-08-25 | 2,486 likes |
| `microsoft/VibeVoice-Realtime-0.5B` | **MIT** | **en only** | 349,658 | 2025-12-04 | `vibevoice_streaming` library, "Realtime TTS", "Streaming text input"; base `Qwen/Qwen2.5-0.5B` |

**Could not verify in this pass** (no source fetched): F5-TTS, Fish-Speech/OpenAudio, SparkTTS, MegaTTS3,
Higgs Audio v2/v3, GPT-SoVITS, Matcha-TTS, Piper — parameter counts, VRAM, streaming support, cloning/instruct
support and licenses. Do not treat anything about those as established from this report.

Two indirect but citable data points on alternatives:

- A Chinese-speaking user on an **RTX 2080 Ti** in issue #89 (comment 3811768740, 2026-01-28): Qwen3-TTS took
  "**生成3秒音频需要15秒左右**" (~15 s for 3 s of audio) while "**Index TTS2 / GPT-SoVits 都很快（2秒左右）**"
  (~2 s). This is a single anecdote, hardware-stated, not a controlled benchmark.
- Qwen's own README long-form benchmark places VibeVoice **above** Qwen3-TTS on long English:
  `long-en` WER VibeVoice 1.780 vs `Qwen3-TTS-12Hz-1.7B-CustomVoice` 2.812, and `long-zh` VibeVoice 22.619 vs
  Qwen 2.356 (so VibeVoice is much worse on Chinese long-form). VoxCPM 4.835/7.474, Higgs-Audio-v2 (chunk)
  5.505/6.917 (<https://pypi.org/project/qwen-tts/>).
  Note the vLLM-Omni blog describes **VoxCPM2** as a single-stage tokenizer-free hybrid (MiniCPM4 28L → FSQ →
  ResidualLM 8L → LocDiT CFM → AudioVAE, 48 kHz) reaching RTF ~0.13 on H20 — i.e. a genuinely faster
  architecture class, but that is server-GPU data.
- CosyVoice 3 beats Qwen3-TTS on Seed-TTS test-zh WER (0.71 vs 0.77) per Qwen's own table, and CosyVoice3 is
  **apache-2.0** at 0.5B — the most direct license-clean alternative in the zh+en space.

---

## 6. Reports specifically about 8 GB VRAM constraints with Qwen3-TTS

### 6.1 Official position: 0.6B yes, 1.7B wants 16 GB

From the official FAQ (<https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/faq>), verbatim:

- **Minimum:** Python 3.9+ (3.12 recommended); "**CUDA-compatible GPU with 8GB+ VRAM**"; 16 GB+ system RAM.
- **Recommended for 1.7B models:** Python 3.12; "**NVIDIA GPU with 16GB+ VRAM** (A100, RTX 4090, etc.)";
  32 GB+ system RAM; FlashAttention-2 support (Ampere or newer).
- **For 0.6B models:** "**8GB VRAM is sufficient**"; 16 GB system RAM.

So Qwen's own guidance says *your 8 GB card is below the recommendation for the 1.7B model* — which matches the
slow-path symptom, since part of the problem is that nothing is batched or resident.

### 6.2 Official OOM mitigation list (and its caveat)

From the same FAQ, "I'm getting CUDA out of memory errors":

1. Use the smaller model — `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.
2. "**Use float16 instead of bfloat16**".
3. Reduce batch size — "process one at a time instead of batching".
4. Reduce `max_new_tokens` — "Default is 2048".
5. `torch.cuda.empty_cache()`.

**Caveat:** recommendation 2 collides with the open PR #355 "cast logits to float32 to prevent FP16 sampling
overflow" (2026-08-07, from the repo's issue list). fp16 on this model is a known risk area.

### 6.3 Concrete memory footprints (so you can plan an 8 GB budget)

- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` on ModelScope: repo `StorageSize` **4,520,219,726 B (≈4.21 GiB)**,
  with `model.safetensors` **3,833,402,552 B** and `speech_tokenizer/model.safetensors` **682,293,092 B**;
  `model_size` 2,087,233,793 params; license apache-2.0.
  Source: <https://modelscope.cn/api/v1/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice>
- BF16 GGUF talker sizes (§2.4): 1.7B 3.58 GiB, 0.6B 1.69 GiB; plus tokenizer 0.33 GiB BF16 / 0.24 GiB Q4_K_M.
- INT8/INT4 GGUF: 1.7B Q8_0 1.90 GiB, Q4_K_M 1.10 GiB; 0.6B Q8_0 0.90 GiB, Q4_K_M 0.56 GiB.
- The pure-C engine states its own footprint as "0.6B ~3 GB, 1.7B ~8 GB" for BF16 mmap — i.e. the 1.7B at BF16
  is right at the edge of 8 GB.
- A user on RTX 3090 reported "~8 GB allocated" while running 1.7B-CustomVoice (HF discussion #18).
- Dozens of community 8-bit/6-bit/4-bit MLX and Q4/Q8 GGUF builds exist (§2.4) specifically because of this.

### 6.4 Direct evidence of what works on a 4060 / 8 GB-class card

Only three data sources published something for this exact class:

1. **`faster-qwen3-tts`, RTX 4060 (Windows), 8 GB-class** — 0.6B RTF 0.23→**2.26**, 1.7B RTF 0.23→**1.83**;
   TTFA 2697→413 ms (0.6B) and 2905→460 ms (1.7B). CUDA graphs "not reliable on torch ≤ 2.5.0"; validated
   ≥ 2.5.1. Blackwell/RTX 50xx needs cu128 wheels (your `torch 2.7.0+cu128` is fine).
   <https://github.com/andimarafioti/faster-qwen3-tts>
2. **Pure-C CUDA engine, "RTX 4060-class (~270 GB/s)"** — 1.7B `--quant-mixed` **RTF 0.44**; batching ~3.35×
   at B=8. Multi-arch binary `sm_80/86/89/120` so one build covers Ampere/Ada/Blackwell.
   <https://raw.githubusercontent.com/gabriele-mastrapasqua/qwen3-tts/main/docs/cuda-performance.md>
3. **RTX 4070 mobile (8 GB laptop class), 0.6B base, fork optimizations** — RTF 2.82 → **0.86–0.98**
   (issue #89 comment 3812963804).

**Could not verify:** no source found that reports Qwen3-TTS specifically on an **RTX 4060 *Laptop*** (as
opposed to a desktop 4060 or a generic "4060-class") other than the `faster-qwen3-tts` "RTX 4060 (Windows)"
row, whose desktop-vs-laptop status is not stated. Also no source reporting measured *peak* VRAM for the 1.7B
on an 8 GB card.

### 6.5 Concurrency and 8 GB

Issue #350 (§3.7) is relevant: one model instance does not handle concurrent calls well, and the official FAQ's
batching advice competes with VRAM. On 8 GB, the practical pattern is a **single-instance queue with a small
batch of 2–4**, not thread-per-request. Note the vLLM-Omni RFC #3163 explicitly warns that the full CUDA-graph
cartesian capture set costs **~4.4 GB** of GPU memory (44 graphs) and was rejected as a default for exactly
that reason, while the conservative default adds ~150 MB.

---

## 7. LOCAL GROUND TRUTH — measured on this machine (not sourced from the web)

Measured 2026-09-19 on the actual target hardware. Script: [`bench/bench_tts.py`](bench/bench_tts.py).
Method: load the checkpoint with the project's own settings (`bf16`, `sdpa`, `device_map=cuda:0`), then call the
inner `Qwen3TTSForConditionalGeneration.generate()` and `speech_tokenizer.decode()` **separately** and time each
with `torch.cuda.synchronize()` around it. `RTF = wall_seconds / audio_seconds`, so **RTF > 1 is a slow-down**
here (this is the inverse of the `faster-qwen3-tts` table convention; `faster-qwen3-tts` reports
`audio/wall`, so their "0.23" == our "4.3").

Hardware: `NVIDIA GeForce RTX 4060 Laptop GPU`, 8188 MiB VRAM, driver 610.47 / CUDA UMD 13.3, ~115 W cap.
Web: `transformers 4.57.3`, `torch 2.7.0+cu128`, `qwen-tts 0.1.1`, `attn=sdpa`, `dtype=bfloat16`.

Text: `"Hello. This is a speed benchmark test."` (38 chars), speaker `aiden`, language `English`.

| Model | Run | Audio | Talker+CodePredictor (stage 1) | Codec decode (stage 2) | Total | RTF total | AR share |
|---|---|---|---|---|---|---|---|
| **1.7B** | 1 | 3.12 s | **19.242 s** | 0.664 s | 19.906 s | 6.38× | **97 %** |
| **1.7B** | 2 | 2.72 s | **16.223 s** | 0.112 s | 16.335 s | 6.01× | **99 %** |
| **0.6B** | 1 | 3.12 s | **15.342 s** | 0.627 s | 15.970 s | 5.12× | **96 %** |
| **0.6B** | 2 | 3.28 s | **15.092 s** | 0.100 s | 15.192 s | 4.63× | **99 %** |

VRAM (measured): 1.7B → **3 985 MB** allocated / 4 112 MB reserved after load, peak **4 085 MB** during
generation. 0.6B → **2 057 MB** allocated / 2 184 MB reserved, peak **2 162 MB**. These are *before* adding
~1 GB Faster-Whisper (fp16) and ~0.5 GB PP-OCRv6 (onnxruntime-gpu), which the server also loads.

### 7.1 What this proves

- **The codec/vocoder is NOT the bottleneck.** It is 0.04–0.66 s, i.e. **1–4 %** of total. Optimizing it
  (PR #191's `compile_codec`) cannot fix this; at best it addresses a rounding error. PR #191's "~85 % of
  batch time" is a **batch-of-12** figure and does not transfer to single-request local inference.
- **The autoregressive code-generation loop is 96–99 % of the time.** Per frame: 1 Talker pass (28 layers) plus
  **15 sequential Code-Predictor passes** (5 layers each), because `num_code_groups = 16`. A 39-frame sentence
  therefore costs ~40 Talker forwards + ~585 Code-Predictor forwards — all serial, all tiny, all dominated by
  Python dispatch and kernel-launch latency rather than arithmetic.
- **This is why 0.6B is not meaningfully faster than 1.7B** (15.1–15.3 s vs 16.2–19.2 s — and run 2 of 1.7B was
  *slower* than both 0.6B runs). The Talker is 3× smaller in the 0.6B, but the **Code Predictor is identical**
  (5 layers, hidden 1024, num_code_groups 16) and the per-step overhead is fixed. Shrinking the model does not
  shrink the overhead. Corroborated by `faster-qwen3-tts`, whose Jetson breakdown puts the Predictor's 15 steps
  at 190 ms of a 330 ms step (58 %) *before* optimization.
- **RTF ≈ 4.6–6.4× means a 3-second sentence needs 15–20 s of wall time.** The project README's
  "TTS（短句）~10–15s" is accurate and is a property of the model+stack, not of any local misconfiguration.

### 7.2 Environment defect found during measurement (independent of TTS speed)

`import qwen_tts` **hung for >7 minutes** in this environment. `faulthandler` traced it to
`librosa.core.notation` → `numba` `@jit(cache=True)` → `numba.core.caching.ensure_cache_path` →
`tempfile._mkstemp_inner`. numba's default JIT cache directory is not writable from this sandbox, so import
blocks instead of failing. Setting `NUMBA_CACHE_DIR` to a writable path fixed it: **>420 s → 17.2 s**.
This is worth hardening in the service (see `bench/README.md`); it also means any timing/startup measurement
taken without that variable set is meaningless.

---

## Actionable conclusions

**Ranked by impact per unit of effort, for `Qwen3-TTS-12Hz-{1.7B,0.6B}-CustomVoice` on an RTX 4060 Laptop 8 GB.**

1. **Stop waiting for upstream, and stop expecting a PyPI upgrade to help.** `qwen-tts 0.1.1` (2026-02-06) is
   the newest release; upstream's last commit is 2026-03-17; there are no releases or tags; and the only
   merged latency work anywhere is in vLLM-Omni, not in `qwen-tts`. Your pin is not "outdated" — it is the
   current release.

2. **Highest-leverage single change: adopt the CUDA-graph + static-KV-cache approach.** The published
   RTX 4060 numbers (`faster-qwen3-tts`: 0.6B 0.23→2.26 RTF, 1.7B 0.23→1.83 RTF) match your observed
   10–15 s/sentence almost exactly, and the root-cause explanation (~500 kernel launches/step, GPU idle
   waiting on Python) is corroborated independently by vLLM-Omni's own 6–14 % GPU-utilization measurements.
   `pip install faster-qwen3-tts` (note it pulls `qwen-tts-hf` for Transformers 5 — do not install both
   `qwen-tts` and `qwen-tts-hf` in one environment). It needs torch ≥ 2.5.1; your 2.7.0+cu128 qualifies.
   Sources: <https://github.com/andimarafioti/faster-qwen3-tts>,
   <https://pypi.org/pypi/qwen-tts-hf/json>.

3. **If you must stay on upstream 0.1.1, apply the community patch set yourself — but target the AR loop.**
   In descending order of value and ascending order of invasiveness:
   - `torch.set_float32_matmul_precision('high')` — one line, free (the project already does this).
   - **CUDA-graph / static-KV-cache the Talker + Code-Predictor decode step.** Per §7 this is where 96–99 % of
     the time is. This is the item that matters.
   - `use_fast_codebook=True` (bypass HF `generate()` for the 15 codebook tokens — this alone is claimed at
     2–3×), `torch.compile` on Talker and CodePredictor, and removing the CPU sync in the EOS check.
   - **Deprioritize PR #191's `compile_codec`.** It optimizes the codec decoder, which §7 measures at 1–4 % of
     local single-request time. The author's ~85 % figure is a batch-of-12 ROCm workload and does not describe
     your case. Still cheap to try, just do not expect the win here.
   Sources: <https://github.com/QwenLM/Qwen3-TTS/pull/191>,
   <https://github.com/QwenLM/Qwen3-TTS/issues/89#issuecomment-3799395212>,
   <https://github.com/dffdeeq/Qwen3-TTS-streaming>, and the local measurements in §7.

4. **Consider Linux/WSL2 before buying hardware.** One user measured the *same* hardware going from RTF 2.91
   (Windows) to 1.02 (Linux) — a 3× improvement attributed to CPU contention
   (<https://github.com/QwenLM/Qwen3-TTS/issues/89#issuecomment-3810737385>). Single report, uncontrolled, but
   it is free to test and consistent with the "host-bound, not GPU-bound" diagnosis. Also install
   **flash-attn 2** — it is not a big win once compiled, but it *reduces VRAM*, which matters at 8 GB
   (<https://mintlify.wiki/QwenLM/Qwen3-TTS/resources/faq>).

5. **Fix your serving shape.** (a) Never call one `Qwen3TTSModel` instance from multiple threads — it is slower
   than sequential (issue #350). Use a single-worker queue. (b) Precompute the clone prompt once with
   `create_voice_clone_prompt()` — official docs claim 2–4 s first generation vs <1 s with a cached prompt.
   (c) Batch 2–4 texts through one `generate_*` call rather than concurrency.

6. **On 8 GB, use the 0.6B model as the default and treat 1.7B as optional.** Qwen's own FAQ puts 1.7B at
   "16 GB+ VRAM recommended" and 0.6B at "8 GB sufficient". The 1.7B BF16 footprint is ~3.83 GB (LLM) +
   0.68 GB (codec) ≈ 4.5 GB before activations/KV; an INT4/INT8 GGUF talker (1.10–1.90 GiB) plus a Q4 codec
   (~0.24 GiB) leaves far more headroom. Do **not** take the FAQ's "use float16" advice without checking
   open PR #355 (FP16 sampling overflow).

7. **The 25 Hz models are the intended faster path — and they are still unreleased.** If 25 Hz ships, it is
   documented as "faster inference, slightly lower quality" and would be a drop-in speed lever.
   Verified as of now: no `Qwen/Qwen3-TTS-25Hz-*` on HF or ModelScope. Do not plan around it.

8. **If you conclude Qwen3-TTS cannot be made fast enough, the license-clean zh+en alternatives to evaluate
   first are `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (apache-2.0, 0.5B, 9 languages, and it beats Qwen3-TTS on
   Seed-TTS test-zh per Qwen's own table) and `hexgrad/Kokoro-82M` / `-v1.1-zh` (apache-2.0, 82M, <12M
   downloads — extremely light). For streaming English only, `microsoft/VibeVoice-Realtime-0.5B` (MIT) is a
   purpose-built realtime model. `IndexTeam/IndexTTS-2` / `2.5` is attractive technically
   (emotion-controllable, multilingual) but is `license:other` — check commercial terms. The pure-C engine
   (`gabriele-mastrapasqua/qwen3-tts`, MIT) is the best option if you want to keep the exact Qwen3-TTS voice
   quality but escape Python: RTF 0.44 on 1.7B / 4060-class, plus INT4/INT8 and batching.

9. **Measure GPU utilization before and after.** If `nvidia-smi` shows ~10–16 % during generation (as
   essentially every report does), the bottleneck is host/launch overhead and any GPU-side tuning you do
   (dtype, attention backend, more VRAM) will not be the fix. If it shows high utilization after applying
   CUDA graphs, then you are finally GPU-bound and further gains need quantization or a smaller model.

---

## Verification notes and gaps

**Tooling caveats (reported honestly):**

- `web_fetch` on GitHub HTML pages returns almost only navigation chrome. I used the GitHub REST API
  (`api.github.com`) and `raw.githubusercontent.com` instead throughout.
- `web_search` returned source *lists* with very short snippets rather than summarised answers, so all factual
  claims here come from direct API/page fetches, not from search snippets.
- ModelScope: `https://modelscope.cn/api/v1/models/{repo}` works; the collection-list and
  `/api/v1/dolphin/models` endpoints returned **404**. I therefore verified only one ModelScope repo in detail
  (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`) and the absence of `Qwen/Qwen3-TTS-25Hz-1.7B-Base`.
- The official docs at `mintlify.wiki/QwenLM/Qwen3-TTS/*` are third-party-hosted mirrors of Qwen's docs; the
  `.md` suffix variant returns the clean source. I treat them as official content because they link to
  `github.com/QwenLM/Qwen3-TTS` and the arXiv report, but they are not on a Qwen-controlled domain.
- `web_fetch` truncates long pages; for the PyPI JSON I verified the `releases` map by also fetching the
  simple index, which lists every file for every version. Both agree: no version beyond 0.1.1.

**Explicitly NOT verified (do not rely on these):**

- Any RTF/VRAM numbers for F5-TTS, Fish-Speech/OpenAudio, SparkTTS, MegaTTS3, Higgs Audio, GPT-SoVITS,
  Matcha-TTS, or Piper.
- Whether the "RTX 4060" in the `faster-qwen3-tts` table is a laptop or desktop part.
- Measured peak VRAM for 1.7B on an 8 GB card.
- The actual content of the `compshare.cn` "Faster-Qwen3-TTS" Bilibili-linked page.
- Any Reddit / Hacker News / Zhihu / CSDN / Juejin thread with measurements.
- Any prefix-cache / cross-request KV reuse implementation for Qwen3-TTS.
- Whether PR #191 or #190/#360 will ever be merged (all are open with no maintainer response visible).
- The substance of issue #237 beyond its title, and the full comment thread of issue #34.
