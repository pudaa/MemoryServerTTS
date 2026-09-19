# Fast local TTS engines to replace/complement Qwen3-TTS 1.7B — technical evaluation (late 2026)

**Target:** one RTX 4060 Laptop 8 GB, Windows, Python 3.12, torch 2.7.0+cu128, transformers 4.57.3, sharing the GPU with Faster-Whisper base fp16 (~1 GB) + PP-OCRv6 small ONNX (~0.5 GB). Budget for a new TTS: **~1.5–3 GB VRAM**, no separate serving stack (no vLLM/TensorRT server, no multi-GPU).

**Baseline to beat:** Qwen3-TTS 1.7B bf16, 19.2 s wall for 3.12 s audio ⇒ RTF ≈ 6.4× (RTFx ≈ 0.16), 96–99 % of it in the AR code-generation loop. Model card confirms 1.929 B params bf16 ([HF API](https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)).

**Read this first — the single most important axis.** Every engine below is tagged NAR or AR:

- **NAR / flow-matching / single-pass** (Kokoro, Piper, MeloTTS, Supertonic 3, Matcha-TTS, F5-TTS) generate a whole utterance (or a fixed few-step latent) in one or a handful of parallel passes. Cost scales with *output length*, not with a sequential per-frame loop. These are where the 20–100× wins live.
- **AR** (everything LLM-backed: Qwen3-TTS, CosyVoice 2/3, IndexTTS-2/2.5, VibeVoice, Chatterbox, Orpheus, CSM, Dia, Maya1, Higgs, MOSS-TTS\*, Sopro, VoxCPM, Fish/OpenAudio, FireRedTTS3, GLM-TTS, Gepard) pays a sequential decode per token/frame. A smaller AR model is *cheaper per frame* but hits the **same structural wall**; on a 4060 laptop that wall is exactly what makes Qwen3-TTS unusable.

---

## 1. Comparison table

**Speed-data conventions.** The only systematic multi-engine benchmark I found that measures TTFA + RTFx + peak VRAM on Windows/CUDA is [tts-bench](https://5uck1ess.github.io/tts-bench/speed.html) — but its rig is a **desktop RTX 5090 32 GB**, not a 4060 laptop. Treat its TTFA as an optimistic floor and its relative ordering as the useful signal. The one **directly relevant consumer-8 GB measurement** I found is a blogger's RTX 4070 8 GB run of Kokoro ([ideasfixer](https://ideasfixer.blogspot.com/2026/04/cpu-vs-gpu-is-hardware-acceleration.html)).

| Engine | Repo id | License | Params | Disk | VRAM | AR/NAR | Streaming | EN quality evidence | Measured speed evidence | transformers pin | Windows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Kokoro-82M** | `hexgrad/Kokoro-82M` | Apache-2.0 | 82 M | `.pth` fp32 ≈ 326 MB; ONNX fp32 310 MB / fp16 156 MB / int8 109 MB; voices 28 MB (54 v) | **925 MB peak, measured** (tts-bench GPU); ~850 MB on RTX 4070 8 GB | **NAR** (StyleTTS2-style, non-AR) | No true streaming token loop; per-sentence chunk yield (fast enough that it doesn't matter) | UTMOS **4.302**, WER **0.065** (tts-bench, 5 prompts); Arena screenshots only in `EVAL.md` | warm TTFA **67 ms**, RTFx **103.7×** (5090); **~508 ms TTFA, 580 char/s, ~850 MB** (RTX 4070 8 GB); CPU 14.4× | `transformers` unpinned (`kokoro` 0.9.4 → `huggingface-hub, loguru, misaki[en]>=0.9.4, numpy, torch, transformers`) | Installer `.msi` for espeak-ng documented in the PyPI README; silent-output reports for some non-EN voices |
| **Kokoro-82M-v1.1-zh** | `hexgrad/Kokoro-82M-v1.1-zh` | Apache-2.0 | 82 M | `kokoro-v1_1-zh.pth` + 103 voices; ONNX same 3 sizes, zh voices 53 MB | same class (≤1 GB) | **NAR** | as above | No first-party metrics; author grades all `zf_*`/`zm_*` voices **D** | same export lineage as v1.0 | same | same |
| **Piper** | `rhasspy/piper-voices` + `OHF-Voice/piper1-gpl` | **GPL-3.0-or-later** | ~25 MB per voice | ~25 MB/voice | CPU (no CUDA path used) | **NAR** (VITS) | chunked, sentence-level | UTMOS **4.077**, WER **0.066** (tts-bench) | warm TTFA **107 ms**, RTFx **58.8×**, 470 MB RAM — **CPU only** in tts-bench | `onnxruntime>=1,<2`, `pathvalidate`; `[zh]` extra: `transformers>=4,<6` | **Prebuilt `win_amd64` wheels** (piper-tts 1.8.0) |
| **Supertonic 3** | `Supertone/supertonic-3` | code MIT / **weights OpenRAIL-M** | 99 M | ~411 MB repo; SDK downloads ~400 MB | ONNX, CPU-first (GPU optional) | **NAR / flow-matching** (`duration_predictor` + `text_encoder` + `vector_estimator` + `vocoder` ONNX) | not token-streaming; sub-second full page | UTMOS **4.195**, WER **0.065** (tts-bench); supertonic 2-step: RTF 0.005–0.015 (4090), 0.012–0.015 (M4 Pro CPU) | RTFx **9.92×**, warm TTFA 741 ms (tts-bench, CPU) | `onnxruntime`, `numpy`, `soundfile`, `huggingface-hub` — **no transformers** | pure-Python + ONNX; no system binaries |
| **MeloTTS** | `myshell-ai/MeloTTS-English` | MIT | ~52 M | n/a | **1.16 GB measured** | **NAR** (VITS) | sentence chunking | UTMOS **3.498**, WER 0.058 (tts-bench) — intelligible but dated naturalness | warm TTFA **117 ms**, RTFx **60.4×** (5090) | not verified (not in the required list) | not verified |
| **Fun-CosyVoice3-0.5B-2512** | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Apache-2.0 | 0.5 B LLM + flow + HiFiGAN | 11.77 GB repo (all revisions); community GGUF 1.72 GB | **not measured on any consumer GPU I could find** | **AR LLM + flow-matching decoder** (bi-streaming) | **Yes — text-in + audio-out bi-streaming, vendor claims latency as low as 150 ms** | Vendor Seed-TTS-eval **test-en WER 2.24 / SIM 71.8** (RL variant: **1.68 / 69.5**); CV3-eval en WER 5.24 | Only datacenter numbers: TRT-LLM on **L20**, batch 1 RTF **0.109**, first-chunk **750 ms @ 4 concurrent** | installed from the GitHub source tree (`pip install -r requirements.txt`), not PyPI — **could not verify the pins** | No Windows instructions; `ttsfrd` wheel is `cp310` **linux_x86_64** only; `sox` needed on Linux; HF discussion titled "Windows Support?" carries a "It works fine in Windows" comment (snippet only) |
| **CosyVoice2-0.5B** | `FunAudioLLM/CosyVoice2-0.5B` | Apache-2.0 | 0.5 B | 4.85 GB repo | 3.2 GB claimed by a blog on a 4090 (low-authority) | **AR LLM + flow-matching decoder** | Yes (stream=True) | Seed-TTS-eval test-en **WER 2.57–3.09 / SIM 65.9** | Vendor/3rd-party only; streaming TTFA 1.5 s vs 3.5 s non-streaming on a 4090 **including browser audio-init overhead** (CSDN blog — weak source) | same as above | same as above |
| **VoxCPM-0.5B / 1.5 / 2** | `openbmb/VoxCPM-0.5B`, `VoxCPM1.5`, `VoxCPM2` | Apache-2.0 | 0.5 B / 0.6 B / 2 B | 2.91 GB (0.5B repo) | vendor table: **~5 GB (0.5B) / ~6 GB (1.5) / ~8 GB (2)**; tts-bench measured **5.65 GB** for VoxCPM2 | **AR + diffusion** (LocEnc→TSLM→RALM→LocDiT) | `generate_streaming()` exists | Seed-TTS-eval: VoxCPM-0.5B test-en **WER 1.85 / SIM 72.9**; VoxCPM2 **1.84 / 75.3**; Gepard's replica of Seed-TTS-eval: VoxCPM2 WER 0.015, SIM 0.867, UTMOS 2.42 | vendor RTF **~0.17 (0.5B) / ~0.30 (2B)** on 4090; tts-bench VoxCPM2 warm TTFA **5.19 s**, RTFx **1.35×** | `transformers>=4.36.2` — compatible | Python ≥3.10,<3.13 (3.12 OK); no system binaries stated |
| **Chatterbox-Turbo** | `ResembleAI/chatterbox-turbo` | MIT | 744 M | 4.04 GB repo | **3.01 GB measured** | **AR** (T3) + 1-step decoder | No realtime claim; paralinguistic tags are its headline | UTMOS **4.274**, WER **0.074** (tts-bench); Gepard's Seed-TTS-eval replica: Chatterbox WER **0.063**, SIM 0.796 | warm TTFA **1.62 s**, RTFx **4.28×**, 3.01 GB VRAM (5090) | **`transformers==5.2.0`, `torch==2.6.0`, `torchaudio==2.6.0`, `diffusers==0.29.0`, `gradio==6.8.0`** — hard conflicts with transformers 4.57.3 **and** torch 2.7.0+cu128 | pip wheel is `py3-none-any`; no Windows claim in the docs |
| **Chatterbox (original / multilingual)** | `ResembleAI/chatterbox` | MIT | 500 M / 1.2 B (tts-bench reports 1.2 B) | 23.65 GB repo (all variants) | **3.24 GB measured** | **AR** | no | UTMOS 4.423, WER 0.061 (tts-bench) | warm TTFA 2.61 s, RTFx 2.24× | same 5.x/torch-2.6 pin family via `chatterbox-tts` | as above |
| **VibeVoice-Realtime-0.5B** | `microsoft/VibeVoice-Realtime-0.5B` | MIT | 1.017 B bf16 (named 0.5B) | 2.04 GB repo | **2.62 GB measured** | **AR LLM + diffusion head** | Yes, streaming text input | Vendor: LibriSpeech test-clean **WER 2.00 / SIM 0.695**; SEED test-en **WER 2.05 / SIM 0.633**. tts-bench: UTMOS 4.043, **WER 0.148** | Vendor claims **~300 ms** first audible latency; **tts-bench measured warm TTFA 3.77 s, RTFx 2.39×** — the two disagree by >12× | `transformers` + `vibevoice_streaming` custom arch; not in the required pin analysis | no Windows-specific notes found |
| **MOSS-TTS-Nano-100M** | `OpenMOSS-Team/MOSS-TTS-Nano-100M` | Apache-2.0 (card) | 0.1 B + 20 M tokenizer | ~ (n/a) | **777 MB measured** | **AR** ("pure autoregressive Audio Tokenizer + LLM") | Yes | UTMOS 4.195-ish tier of small models; **not scored on tts-bench** | warm TTFA **4.92 s**, RTFx **1.61×**, 777 MB VRAM (5090) — fast-to-load, slow-to-speak | `transformers` (own repo install) | none stated |
| **MOSS-TTS-Realtime** | `OpenMOSS-Team/MOSS-TTS-Realtime` | Apache-2.0 | **2.332 B** bf16 | 4.68 GB repo | not measured | **AR** (`MossTTSRealtime`, streaming module shipped) | Yes (multi-turn dialogue context) | no first-party EN WER found | none | docs recommend **Transformers 5.0.0** | none stated |
| **Sopro V2 Turbo** | `samuel-vitorino/sopro-v2-turbo` | Apache-2.0 | 121.6 M | 1.31 GB repo | **1.22 GB measured (streaming)** / 993 MB (offline) | **AR** (+ 2-step acoustic solver) | Yes (`tts.stream`) | UTMOS 4.0-tier; not in tts-bench scores for default voice | vendor: **~300 ms TTFA on laptop CPU**, RTF **0.21 streaming / 0.24 offline on M3 CPU**, 0.07 on H100; tts-bench: warm TTFA **132 ms**, RTFx 8.88× (5090) | pip `sopro` | "runs in the browser via ONNX"; no Windows note |
| **Gepard 1.0** | `nineninesix/gepard-1.0` | Apache-2.0 (codec: NVIDIA Open Model License) | 555.7 M | 2.25 GB repo | not measured | **AR** (decoder-only, one pass per frame) | Yes, built for it | Seed-TTS-eval (1088 prompts): WER **0.036**, SIM 0.585, UTMOS 2.64, NISQA-MOS **4.25** (best naturalness in that table) | Vendor: **~25× realtime, TTFA ~50 ms on RTX 5090 — vLLM path only**; PyTorch runner "isn't tuned for throughput" | `transformers` + custom `qwen3_5_text` arch | none stated; **vLLM path conflicts with "no separate serving stack"** |
| **IndexTTS-2 / 2.5** | `IndexTeam/IndexTTS-2`, `IndexTeam/IndexTTS-2.5` | IndexTTS-2: no license file in card (Apache-2.0 per tts-bench); **2.5: bilibili-model-license (non-standard)** | 1.5 B / ~0.8 B GPT backbone | 5.92 GB / 5.49 GB | 2.5 card says **~6 GB**; tts-bench measured **7.60 GB** | **AR (GPT) + flow-matching mel decoder + BigVGAN** | segmentation, not token streaming | Seed-TTS-eval test-en WER **2.23 / SIM 70.6** | warm TTFA **6.03 s**, RTFx **1.11×**, 7.60 GB (5090) | 2.5 requires **Python 3.10–3.11** | **DeepSpeed does not install on Windows/py3.12** ([issue #393](https://github.com/index-tts/index-tts/issues/393)) |
| **F5-TTS** | `SWivid/F5-TTS` | **CC-BY-NC-4.0** | 0.3 B | 14.83 GB repo (all checkpoints) | **802 MB measured** | **NAR / flow-matching** | no | Vendor Seed-TTS-eval test-en WER **2.00 / SIM 64.7–67.0** | warm TTFA **845 ms**, RTFx **5.32×** (5090) | f5-tts package; not verified | not verified |
| **Spark-TTS-0.5B** | `SparkAudio/Spark-TTS-0.5B` | **CC-BY-NC-SA-4.0** | 0.5 B | 3.94 GB repo | not measured | AR (LLM + BiCodec) | no | test-en WER 1.98–3.14 / SIM 57.3 | none | vendored wav2vec2; not verified | not verified |
| **OpenAudio S1-mini / Fish-Speech** | `fishaudio/s1-mini` (gated) | **CC-BY-NC-SA-4.0**, gated (non-commercial checkbox) | 0.5 B dual-AR | 3.61 GB repo | not measured | **AR (dual-AR)** | no | Seed-TTS test-en WER 1.94 / SIM 55.0 | none on consumer GPU | fish-speech from source | not verified |
| **MegaTTS3** | `ByteDance/MegaTTS3` | Apache-2.0 | 0.5 B | 4.26 GB repo | not measured | AR (duration LM + diffusion transformer) + WaveVAE | no | Vendor-independent Seed-TTS-eval: test-en **WER 2.79 / SIM 77.1** (VoxCPM table) | none | not verified | not verified |
| **GPT-SoVITS** | `RVC-Boss/GPT-SoVITS` | MIT (repo) | ~0.1–0.2 B | not measured | not measured | **AR (GPT) + VITS decoder** | partial | CV3-eval en WER **12.5** (worst in the VoxCPM CV3 table) | none | not verified | community Windows support exists |
| **Higgs Audio v2/v3** | `bosonai/higgs-tts-2-3b-base`, `bosonai/higgs-audio-v3-tts-4b` | **other / Research (NC)** for v3 | 5.77 B bf16 (v2), 4 B (v3) | 34.7 GB (v2 repo) | v3 not on tts-bench | AR | no | v2 Seed-TTS test-en WER 2.44 / SIM 67.7 | none | not verified | not verified |
| **Qwen3-Omni** | `Qwen/Qwen3-Omni-30B-A3B-*` | other (Qwen) | 30 B MoE (A3B) | ~60 GB | ≫8 GB | AR MoE | no | Seed-TTS test-en WER **1.39** | n/a | needs recent transformers | n/a |
| **Step-Audio 2-mini / EditX** | `stepfun-ai/Step-Audio-2-mini`, `Step-Audio-EditX` | Apache-2.0 | 3 B class | not measured | not measured | AR | no | no EN WER found for the TTS head | tts-bench: EditX not scored | custom_code, transformers | not verified |
| **NeuTTS Air** | `neuphonic/neutts-air` | Apache-2.0, gated (affiliation form) | 748 M (BF16) / 748 M GGUF | 1.5 GB file | **3.26 GB measured** | **AR** (Qwen2ForCausalLM backbone) | no | UTMOS 4.0-tier; not in the tts-bench top-15 UTMOS list I read | warm TTFA 417 ms, RTFx **1.62×** (5090) | `transformers` (Qwen2) | n/a |
| **Kitten TTS Nano/Micro/Mini 0.8** | `KittenML/kitten-tts-nano-0.8-fp32` etc. | Apache-2.0 | 15 M (nano), ~50 MB files | ~50 MB fp32, int8 variant exists | CPU-first | **NAR** (StyleTTS 2 arch) | no | UTMOS **3.665**, WER **0.093** (Nano 0.1) — weakest naturalness of the tiny models | KittenTTS Nano 0.1 CPU: warm TTFA 1.20 s, RTFx 6.39× | `onnxruntime`, `phonemizer`, `espeakng-loader` | `espeakng-loader` bundles espeak data (no MSI needed); PyPI install is a GitHub release wheel for 0.8 |
| **Piper-adjacent / Matcha-TTS** | `shivammehta25/Matcha-TTS` | MIT | ~19 M | n/a | tiny | **NAR / conditional flow matching** | no | paper reports MOS ≈ 4.15 on LJSpeech (2023) — not comparable to 2026 systems | none | none | pure PyTorch |
| **Dia-1.6B / Orpheus-3B / CSM-1B / Maya1 / Zonos** | see individual repos | Apache-2.0 (Zonos, Dia, Orpheus, CSM, Maya1) | 1.6 B / 3.78 B / 1 B / 3 B / 1.62 B | 12.9 GB / 56.7 GB / – / – / 3.65 GB repos | tts-bench: Dia 6.32 GB, CSM 3.51 GB, Maya1 6.72 GB, Zonos 4.48 GB, Orpheus not measured | all **AR** | none (Dia/Orpheus/CSM) | Orpheus UTMOS 4.005/WER 0.102; CSM 4.152/0.114; Maya1 4.487/0.066; Zonos 3.5-tier; Dia UTMOS 2.387/WER **0.317** | warm TTFA 12–23 s, RTFx 0.3–0.6× (all fail) | mixed | mixed |

### AR vs NAR — the decisive axis, restated

- **NAR / flow-matching / single-pass → recommended:** Kokoro (82 M), Piper (VITS), MeloTTS (VITS), Supertonic 3 (flow-matching, 2–8 steps), Kitten TTS (StyleTTS2), Matcha-TTS.
- **AR but small + streaming, structurally cheaper than Qwen3-TTS:** Fun-CosyVoice3-0.5B (0.5 B, 25 Hz semantic tokens + flow decoder, bi-streaming), CosyVoice2-0.5B, MOSS-TTS-Nano (0.1 B — but measured *slower* than Kokoro by 70× in RTFx), Sopro V2 Turbo (121 M + 2-step solver).
- **AR and would hit the same wall:** VibeVoice-{Realtime,1.5B,7B}, Chatterbox(Turbo), IndexTTS-2/2.5, Orpheus-3B, CSM-1B, Dia-1.6B, Maya1, Higgs v2/v3, MOSS-TTS 8 B, Fish/OpenAudio, MegaTTS3, FireRedTTS3, GLM-TTS, Gepard, Voxtral-4B-TTS, Zonos.
- **Gray zone — "AR + diffusion" (VoxCPM family):** AR backbone ("TSLM/RALM") plus a diffusion decoder (LocDiT). Measured 1.35× RTFx on a 5090 and 5.65 GB VRAM ⇒ fails both budget and latency.

---

## 2. Top-3 recommendations

### #1 — Kokoro-82M, on GPU, PyTorch (`pip install kokoro`) — *the drop-in fix for the chat box*

**Why:** it is the only engine in this evaluation that is simultaneously (a) NAR, (b) Apache-2.0 with published voice presets, (c) measured **under 1 GB VRAM**, and (d) measured **sub-second TTFA with 580 char/s throughput on an 8 GB consumer GPU**. A 20–300 char English chat sentence is ~50–200 chars/s worst case; at the blogger's measured 580 char/s that is **under 0.5 s of compute for a 250-char sentence**, comfortably inside your ~2 s target with whisper+OCR still resident on the other ~2.7 GB.

- **English quality:** UTMOS **4.302**, WER **0.065** on the tts-bench 5-prompt harness — the joint-highest naturalness of any sub-1 B model there, and beat by only Chatterbox/DramaBox/Higgs/LFM/Scylla (all ≥744 M, all AR and all ≥3 GB except LFM). The author's own `VOICES.md` flags the *male* American voices as weak (`am_adam` graded **F+**) — **use `af_heart`/`af_bella`/`bf_emma`**, not the `am_*` set.
- **Chinese caveat:** Kokoro-82M-v1.1-zh exists (Apache-2.0, 103 voices) but the author grades every `zf_*`/`zm_*` voice **D**. Chinese is a weak spot, not a strength — see the "hybrid" plan below.
- **Instruct/style:** **none.** No emotion/style/instruction channel at all. Only `speed` and the voice preset. This is the one thing you lose versus Qwen3-TTS `instruct`.
- **Voice cloning:** none (style vectors only). Not required for you.
- **Integration cost: lowest of any candidate.** `pip install kokoro>=0.9.4 soundfile`; `transformers` is unpinned; no torch/CUDA change; no `espeak-ng` for pure-English phonemisation of in-vocabulary words (`misaki[en]` is pure Python) — but install the espeak-ng MSI anyway for OOV fallback, per the PyPI README's own Windows section.
- **Watch out for:** the 100–200 token "goldilocks range" the author documents — very short utterances (<10–20 tokens) degrade. **This matters for your dictation use case** ("ahead" is 1 token). Mitigation the author suggests: bundle short utterances together. Since dictation word audio is pre-generated offline into a cache anyway, generate them in batches.

```powershell
pip install "kokoro>=0.9.4" soundfile
# espeak-ng (only needed for OOV fallback): install the .msi
#   https://github.com/espeak-ng/espeak-ng/releases
```
Optional ONNX sidecar (same Apache-2.0 weights, no torch graph, lower VRAM): download `kokoro-v1.0.fp16.onnx` (156 MB) + `voices-v1.0.bin` (28 MB) from [`kokoro-onnx` release `model-files-v1.1`](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1) and `pip install kokoro-onnx`. fp16 spectral correlation vs fp32 is **0.999**; int8 is **0.916** — **use fp16, not int8**, given your pronunciation-quality requirement.

### #2 — Supertonic 3 (ONNX, 99 M) — *the "zero-risk, zero-dependency" fallback, with a Chinese gap*

**Why:** 99 M params, four ONNX graphs, **dependencies are literally `onnxruntime + numpy + soundfile + huggingface-hub`** — it cannot break your torch/transformers stack, cannot pull a different CUDA, and needs no system binary. It is NAR (flow-matching with a separate duration predictor). Published vendor RTF is **0.005–0.015 on RTX 4090** (2-step) and **0.012–0.015 on an M4 Pro CPU**; tts-bench measured it CPU-only at RTFx 9.92× / TTFA 741 ms, i.e. it does not even need your GPU.

- **English quality:** UTMOS **4.195** / WER **0.065** (tts-bench) — statistically the same band as Kokoro. The vendor also publishes text-normalisation comparisons (currency, dates, phone numbers, units) where it claims to beat ElevenLabs/OpenAI/Gemini; that is a **vendor claim**, marked as such.
- **Presets + style tags:** 10 built-in voice styles (`F1–F5`, `M1–M5`) as JSON — this is the closest thing to a drop-in preset-voice model after Kokoro. 10 inline expression tags (`<laugh>`, `<breath>`, `<sigh>`, …) give you a *partial* substitute for `instruct`.
- **Blocker for your app:** **Chinese is not in the 31 supported languages.** `zh` is absent from both the HF card and the PyPI page. If Chinese chat replies must be read aloud, this cannot be the only engine.
- **License nuance (important):** the PyPI metadata declares **MIT** (`license_expression: "MIT"`, code), but the HF card declares the **weights** as **OpenRAIL-M**. OpenRAIL-M is commercially usable but carries downstream use restrictions — read it before shipping.

```powershell
pip install supertonic
# optional OpenAI-compatible local HTTP endpoint:
pip install "supertonic[serve]"
```

### #3 — Fun-CosyVoice3-0.5B-2512 — *the only candidate that matches Qwen3-TTS on features (presets + instruct + Chinese) while being 3.4× smaller*

**Why it is on the list:** it is the only engine here that keeps **English + Chinese + `instruct`-style control + a streaming path** in one Apache-2.0 package, and its English numbers are the best of the open small models in its own table (**test-en WER 2.24 / SIM 71.8**; RL variant **1.68 / 69.5** vs CosyVoice2's 2.57 / 65.9 and VibeVoice-1.5B's 3.04 / 68.9). It is explicitly **bi-streaming** with a vendor-claimed **150 ms** latency floor, and it supports pronunciation inpainting (CMU phonemes for English) — directly useful for a pronunciation-teaching product.

**Why it is #3 and not #1 — and why you must measure before committing:**

1. **It is still AR** (LLM backbone). Your entire problem is the AR loop. Going 1.7 B → 0.5 B should cut per-frame cost ~3.4×, but **I found zero 8 GB-class VRAM or RTF measurements** for it. The only published speed data are TensorRT-LLM on an **L20** (RTF 0.109 at batch 1; first-chunk 750 ms at concurrency 4) — a datacenter card with a TensorRT/Triton stack you explicitly do not want.
2. **Installation is from the GitHub source tree**, not PyPI: `git clone --recursive`, conda, `pip install -r requirements.txt`, then `sys.path.append('third_party/Matcha-TTS')`. `ttsfrd` ships only as a `cp310` **linux_x86_64** wheel; `sox` is a Linux/CentOS apt/yum step. A HF discussion is literally titled "Windows Support?" — but I could **not read the comment bodies** (HF discussion HTML returns mostly chrome), so Windows viability is **unverified**.
3. **Disk:** 11.77 GB for the repo across revisions; the community GGUF re-export (`Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF`, Apache-2.0, F16/F32/Q2_K…Q8_0) totals **1.72 GB** with 859 M parameters including the frontend, plus int8 ONNX frontends — an **unofficial** export. Do not assume the GGUF path is supported by the reference code.

**Verdict:** keep it as the feature-complete fallback. Before adopting, run its `stream=True` path on your own card and measure TTFA/RTF/VRAM; if the 0.5 B AR loop still cannot make 2 s for a 200-char sentence on a 4060, it fails for the same reason Qwen3-TTS does.

```powershell
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt
# then: sys.path.append('third_party/Matcha-TTS')
```

---

## 3. NOT viable / avoid (and why)

| Engine | Failure mode |
|---|---|
| **VibeVoice-Realtime-0.5B** | **Product-blocking, not technical.** Model card: *"Embedded an audible disclaimer (e.g. 'This segment was generated by AI') automatically into every synthesized audio file"*, plus an imperceptible watermark, plus *"We do not recommend using VibeVoice in commercial or real-world applications"*. You cannot read a chat reply aloud with an AI disclaimer spliced in. Also **English-only is enforced as a licence term** ("Unsupported language"). And tts-bench measured **3.77 s warm TTFA / 2.39× RTFx** in a single-speaker setup — the "~300 ms" headline did not reproduce. |
| **VibeVoice-1.5B / 7B** | 2.704 B and 7 B bf16; measured 5.26 GB / 17.63 GB VRAM, RTFx 1.61× / 1.55×. Same disclaimer/watermark regime. |
| **IndexTTS-2 / IndexTTS-2.5** | **7.60 GB measured VRAM** + 6.03 s TTFA + 1.11× RTFx ⇒ both VRAM and latency fail. 2.5 needs **Python 3.10–3.11** (you are on 3.12) and its DeepSpeed dependency **fails to compile on Windows/py3.12**. 2.5's licence is the non-standard `bilibili-model-license`. |
| **Chatterbox / Chatterbox-Turbo** | Hard dependency conflict: `transformers==5.2.0` and `torch==2.6.0` (exact pins on Python <3.14) vs your transformers 4.57.3 and torch 2.7.0+cu128. Installing normally would downgrade torch and break Faster-Whisper/Qwen3-TTS. VRAM 3.01 GB (Turbo) / 3.24 GB (base) is already over budget, and Turbo is AR with a measured 1.62 s TTFA — slower than Kokoro by 24×. Only revisit if you are willing to run it in a **separate venv + separate process**, and even then it loses on latency. |
| **F5-TTS** | Licence **CC-BY-NC-4.0** — non-commercial. Also NAR but only 5.32× RTFx / 845 ms TTFA, i.e. 20× slower than Kokoro for the same class of model. |
| **Spark-TTS, OpenAudio S1-mini / Fish-Speech, Fish-S2-Pro** | Non-commercial licences (CC-BY-NC-SA-4.0 / research-only), gated. Fish/S1 are **dual-AR** and S2-Pro is 4 B — the 5090 head-to-head measured Fish-S2-Pro at **RTFx 0.39×**, TTFA **9.74 s** uncompiled. |
| **F5-TTS / Spark / OpenAudio on quality-per-VRAM** | Even ignoring licences, vendor-intelligibility WER is *not* better than Kokoro/CosyVoice3 while being 6–10× larger. |
| **Higgs Audio v2/v3** | 5.77 B / 4 B params, 34.7 GB repo, "other"/Research(NC) licence, AR. v2's test-en WER 2.44 is worse than Fun-CosyVoice3-RL's 1.68. |
| **Qwen3-Omni 30B-A3B** | 30 B MoE, ≫8 GB VRAM, needs a serving stack. Non-starter. |
| **Orpheus-3B, CSM-1B, Dia-1.6B, Maya1, Zonos-v0.1** | All AR; measured warm TTFA 12–23 s and RTFx 0.3–1.6× on a 5090. Dia also has a catastrophic WER of 0.317. Zonos v0.1 = 4.48 GB VRAM, 0.71× RTFx. |
| **VoxCPM-0.5B / 1.5 / 2** | Vendor VRAM table: **~5 / ~6 / ~8 GB**; tts-bench confirms VoxCPM2 at **5.65 GB** with 5.19 s TTFA / 1.35× RTFx. Fails the 1.5–3 GB budget outright. Note VoxCPM-0.5B's docker-free `pip install voxcpm` is otherwise clean (`transformers>=4.36.2`, Python <3.13) — the *only* reason to exclude it is footprint. |
| **FireRedTTS3** | 2.12 B, measured **14.18 GB VRAM**, 8.50 s warm TTFA, 0.83× RTFx. |
| **ZipVoice-123M** | Although Apache-2.0 and tiny on paper, tts-bench measured **53.16 GB peak VRAM** and 0.60× RTFx on CUDA — an evidently broken export. Avoid until that is explained. |
| **Matcha-TTS** | MIT and NAR, but it is a **training/research repo** (ICASSP 2024) with no production preset-voice checkpoint or packaging. Its architecture is what CosyVoice is built on; use CosyVoice or Supertonic instead of driving Matcha yourself. |
| **Piper** | Technically excellent (107 ms TTFA on CPU, prebuilt Windows wheels) but **GPL-3.0-or-later** ([`piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl), 5.6 k stars). Shipping GPL-3.0 code inside a proprietary backend is a real legal exposure. **Viable only if your legal position allows GPL, or you invoke a separate `piper.exe` process and accept the distribution analysis.** Otherwise use Kokoro (Apache-2.0), which is the same architecture class and faster on GPU. |
| **MOSS-TTS-Realtime** | 2.332 B params, docs recommend **Transformers 5.0.0** (conflicts with your 4.57.3), no measured EN quality, no measured consumer-GPU speed. |
| **Gepard 1.0** | Apache-2.0 and genuinely built for streaming, but its published speed (**25× RT, 50 ms TTFA**) is on the **vLLM path on an RTX 5090**, and the card says the plain PyTorch runner "isn't tuned for throughput". The vLLM requirement is exactly the heavyweight serving stack you excluded. Its WER (0.036) and SIM (0.585) are also the weakest in its own benchmark. |
| **GPT-SoVITS** | CV3-eval English WER **12.5 %** — the worst English number in the whole CV3 table. |
| **Kitten TTS Nano** | UTMOS **3.665** — a full 0.64 MOS below Kokoro for a 0.3 MB disk saving. Fine for beeps, wrong for a pronunciation tutor. |

---

## 4. Could not verify

1. **Any RTF/TTFA/VRAM measurement of any CosyVoice model on a 4060-class or 8 GB laptop GPU.** Searches returned only: (a) vendor claims, (b) a TensorRT-LLM benchmark on an **L20**, (c) an AI-written CSDN marketing post claiming 3.2 GB VRAM and 1.5 s/3.5 s TTFA on an **RTX 4090** whose own arithmetic includes ~1100 ms of browser/Web-Audio init overhead, and (d) a second CSDN post about streaming-vs-non-streaming. None is a credible 4060 measurement.
2. **Whether Fun-CosyVoice3-0.5B-2512 actually runs on Windows.** The HF discussion list shows a thread titled "Windows Support?"; the fetched HTML returned only navigation chrome, so I could not read the answer. The dependency evidence (Linux-only `ttsfrd` wheel, `sox` apt/yum step, POSIX-style install docs) points the other way.
3. **The exact `requirements.txt` pins for CosyVoice 2/3** (whether they force a different torch/transformers). The repo install is `pip install -r requirements.txt` from the source tree; I did not read the file.
4. **Kokoro first-party speed numbers.** `EVAL.md` in the Kokoro repo contains only **three JPEG screenshots** of arena leaderboards (dated 2025-02-26) and no numeric table. Every Kokoro number in this report comes from tts-bench (5090), the RTX 4070 blogger, or vendor-side comparisons from Supertone.
5. **Any first-party 4060 RTX speed number for Kokoro.** A content-farm "recipe" page targeting the 4060 says verbatim: *"No first-party speed numbers exist for the RTX 4060 yet"* and cites an A10G gist (96× RT) and community 4090 figures (RTF ~0.04–0.06).
6. **Whether Kokoro's silent-output Windows bug affects English voices.** Secondary source reports it for non-English voices (e.g. Spanish `em_alex`); I did not read the upstream issues.
7. **MOSS-TTS-Nano's licensing.** The HF card declares `apache-2.0`; the README's own License section says to treat the repository as *"not yet licensed for redistribution"* if no root `LICENSE` file is present. Contradictory — verify before any use.
8. **English quality of Kokoro from a single authoritative source.** The only objective scores are tts-bench's, computed over **5 prompts** — the site itself warns *"thin — WER is a failure-detector, not a fine ranking"*. Kokoro appears in **no** Seed-TTS-eval table I found (CosyVoice, VoxCPM, Gepard, MOSS tables all omit it), so there is **no cross-vendor, large-N English WER/UTMOS for Kokoro**.
9. **Supertonic 3's WER/CER figures.** The card shows them only as an image (`s3_vs_measured_wer_range_voxcpm2.png`) with no numeric table, so I could not extract the per-language numbers.
10. **FireRedTTS3's languages and reference numbers** — the repo has no `language:` cardData and no metrics table in the API metadata; only a paper id (arXiv:2608.17492) that I did not read.
11. **Whether the tts-bench harness uses each engine's real streaming path.** The VibeVoice-Realtime discrepancy (3.77 s measured vs ~300 ms claimed) strongly suggests it may not, so treat cross-engine TTFA comparisons from that site as indicative, and same-engine RTFx comparisons as the more trustworthy part.
12. **CosyVoice2's claimed "150 ms first-package latency"** — the equivalent sentence appears in the Fun-CosyVoice3 README for CosyVoice 3; I did not read arXiv:2412.10117 to confirm the CosyVoice2 number independently.

---

## 5. Actionable recommendation

**Ranked plan for this deployment:**

0. **Before anything else: test CUDA graphs on the Qwen3-TTS you already have.** tts-bench measured Qwen3-TTS-1.7B-Base at warm RTFx **0.70×** but the *same checkpoint* as "Qwen3-TTS 1.7B (CUDA-graph)" at warm RTFx **3.76×** with TTFA **1.60 s** (down from 8.95 s) — a **5.4× speedup for a capture-mode change, at essentially identical VRAM** (4.89 vs 4.64 GB). That is the highest return-per-hour item in this whole report, because it needs no new dependency, no new licence review, and no change to your voice/`instruct` API. It will not make an AR model NAR, but at a 5× factor your 19.2 s becomes ~3.5 s — and if the 4060 scales similarly it may already clear "usable" for the chat box. **Verify on your own card**; the 5.4× is a 5090 number.
1. **Adopt Kokoro-82M as the primary chat-box TTS** (`pip install "kokoro>=0.9.4" soundfile` + espeak-ng MSI + pre-download `af_heart`/`af_bella`/`bf_emma`). It is the only candidate that is NAR, Apache-2.0, ~925 MB VRAM, and measured at 580 char/s on an 8 GB consumer GPU. Expected effect: chat-reply audio goes from ~19 s to well under 1 s of compute, freeing ~3–4 GB of the card. Constraints to design around: **no `instruct`** (replace style control with per-preset voices or drop it), and **weak short-utterance handling** (relevant to dictation) — batch the dictation words.
2. **Add Supertonic 3 as a zero-dependency second engine** (`pip install supertonic`) for English-only paths and for its expression tags. It adds ~0 to your dependency risk and runs acceptably on CPU, so it can absorb load without touching the GPU. **It cannot cover Chinese.**
3. **Keep Qwen3-TTS (with CUDA graphs) as the Chinese + `instruct` + voice-cloning tier.** Do not delete it. Route: English chat → Kokoro; Chinese chat and any styled/cloned output → Qwen3-TTS; offline dictation word cache → Kokoro in batches (best-of-N + your existing ASR verification still applies, and it will now be ~100× cheaper per candidate).
4. **Pilot Fun-CosyVoice3-0.5B only if the Qwen3-TTS Chinese/instruct path becomes the bottleneck.** Budget half a day for a Windows install attempt and a real TTFA/VRAM measurement before any commitment; the dependency story is Linux-shaped and 11.77 GB of disk.
5. **Do not spend time on:** VibeVoice (disclaimer/watermark + commercial-use warning), IndexTTS-2/2.5 (>7.6 GB, py3.11, Windows DeepSpeed failure), Chatterbox (transformers 5.2 + torch 2.6 pins would downgrade your stack; 3 GB VRAM; AR), F5-TTS/Spark/OpenAudio/Fish (non-commercial licences), Piper (GPL-3.0), MOSS-TTS-Realtime (transformers 5.0), Gepard (vLLM-only speed), VoxCPM family (5–8 GB VRAM), FireRedTTS3 (14 GB), ZipVoice (broken CUDA export).

**The one risk to watch:** Kokoro's `speed` knob and its documented weakness on very short inputs. If your dictation cache must hold single-word clips like "ahead" at tutor-grade pronunciation quality, validate Kokoro on 20–30 of those words **before** you decommission Qwen3-TTS for that path — and if they are not good enough, use Kokoro for sentence audio and keep Qwen3-TTS (CUDA-graphed) for the word cache, where latency does not matter because it is pre-generated offline.

---

### Primary sources

- [Fun-CosyVoice3-0.5B-2512 model card](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) · [HF API](https://huggingface.co/api/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) · [CosyVoice GitHub (now QwenAudio/CosyVoice)](https://github.com/QwenAudio/CosyVoice) · [CosyVoice3 TensorRT-LLM/Triton README](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice3.md) · [CosyVoice2-0.5B HF API](https://huggingface.co/api/models/FunAudioLLM/CosyVoice2-0.5B)
- [Kokoro-82M HF API](https://huggingface.co/api/models/hexgrad/Kokoro-82M) · [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) · [EVAL.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/EVAL.md) · [kokoro on PyPI (0.9.4)](https://pypi.org/project/kokoro/) · [kokoro-onnx model-files-v1.1 release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1) · [Kokoro on RTX 4070 8 GB benchmark](https://ideasfixer.blogspot.com/2026/04/cpu-vs-gpu-is-hardware-acceleration.html)
- [VibeVoice-Realtime-0.5B model card](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) · [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)
- [IndexTTS-2.5 model card](https://huggingface.co/IndexTeam/IndexTTS-2.5) · [index-tts issue #393 (DeepSpeed/Windows)](https://github.com/index-tts/index-tts/issues/393)
- [ResembleAI/chatterbox HF API](https://huggingface.co/api/models/ResembleAI/chatterbox) · [chatterbox-turbo HF API](https://huggingface.co/api/models/ResembleAI/chatterbox-turbo) · [chatterbox-tts on PyPI (0.1.7)](https://pypi.org/project/chatterbox-tts/)
- [Supertonic 3 model card](https://huggingface.co/Supertone/supertonic-3) · [supertonic on PyPI (1.3.1)](https://pypi.org/project/supertonic/)
- [VoxCPM-0.5B model card](https://huggingface.co/openbmb/VoxCPM-0.5B) · [voxcpm on PyPI (2.0.3)](https://pypi.org/project/voxcpm/) · [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Sopro V2 Turbo model card](https://huggingface.co/samuel-vitorino/sopro-v2-turbo) · [Gepard 1.0 model card](https://huggingface.co/nineninesix/gepard-1.0) (Seed-TTS-eval table incl. Chatterbox, Qwen3-TTS, VoxCPM2)
- [MOSS-TTS-Nano-100M card](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) · [MOSS-TTS-Realtime card](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) · [MOSS-TTS-Realtime HF API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS-Realtime)
- [tts-bench speed page (Windows RTX 5090)](https://5uck1ess.github.io/tts-bench/speed.html) · [scores page](https://5uck1ess.github.io/tts-bench/scores.html) · [capabilities page](https://5uck1ess.github.io/tts-bench/capabilities.html)
- [piper-tts on PyPI (1.8.0, GPL-3.0)](https://pypi.org/project/piper-tts/) · [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) · [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
- [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS) · [kitten-tts-nano-0.8-fp32 card](https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32) · [kittentts on PyPI](https://pypi.org/project/kittentts/)
- [F5-TTS HF API](https://huggingface.co/api/models/SWivid/F5-TTS) · [fishaudio/s1-mini HF API](https://huggingface.co/api/models/fishaudio/s1-mini) · [Spark-TTS-0.5B HF API](https://huggingface.co/api/models/SparkAudio/Spark-TTS-0.5B) · [ByteDance/MegaTTS3 HF API](https://huggingface.co/api/models/ByteDance/MegaTTS3) · [neuphonic/neutts-air HF API](https://huggingface.co/api/models/neuphonic/neutts-air)
- [zai-org/GLM-TTS HF API](https://huggingface.co/api/models/zai-org/GLM-TTS) · [FireRedTeam/FireRedTTS3 HF API](https://huggingface.co/api/models/FireRedTeam/FireRedTTS3) · [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) · [Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF](https://huggingface.co/Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF) · [Qwen/Qwen3-TTS-12Hz-1.7B-Base HF API](https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)

*Speed figures from tts-bench were measured on a Windows RTX 5090 32 GB, not on an RTX 4060 Laptop. Vendor figures (CosyVoice3 150 ms, VibeVoice ~300 ms, VoxCPM RTF 0.17, Sopro ~300 ms, Supertonic RTF tables, Gepard 25×) are vendor-published and marked as such throughout.*
