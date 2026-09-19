# Qwen3-TTS-12Hz inference: what is left beyond CUDA graphs

Research date: **2026-09-19**. Target deployment: `Qwen3-TTS-12Hz-0.6B/1.7B CustomVoice`, RTX 4060 **Laptop** 8 GB, Windows, torch 2.7.0+cu128, transformers 4.57.3, already on `andimarafioti/faster-qwen3-tts` (static KV cache + `torch.cuda.CUDAGraph` on Talker and the 5-layer Code Predictor).

Everything below is either quoted from a URL or explicitly flagged as *derived* / *unverified*. No number appears without a source.

---

## 0. Facts established first (these reframe the whole question)

**Your GPU's bandwidth is 256 GB/s.** Notebookcheck's spec table for the RTX 4060 Laptop: "Memory Bus Width 128 Bit / Memory Type GDDR6 / Memory Bandwidth 256 GB/s / Max. Amount of Memory 8 GB". ([notebookcheck](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-Laptop-GPU-Benchmarks-and-Specs.675692.0.html))

**Your numbers match the upstream author's own RTX 4060 (Windows) row.** faster-qwen3-tts README benchmarks: 0.6B RTF 2.26 / TTFA 413 ms; 1.7B RTF 1.83 / TTFA 460 ms on "RTX 4060 (Windows)". ([README](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/README.md)) Your 0.36–0.61 wall/audio RTF is the same operating point. Longer text → more codec tokens, so your 0.36–0.61 range is consistent with that single-point benchmark.

**Single-stream decode on a 270 GB/s-class GPU is not bandwidth-bound — it is kernel-launch-latency-bound.** The strongest evidence I found is a measurement on an A100 (1.5 TB/s, 5–6× the reference card) that came in at RTF 0.50–0.55 instead of the ~0.1 a bandwidth extrapolation predicts: *"past the point where weights stream fast enough, single-stream decode becomes kernel-launch-latency-bound (hundreds of small dependent launches per frame, costlier on virtualized cloud GPUs). Big-bandwidth cards pay off in batch throughput, not single-stream latency."* ([gabriele-mastrapasqua/qwen3-tts docs/cuda-performance.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/cuda-performance.md), also mirrored at [bon5co fork](https://github.com/bon5co/qwen3-tts/blob/main/docs/cuda-performance.md))

**The Code Predictor — not the Talker — dominates the frame, and its cost does not shrink with the Talker.** Measured per-frame profile (Apple Silicon, merged PR):

| Component | 0.6B (ms/frame) | 1.7B (ms/frame) | Ratio |
|---|---|---|---|
| Talker step | 23.6 | 92.2 | 3.9× |
| Code Predictor | 69.6 | 74.9 | ~same |
| Speech decoder | 46 | 56 | ~same |

*"CP is the bottleneck for both models (15 sequential passes per frame)."* ([PR #10](https://github.com/gabriele-mastrapasqua/qwen3-tts/pull/10))

That single table explains your observation that "the 0.6B is only 1.17–1.22× faster than the 1.7B despite being 3× smaller": the Talker scales 3.9× with size, the Predictor is flat, and the codec **embedding tables and head are shared** between the two sizes. From the official config: Talker `num_hidden_layers: 28`, `hidden_size: 1024`, `intermediate_size: 3072`, `num_attention_heads: 16`, `num_key_value_heads: 8`, `head_dim: 128`, `vocab_size: 3072`; Code Predictor `num_hidden_layers: 5`, same hidden/intermediate, `num_code_groups: 16`, `vocab_size: 2048`. ([config.json, 0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/raw/main/config.json))

*Derived (my arithmetic, not a source):* 15 sequential predictor passes × 5 layers = **75 layer-passes per frame vs the Talker's 28**, and 15 dependent kernel-launch chains vs 1 — so the predictor is the deeper *launch* chain even where it is the shallower *model*. A separate analysis says the same in words: *"On 0.6B the CP is ~74–90% of the frame (the real bottleneck)."* ([speculative-decoding-analysis.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/speculative-decoding-analysis.md))

**You are measuring the decoder inside the loop.** `faster_qwen3_tts/generate.py` times `t_decode` around the AR loop only, and the codec decode happens after it returns (`speech_tokenizer.decode(...)` in `model.py`). Your 31/40 ms-per-step figures therefore exclude codec decode. The codec decoder was 46–56 ms/frame in the Apple-silicon profile above — on your 4060 it is *outside* the RTF you quoted, so it is a separate (and on Windows, possibly CPU-side) cost to look for. ([generate.py](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/faster_qwen3_tts/generate.py), [model.py](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/faster_qwen3_tts/model.py))

---

## 1. Prioritised candidate table

Effort: S = <1 h, M = half day, L = multi-day. "Applies?" = works on CUDA-graph + 8 GB + Windows.

| # | Optimisation | What it changes | Verified speedup (source) | Effort | Risk | Applies? |
|---|---|---|---|---|---|---|
| 1 | **Fixed-batch decode: batch the Talker + Predictor graphs at B ≥ 2** | Converts 15×B sequential matvecs into batched matmuls; each weight row read once for all B lanes; amortises every per-kernel cost over B frames | Per-step Talker+CP **3.35× at B=8** (Talker 4.1×, CP **2.7×**); end-to-end server **~3× at B=8**. ([cuda-performance.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/cuda-performance.md)) Independently: 1.7B, batch=3 paragraphs, wall 55.82 s → **19.86 s (2.81×)**, peak VRAM 4.64 → **5.99 GB**. ([issue #89 comment](https://github.com/QwenLM/Qwen3-TTS/issues/89)) | **L** | Graph is captured at fixed shape → needs B baked in at capture; faster-qwen3-tts has **no** batching today | ✅ 8 GB holds B=3–4 at 1.7B (5.99 GB measured); B=8 at 0.6B plausible |
| 2 | **Reduce the static cache `max_seq_len` (2048 → ~1152)** | Cuts KV traffic per step and the per-step mask copy; frees ~120 MB | **No published ms/step-vs-cache-length curve found.** *Derived:* 28 layers × 8 KV heads × 128 dim × 2 × 2 B = 114 KB per token; ~2048 tokens ≈ **234 MB** read/step-equivalent ≈ **0.9 ms at 256 GB/s** — i.e. **<3 % of your 31 ms**. | **S** | Long inputs raise `RuntimeError: Input is too long`; 200-word reply needs ~960 frames | ✅ trivial, low upside |
| 3 | **Measure Talker vs Predictor on *your* GPU, then fuse the predictor loop instead of guessing** | Tells you where the 31 ms actually goes; the only route to a *single-stream* win is fewer dependent launches per frame | Method, not a result. Precedent for the win: NVIDIA's vLLM-Omni PR captures *"the talker and code predictor as a single full CUDA graph"* with *"fewer graph launches and a single replay on decode."* ([PR #3221](https://github.com/vllm-project/vllm-omni/pull/3221)) | **M** | Low (measurement) | ✅ |
| 4 | **Windows → WSL2 move (evidence is mixed, test it)** | Changes driver/runtime stack under identical hardware | Faster-qwen3-tts maintainer: *"Windows performance should be as good as linux"* ([issue #92](https://github.com/andimarafioti/faster-qwen3-tts/issues/92)). Contradicting community data: same user got **3–5 s/sentence native Windows** → **187 ms TTFA / RTF 4.546** under WSL2 on an RTX 5090 ([issue #92](https://github.com/andimarafioti/faster-qwen3-tts/issues/92)); another 5090 **Laptop** on native Windows got RTF 3.542/3.606 ([issue #97](https://github.com/andimarafioti/faster-qwen3-tts/issues/97)) | **M** | Dual-boot/WSL setup cost | ⚠️ Unresolved — the "Windows is slow" data point is about the *stock* `qwen-tts` package, not the CUDA-graph path |
| 5 | **INT8 / Q4_K_M weight quantisation (GGUF via `qwentts.cpp`)** | Halves/quarters weight bytes read per step | C engine on a **~270 GB/s RTX-4060-class GPU, 1.7B**: resident fused **int8 RTF 0.55**, **int4-Talker+int8-CP RTF 0.44**; dp4a cut **1.7B Talker 8.4 → 5.6 ms/frame (−33 %)**, 0.6B Talker −19 % / CP −16 %. ([cuda-performance.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/cuda-performance.md)) Q8_0/Q4_K_M files exist for 0.6B and 1.7B talkers + shared tokenizer. ([model card](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF)) | **L** (change runtime) | Different runtime, loses Python/streaming API; `faster-qwen3-tts` GGML adapter has ABI gaps | ⚠️ `qwentts-cpp-python` wheels are **Linux only** for the adapter path |
| 6 | **NVFP4 int4 checkpoint** | Weight-only FP4 on the 28 decoder layers | Author-reported on DGX Spark: RTF p50 0.4646 → **0.3224 (−30.6 %)**, first-audio latency −24.8 %, decoder weight memory −52.8 %, no median CER change. ([UrocyonF/Qwen3-TTS-12Hz-1.7B-NVFP4](https://huggingface.co/UrocyonF/Qwen3-TTS-12Hz-1.7B-NVFP4)) | **L** | **NVFP4 is Blackwell-only.** LLM Compressor: *"NVFP4 … introduced with the NVIDIA Blackwell GPU architecture … All NVIDIA Blackwell GPUs or later."* ([llm-compressor compression schemes](https://docs.vllm.ai/projects/llm-compressor/en/0.9.0.2/guides/compression_schemes/)) vLLM's hardware matrix marks FP8 W8A8 ✅ for **Ada** and does **not** list NVFP4. ([vLLM supported hardware](https://docs.vllm.ai/en/v0.7.1/features/quantization/supported_hardware.html)) | ❌ **No** — your 4060 is Ada (sm_89) |
| 7 | **`andimarafioti/faster-qwen3-tts` perf stack #108–#112** | Removes CPU-side launches/syncs in the sampling hot path | Author's own cumulative table (GB10, 0.6B, 92-step utterance): `main` 35.86 ms/step → stack **35.84 ms/step**; TTFA 421.1 ms → **414.4 ms**. Every PR is **within noise of zero** on ms/step. ([PR #112 body](https://github.com/andimarafioti/faster-qwen3-tts/pull/112)) | **S** | Stacked branches; head branch `perf/fused-codebook-embeddings` exists, all 5 PRs **open**, **0 reviews, 0 comments** | ⚠️ Realistically **~0 %** on a 4060 (see §3) |
| 8 | **`flash_attention_2`** | Attention kernel choice | *"Attention backends (eager, SDPA, Flash Attention 2): all identical RTF. Attention is not the bottleneck."* ([BLOG.md](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/BLOG.md)) Community thread: FA2 vs no-FA2 → RTF 0.35 vs 0.36 streaming, 0.27 vs 0.27 non-streaming; maintainer called it *"about 3% (probably measurement noise)"*. ([issue #89](https://github.com/QwenLM/Qwen3-TTS/issues/89)) | **S** | None | ❌ Not worth it |
| 9 | **`torch.compile`** | Inductor fusion | *"torch.compile: we patched three Triton incompatibilities to get it working on Jetson for the first time. Zero speedup — dynamic KV-cache shapes defeat the compiler."* ([BLOG.md](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/BLOG.md)) Its one documented win is **on top of** a static-shape setup: vLLM-Omni applies `torch.compile(mode="default", dynamic=True)` to the re-prefill predictor and reports RTF **0.234 → 0.124 at c=1** with CUDA graphs. ([vllm-omni perf doc](https://docs.vllm.ai/projects/vllm-omni/en/stable/design/qwen3_omni_tts_performance_optimization/)) | **M** | Compile time; already partly covered by your graph | ⚠️ Only inside a capture-friendly re-prefill design |
| 10 | **vLLM-Omni + cross-request batching (whole-stack switch)** | Moves to continuous batching + engine CUDA graphs + chunked streaming | H20, CustomVoice Vivian, **c=8**, streaming PCM: TTFT/TTFA median **878.5 → 130.6 ms**, RTF **0.389 → 0.2565**. ([PR #3485, merged](https://github.com/vllm-project/vllm-omni/pull/3485)) H200, c=1: RTF 0.16. ([perf doc](https://docs.vllm.ai/projects/vllm-omni/en/stable/design/qwen3_omni_tts_performance_optimization/)) | **L** | Windows + 8 GB serving stack; heavy dependency surface | ⚠️ Different stack, not a patch |

---

## 2. Top-3 actionable items — exact steps

### A. Batch the two CUDA graphs (the only lever with a documented >2× number)

This is the one item with reproduced, multi-source evidence of a ~3× per-step and ~2.8–3× end-to-end gain, and faster-qwen3-tts has **no** batching: *"Batch generation … Not available, I focused on local voice agents where single stream of generation is the goal."* — maintainer ([issue #80](https://github.com/andimarafioti/faster-qwen3-tts/issues/80)). The same maintainer, on concurrency: *"Weird things happen to cuda graphs when you try to run things concurrently … Best approach would be to support batching."* ([issue #85](https://github.com/andimarafioti/faster-qwen3-tts/issues/85))

Evidence for feasibility at your scale — measured, not estimated:
- 1.7B, batch=3 paragraphs, 8 GB-class card: peak GPU memory **4.64 GB → 5.99 GB** (**+1.35 GB, ~+29 %**). ([issue #89](https://github.com/QwenLM/Qwen3-TTS/issues/89))
- C engine, 1.7B, B=8: **~3× end-to-end**, per-step **3.35×** (Talker 4.1×, CP 2.7×). ([cuda-performance.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/cuda-performance.md))
- vLLM-Omni: `max_num_seqs: 1 → 10` alone gave **2–3× TTFA at c=4/8**, and batching alone took RTF **2.19 → 0.29 at c=10**. ([PR #3485](https://github.com/vllm-project/vllm-omni/pull/3485), [perf doc](https://docs.vllm.ai/projects/vllm-omni/en/stable/design/qwen3_omni_tts_performance_optimization/))

Concrete steps:
1. **Pick the batch size from VRAM, not from hope.** Measure `torch.cuda.max_memory_allocated()` after warmup at B=1 on 0.6B, then add ~1.35 GB per +2 lanes (the measured slope above) and cap at ~7.2 GB leaving headroom for the codec decode. Start at B=2 or 4 on 0.6B.
2. **Capture at the exact B you will run.** `TalkerGraph` allocates `input_buf = torch.zeros(1, 1, hidden)`, `position_ids = torch.zeros(3, 1, 1)`, and `StaticCache(config=..., max_cache_len=...)`; `PredictorGraph` allocates `input_buf = torch.zeros(1, 2, hidden)` and `output_tokens = torch.zeros(15)`. Every one of those leading `1`s must become `B` at construction, and the graph captured once per B — a graph replayed at a shape it was not captured for will silently misbehave. ([talker_graph.py](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/faster_qwen3_tts/talker_graph.py), [predictor_graph.py](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/faster_qwen3_tts/predictor_graph.py))
3. **Right-pad all B sequences to one length and mask them.** `TalkerGraph.set_generation_state` already builds a padding-aware mask from `attention_mask` and keys the mask table cache on `tuple(pad_counts.tolist())` — it will rebuild the whole 2048-entry table when the pattern changes, so pin one stable pad pattern (pad to a fixed bucket) instead of letting it vary per request.
4. **Stop lanes at EOS independently.** The decode loop breaks on `token.item() == eos_id` (a sync). With B lanes you need per-lane done-flags and a "keep stepping until all lanes done" loop, writing into the preallocated history buffers (which is exactly what PR #111 builds — see §3, it is a prerequisite for static shapes here).
5. **Accept that per-request latency gets worse while aggregate throughput gets better.** Measured: B=8 on a 270 GB/s card gave aggregate RTF 0.47 but **per-request RTF 1.27** ([cuda-performance.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/cuda-performance.md)). Batching is the right answer for an audiobook/long-form workload and the wrong answer for a single interactive stream.

### B. Measure the Talker/Predictor split on your own GPU before optimising further

Everything in the public record about "which half dominates" is from Apple Silicon or Jetson. Port it, cheaply:

1. Wrap each replay with `torch.cuda.Event(enable_timing=True)` pairs around `predictor_graph.run(...)` and `talker_graph.run(...)` inside the `fast_generate` loop, accumulate over ≥200 steps, report medians. Do **not** use `torch.cuda.synchronize()` per step — that is the exact sync PR #110 exists to remove and it will distort the measurement.
2. The decision rule: if Predictor ≥ ~60 % of the step (as the Apple-silicon profile found at 0.6B: 69.6 vs 23.6 ms/frame), then the Talker's 28 layers are not your problem and **no Talker-side quantisation or FA2 tune can give you 2×**.
3. If the Predictor dominates, the first structural fix is not a rewrite: check whether `StaticCache.reset()` is still called per replay. PR #109 removes ~10 zero-fill launches per step (`5 layers × K/V`) and is a **3-line** change; `PredictorGraph.run` currently ends with `self.static_cache.reset()` before `self.graph.replay()`. ([PR #109](https://github.com/andimarafioti/faster-qwen3-tts/pull/109), [predictor_graph.py](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/faster_qwen3_tts/predictor_graph.py))
4. Adopt the design that removes launches: NVIDIA captures talker+code-predictor **as a single full CUDA graph** on decode-only batches, with the rationale "fewer graph launches and a single replay on decode". ([vllm-omni PR #3221](https://github.com/vllm-project/vllm-omni/pull/3221)) That is the shape of change that actually reduces a launch-latency-bound step.

### C. Cheap experiments that take an afternoon and close open questions

1. **Cache length sweep.** Construct `FasterQwen3TTS.from_pretrained(..., max_seq_len=N)` for N ∈ {512, 768, 1024, 1152, 2048} on a fixed ~75 s script, record `ms/step`. This either confirms the derived <3 % estimate or falsifies it in an afternoon. (The constructor exposes `max_seq_len`, default 2048, and the talker raises `RuntimeError: Input is too long` past it. ([model.py](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/faster_qwen3_tts/model.py), [talker_graph.py](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/faster_qwen3_tts/talker_graph.py)))
2. **Check the mask's effect on SDPA kernel selection.** Compare `ms/step` against a build where, for the single-query decode, the explicit `[1,1,1,max_seq_len]` additive mask is replaced by a mask-free causal call. The project's own parity notes say static-cache masking *"often selects a different SDPA kernel (masked attention)"* than the mask-free dynamic path. ([README, Parity](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/README.md)) If the masked path is materially slower, that is a real, otherwise-invisible per-step cost.
3. **Move codec decode off the critical path / off the CPU.** The codec decoder was 46–56 ms/frame in the Apple profile; the C engine reports that leaving the conv decoder on the host was *"the difference between RTF 0.94 and 0.39"* in one cloud run. ([PR #10](https://github.com/gabriele-mastrapasqua/qwen3-tts/pull/10), [cuda-performance.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/cuda-performance.md)) Confirm whether your `speech_tokenizer.decode` is running on GPU or CPU, and how much wall time it adds outside `ms_per_step`.
4. **Chunk-size is a free TTFA dial.** Jetson 0.6B: chunk_size 4 → TTFA 362 ms, 8 → 556 ms, 12 → 753 ms; measured on your class of card, 5090 Laptop native Windows: **4 → 167 ms, 8 → 237 ms, 12 → 314 ms**. ([README](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/README.md), [issue #97](https://github.com/andimarafioti/faster-qwen3-tts/issues/97)) Going 8 → 4 buys ~70 ms of TTFA for a small RTF cost.

---

## 3. Already exhausted / not worth it

**The faster-qwen3-tts perf PR series #108–#112 will not help you.** All five are **open**, **unreviewed** (0 reviews, 0 comments each as of 2026-09-19), stacked, and their head branch `perf/fused-codebook-embeddings` exists. The author's own cumulative table on GB10 is the honest verdict:

| Stack level | ms/step | TTFA (chunk=8) |
|---|---|---|
| `main` | 35.86 | 421.1 ms |
| + #108 suppress-mask vectorisation | 35.91 | 415.7 ms |
| + #109 predictor reset removal | 35.68 | 412.6 ms |
| + #110 deferred EOS check | 35.99 | 416.6 ms |
| + #111 rep-penalty history buffer | 35.93 | 411.9 ms |
| + #112 fused codebook embeddings | **35.84** | **414.4 ms** |

([PR #112](https://github.com/andimarafioti/faster-qwen3-tts/pull/112))

Individual verdicts:
- **#108** vectorises the `suppress_mask` build (~1024 indexed writes → 2 slice ops) and precomputes the EOS-suppress index. Author: *"TTFA improves ~5 ms/call from removing the ~1024 launches; per-step cost is unchanged as expected."* ([#108](https://github.com/andimarafioti/faster-qwen3-tts/pull/108)) It is a per-request, not per-step, win. Harmless.
- **#109** removes ~10 zero-fill launches/step. 35.91 → 35.68 ms (**−0.6 %**). Safety argument given: *"Every replay of the captured graph rewrites all attended KV positions before reading them."* ([#109](https://github.com/andimarafioti/faster-qwen3-tts/pull/109)) **Cheapest correct change in the stack**; take it alone if you take anything.
- **#110 deferred EOS check** — the author *recommends against it on your hardware*: *"GB10 is the wrong machine to judge this PR … The target is launch-bound hosts … On Jetson, where ~16 ms of a 54 ms step is CPU glue, this is the change that lets the CPU launch step k+1 while the GPU still executes step k. Recommendation: benchmark on the Jetson before merging — if it doesn't deliver there, drop this PR."* Its own GB10 measurement is **35.68 → 35.99 ms (worse)**, attributed to *"~1 discarded overshoot step per utterance (+1/92 ≈ +0.4 %)"*. ([#110](https://github.com/andimarafioti/faster-qwen3-tts/pull/110)) You are on a GPU-bound 4060 with the same profile as GB10 → **skip it**.
- **#111** preallocated rep-penalty history replaces `torch.stack` over a growing Python list (O(n) launches/step, O(n²) per generation). Measured 35.99 → 35.93. ([#111](https://github.com/andimarafioti/faster-qwen3-tts/pull/111)) **But it is a prerequisite for your batching work** (constant shapes), which is the real reason to take it.
- **#112** fuses 15 per-codebook embedding gathers + a `cat` into one `F.embedding`. Measured 35.93 → 35.84. Note the stated VRAM cost: *"costs one duplicated copy of the tables in VRAM"* — with 15 tables × 2048 × 1024 at bf16 that is ~63 MB, ~126 MB if both the original and fused copies persist. ([#112](https://github.com/andimarafioti/faster-qwen3-tts/pull/112)) On 8 GB that is a conscious trade for a ~0.25 % win.

**Also already covered / ruled out:**
- **Flash Attention 2** — see §1 row 8. Three independent statements that it does not change RTF here.

  *Correction worth making:* a claim circulates that FA2 gives "+15 %". It was retracted in-thread by the person who posted it — *"That was the original Qwen3-TTS project doing a clone and creating a voice file based on my data, sorry that wasn't clear."* — i.e. it referred to the **stock** repo, not a compiled/grabbed path, and the maintainer of the compared fork replied *"in the compiled path, flash-attn vs eager makes no real difference."* ([issue #89](https://github.com/QwenLM/Qwen3-TTS/issues/89))
- **`torch.compile` on the Talker** — zero speedup on dynamic KV shapes, per the project's own "what we tried first" list; already superseded by your CUDA graphs. ([BLOG.md](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/BLOG.md))
- **Custom fused CUDA kernels (RMSNorm, SiLU)** — *"fused RMSNorm 8.4× faster, fused SiLU 2.2×: only 1.25× end-to-end. These ops are ~4 % of compute."* ([BLOG.md](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/BLOG.md)) This is the clearest statement that elementwise-fusion work is a dead end here.
- **`torch.set_float32_matmul_precision('high')`** — mentioned positively in the community thread alongside compile+graphs, but no isolated measurement was ever posted. Treat as ~free to try, unproven. ([issue #89](https://github.com/QwenLM/Qwen3-TTS/issues/89))
- **nano-qwen3tts-vllm** — 3.25× faster than stock on H100, 3.85× on L4 (RTF 0.399 / 0.742). ([issue #89](https://github.com/QwenLM/Qwen3-TTS/issues/89)) Those are *versus stock*, and you are already past that point; the port's KV-cache block allocator also *"breaks on Jetson's unified memory"*. Not an incremental win for you.
- **Reading the `codec_head` (3072→1024) or embedding lookups as the bottleneck** — the fused-embedding PR moved ms/step by 0.09. Empirically closed.
- **`repetition_penalty` vectorisation** — already in `main` (`apply_repetition_penalty` is called with a stacked history; the README documents the vectorised form). ([generate.py](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/faster_qwen3_tts/generate.py), [BLOG.md](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/BLOG.md))
- **Concurrency on one instance** — not just unsupported, actively harmful: *"Weird things happen to cuda graphs when you try to run things concurrently."* ([issue #85](https://github.com/andimarafioti/faster-qwen3-tts/issues/85)) Queue serialisation is the current correct design. Batch (§2A) is the supported alternative.

---

## 4. Could not verify

- **NVFP4 on Ada.** No source states outright that NVFP4 *fails* on sm_89. What is verified: LLM Compressor describes it as *"introduced with the NVIDIA Blackwell GPU architecture … All NVIDIA Blackwell GPUs or later"*, and vLLM's hardware matrix omits it while marking FP8 W8A8 ✅ on Ada. ([llm-compressor](https://docs.vllm.ai/projects/llm-compressor/en/0.9.0.2/guides/compression_schemes/), [vLLM](https://docs.vllm.ai/en/v0.7.1/features/quantization/supported_hardware.html)) The UrocyonF card reports no GPU beyond *"an NVIDIA DGX Spark system"*. **Net: do not plan on NVFP4 for a 4060.**
- **GGUF / `qwentts.cpp` throughput on a 4060 specifically.** The model card publishes file sizes only (Q8_0 talker 1.7B **2.1 GB**, Q4_K_M **1.2 GB**; tokenizer Q8_0 **291 MB** / Q4_K_M **255 MB**) and claims 🥇 1st on TTFA in HF's Open TTS Leaderboard *"as of September 9, 2026"* — but **no RTF or ms/step figure on the card or in the repo README**. ([Serveurperso/Qwen3-TTS-GGUF](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF), [qwentts.cpp README](https://github.com/ServeurpersoCom/qwentts.cpp/blob/master/README.md))
- **Is `bon5co/qwen3-tts` reliable?** It is a **fork** of `gabriele-mastrapasqua/qwen3-tts` (0 stars, created 2026-07-11, pushed 2026-07-11), and its `docs/cuda-performance.md` content is also present in the parent, whose README links that same page. ([repo API](https://api.github.com/repos/bon5co/qwen3-tts), [parent docs](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/cuda-performance.md)) I cite the parent. Treat the C-engine RTF/quant numbers as author-reported and ear-validated, not independently reproduced.
- **`elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX`.** An ONNX export exists with a split graph (`talker_prefill.onnx`, `talker_decode.onnx`, `code_predictor.onnx`, `vocoder.onnx`, `speaker_encoder.onnx`). ([HF API](https://huggingface.co/api/models/elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX)) I found **no benchmark, no `onnxruntime` session-option guidance, no Windows/CUDA-EP report, and no maintainer**. Not actionable as-is.
- **Is `qwen3-tts-1.7b-*` the community's "faster" fork?** No. `gabriele-mastrapasqua/qwen3-tts` is a pure-C engine with `--int8` / `--int4` (see §1 row 5), but I could not confirm any reported Windows-8 GB installation of it.
- **`docs/perf-analysis-2026-07.md`** returned no body from the API (it is a file, not a PR). Its contents are unread.
- **A published ms/step-vs-`max_seq_len` curve.** Does not appear to exist anywhere. My 0.9 ms estimate is *derived* arithmetic from the verified config (28 layers, 8 KV heads, head_dim 128, bf16) and the verified 256 GB/s — not a measurement.
- **`Qwen3-TTS` issue #89's raw 4090/5090 tables** are user-posted screenshots in several cases; I quote only the in-thread text tables that reproduce as text.

---

## 5. Final assessment: is another 2× available?

**For a single interactive stream on this laptop: no — you are close to the ceiling, and the ceiling is set by launch latency, not by your GPU's FLOPs or bytes.**

Reasoning, from verified numbers only:

1. Your per-step cost is not explained by bandwidth. At 256 GB/s, the Talker's ~0.5 G parameters at bf16 are ~1 GB → ~4 ms, and the KV traffic at 2048 tokens is ~0.9 ms. You measure **31 ms**. The gap is 5–8×, and the same gap appears independently on an A100 at 1.5 TB/s (measured RTF 0.50–0.55 where bandwidth predicted ~0.1). The documented explanation is *"hundreds of small dependent launches per frame"* ([cuda-performance.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/cuda-performance.md)). CUDA graphs already removed the *CPU* half of that cost; the *GPU-side* per-node cost of ~500 dependent nodes per step is what remains, and no amount of quantisation, Flash Attention, `torch.compile`, or kernel micro-fusion removes it — the project's own data shows a 8.4× faster RMSNorm kernel bought 1.25× end-to-end because "these ops are ~4 % of compute" ([BLOG.md](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/BLOG.md)).

2. The route to a genuine single-stream 2× is to **cut the number of dependent nodes per frame**, and the only published design that does it is capturing Talker + code predictor as **one** graph (NVIDIA, [PR #3221](https://github.com/vllm-project/vllm-omni/pull/3221)) rather than your current two graphs replayed 15 times. That is speculative for your codebase and the gain is not published as a number.

3. An anchor for "how good could a Python implementation get on this GPU class": the pure-C engine on a **~270 GB/s RTX-4060-class card** reports **RTF 0.44 for 1.7B** with int4-Talker + int8-CP + CUDA graphs + dp4a. ([cuda-performance.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/cuda-performance.md), measured point in bold on the RTX 3050/4060 row) Taking the README's own 1.7B RTX 4060 row of RTF 1.83 as your comparable, that is a **~4× gap that is entirely runtime engineering** (C + GGUF quants + dp4a + fused Talker/CP), not GPU headroom. It is achievable, but only by leaving the Python/CUDA-graph stack — and that engine's Windows story for the Python adapter is unverified.

4. **For throughput, 2× is comfortably available.** Batching is measured at **2.81× end-to-end wall time** at B=3 with **+1.35 GB VRAM**, **~3× at B=8** in the C engine, and vLLM-Omni shows `max_num_seqs: 1 → 10` alone giving **2–3× TTFA at c=4/8** and RTF **2.19 → 0.29 at c=10**. If your workload is long-form/audiobook, batching is the answer and you should stop looking for per-step tricks.

**The measurement that would settle it** (≈1 hour, no code changes beyond instrumentation):

> Instrument `faster_qwen3_tts/generate.py` with CUDA events around `predictor_graph.run(...)` and `talker_graph.run(...)` and accumulate medians over ≥200 steps on a fixed 200-word prompt, with no per-step `cuda.synchronize()`. Report the three numbers: Talker ms, Predictor ms, other ms.
>
> - If **Predictor ≥ 60 %** of the 31 ms → the Talker is not your bottleneck and no Talker-side change (quantisation, FA2, cache length, mask layout) can reach 2×. The remaining single-stream gains are: (a) merge the two graphs into one, (b) reduce predictor sequential depth. Both are research, not tuning.
> - If **Predictor ≤ 40 %** → the Talker's 28 layers dominate and `max_seq_len`/mask-layout/attention-kernel experiments move back onto the table; run the cache-length sweep in §2C.1 immediately.
> - In both cases, if the sum of the two is well *below* 31 ms, the remainder is sampling/glue, and PRs #108/#111 matter after all — that is the one outcome under which the open perf PRs are worth adopting.
