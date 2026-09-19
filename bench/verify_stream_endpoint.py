"""验证 /api/v1/tts/synthesize-stream：首片延迟、PCM 正确性、响应头、边界情况

不启动整套服务（server.py 会加载 ASR/OCR，没必要），只挂载 TTS router，
用 FastAPI TestClient 真实走 HTTP 路径。

用法（memory-tts-bp 环境，项目根目录）:
    python bench/verify_stream_endpoint.py --model ./models/qwen-0.6b
"""
import argparse
import json
import os
import struct
import sys
import time

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba-cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
_BP = os.path.join(os.getcwd(), ".backport")
if os.path.isdir(_BP):
    sys.path.insert(0, _BP)
# 脚本在 bench/ 下，需把项目根目录加入 sys.path 才能 import src.*
sys.path.insert(0, os.getcwd())

import asyncio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

TEXT_SHORT = "Hello there, my friend."
TEXT_LONG = (
    "That's a really thoughtful question, and I'm glad you asked it. Let me walk through it step by step. "
    "First, it helps to understand what you're actually trying to achieve, because the best method depends "
    "entirely on your goal. If your aim is fluency, then daily practice matters far more than long study "
    "sessions once a week. Second, try to make the practice enjoyable, since you'll stick with something "
    "you like much longer than something that feels like a chore. You could watch short videos, read simple "
    "articles, or talk with a partner about topics you genuinely care about."
)


def build_app(model_path):
    from fastapi import FastAPI
    from src.tts.router import router as tts_router
    from src.tts.config import TTSConfig
    from src.tts.model_loader import TTSModelManager

    app = FastAPI()
    app.state.model_lock = asyncio.Lock()
    app.state.tts_config = TTSConfig()
    print(f"=== loading {model_path} ===", flush=True)
    t0 = time.perf_counter()
    app.state.model = TTSModelManager(config=app.state.tts_config)
    print(f"load={time.perf_counter()-t0:.1f}s backend={app.state.model.backend}", flush=True)
    app.state.stream_loop = None  # TestClient 内部启动后由 lifespan 设置
    app.include_router(tts_router)

    @app.on_event("startup")
    async def _startup():
        app.state.stream_loop = asyncio.get_running_loop()

    return app


def probe(url, voice, label, text):
    """对**真实运行中的 HTTP 服务**做流式探测（TestClient 会缓冲，不能用）。

    用裸 socket 手工解析 chunked 响应：本机 httpx/requests 受代理环境变量影响
    会解析 URL 失败，而我们只需要"首字节到达时间"，裸 socket 最可靠也最直接。
    """
    import socket
    from urllib.parse import urlparse

    u = urlparse(url)
    host, port = u.hostname, u.port or 80
    path = u.path + (("?" + u.query) if u.query else "")
    payload = json.dumps({"text": text, "voice": voice, "language": "English"}).encode()

    print(f"\n--- {label} ({len(text)} 字符) → {url} ---", flush=True)
    t0 = time.perf_counter()
    s = socket.create_connection((host, port), timeout=300)
    s.sendall(
        f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n".encode() + payload
    )

    # ── 读响应头 ──
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = s.recv(4096)
        if not d:
            break
        buf += d
    head, _, rest = buf.partition(b"\r\n\r\n")
    lines = head.decode("latin1").split("\r\n")
    status = lines[0]
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    print(f"  {status}", flush=True)
    print(f"  X-TTS-TTFA-Ms={headers.get('x-tts-ttfa-ms')} "
          f"X-Audio-Sample-Rate={headers.get('x-audio-sample-rate')} "
          f"Transfer-Encoding={headers.get('transfer-encoding')} "
          f"Content-Type={headers.get('content-type')}", flush=True)

    sr = int(headers.get("x-audio-sample-rate", "24000"))
    chunked = "chunked" in headers.get("transfer-encoding", "").lower()

    # ── 流式读 body ──
    first_at = None
    total = 0
    nchunk = 0
    pending = rest
    body = bytearray()

    def feed(data):
        nonlocal first_at, total, nchunk, pending
        if not data:
            return
        if first_at is None:
            first_at = time.perf_counter() - t0
        nchunk += 1
        total += len(data)
        body.extend(data)

    if chunked:
        # 解析 chunked 编码，逐个 chunk 记录到达时间
        while True:
            while b"\r\n" not in pending:
                d = s.recv(65536)
                if not d:
                    pending = b""
                    break
                pending += d
            if not pending:
                break
            line, _, pending = pending.partition(b"\r\n")
            try:
                size = int(line.split(b";")[0], 16)
            except ValueError:
                break
            if size == 0:
                break
            while len(pending) < size + 2:
                d = s.recv(65536)
                if not d:
                    break
                pending += d
            feed(pending[:size])
            pending = pending[size + 2:]
    else:
        feed(pending)
        while True:
            d = s.recv(65536)
            if not d:
                break
            feed(d)
    s.close()

    elapsed = time.perf_counter() - t0
    dur = total / 2 / sr
    print(f"  ★ 首片到达={first_at*1000:.0f}ms  总耗时={elapsed:.2f}s  片数={nchunk}  "
          f"字节={total}  音频={dur:.2f}s  RTF_wall={elapsed/dur:.3f}x", flush=True)
    pcm = np.frombuffer(bytes(body[: len(body) // 2 * 2]), dtype="<i2")
    if len(pcm):
        print(f"  PCM 校验: samples={len(pcm)} min={pcm.min()} max={pcm.max()} "
              f"rms={np.sqrt(np.mean(pcm.astype(np.float64)**2)):.1f} "
              f"全零={bool(np.all(pcm == 0))}", flush=True)
    return first_at, nchunk, dur


def serve(model_path, port, backend, voice):
    """启动一个**最小**服务（只挂 TTS router，不加载 ASR/OCR）。"""
    os.environ["TTSCONF_BACKEND"] = backend
    os.environ["QWEN_TTS_MODEL_PATH"] = model_path
    import uvicorn
    app = build_app(model_path)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen-0.6b")
    ap.add_argument("--voice", default="aiden")
    ap.add_argument("--backend", default="faster", choices=["faster", "upstream"],
                    help="faster=CUDA Graph backport; upstream=原动态KV")
    ap.add_argument("--serve", action="store_true", help="只启动最小服务（供另一进程探测）")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--probe-url", default=None, help="对运行中的服务做流式探测")
    args = ap.parse_args()

    if args.serve:
        serve(args.model, args.port, args.backend, args.voice)
        return

    # 真实 HTTP 探测模式（验证真正的分块流式，不受 TestClient 缓冲影响）
    if args.probe_url:
        probe(args.probe_url, args.voice, "短句", TEXT_SHORT)
        probe(args.probe_url, args.voice, "长文本(约110词)", TEXT_LONG)
        return

    # 通过环境变量覆盖配置（比 TTSConfig 的 env 前缀约定）
    os.environ["TTSCONF_BACKEND"] = args.backend
    os.environ["QWEN_TTS_MODEL_PATH"] = args.model
    print(f"backend={args.backend} model={args.model}", flush=True)

    app = build_app(args.model)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        for label, text in [("短句", TEXT_SHORT), ("长文本(约110词)", TEXT_LONG)]:
            print(f"\n--- {label} ({len(text)} 字符) ---", flush=True)
            body = {"text": text, "voice": args.voice, "language": "English"}
            t0 = time.perf_counter()
            first_at = None
            total = 0
            nchunk = 0
            with client.stream("POST", "/api/v1/tts/synthesize-stream", json=body) as r:
                print(f"  HTTP {r.status_code}  Content-Type={r.headers.get('content-type')}", flush=True)
                print(f"  X-Audio-Sample-Rate={r.headers.get('x-audio-sample-rate')} "
                      f"X-Audio-Format={r.headers.get('x-audio-format')} "
                      f"X-TTS-TTFA-Ms={r.headers.get('x-tts-ttfa-ms')}", flush=True)
                sr = int(r.headers.get("x-audio-sample-rate", "24000"))
                buf = bytearray()
                for chunk in r.iter_bytes():
                    if first_at is None:
                        first_at = time.perf_counter() - t0
                    nchunk += 1
                    total += len(chunk)
                    buf.extend(chunk)
            elapsed = time.perf_counter() - t0
            dur = total / 2 / sr
            print(f"  首片到达={first_at*1000:.0f}ms  总耗时={elapsed:.2f}s  "
                  f"片数={nchunk}  字节={total}  音频={dur:.2f}s  RTF_wall={elapsed/dur:.3f}x",
                  flush=True)

            # PCM 正确性：int16 范围内、非全零、无削波爆表
            pcm = np.frombuffer(bytes(buf), dtype="<i2")
            print(f"  PCM 校验: samples={len(pcm)} min={pcm.min()} max={pcm.max()} "
                  f"rms={np.sqrt(np.mean(pcm.astype(np.float64)**2)):.1f} "
                  f"全零={bool(np.all(pcm == 0))} "
                  f"触顶(|x|>=32767)={int((np.abs(pcm) >= 32767).sum())}", flush=True)

            # 与整段接口对比：时长应接近
            t1 = time.perf_counter()
            with client.stream("POST", "/api/v1/tts/synthesize", json=body) as r2:
                wav = b"".join(r2.iter_bytes())
            t_full = time.perf_counter() - t1
            if wav[:4] == b"RIFF":
                import io
                import soundfile as sf
                d, sr2 = sf.read(io.BytesIO(wav))
                print(f"  [对照] 整段接口: 首字节需等 {t_full:.2f}s, 音频 {len(d)/sr2:.2f}s "
                      f"(流式首片仅 {first_at*1000:.0f}ms)", flush=True)

        # 边界：空文本
        print("\n--- 边界: 空文本 ---", flush=True)
        r = client.post("/api/v1/tts/synthesize-stream", json={"text": "   "})
        print(f"  HTTP {r.status_code} (期望 400) body={r.text[:80]!r}", flush=True)


if __name__ == "__main__":
    main()
