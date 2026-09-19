"""探测：ASR 运行期间事件循环是否被"饿死"（解释 TTS 首片为何发不出去）

现象：服务端日志记 ttfa=400ms（说明 worker 很快产出了首片），
但客户端实测首字节要等 5.3s（正好等于 ASR 耗时）。
怀疑：ASR 在 threadpool 里跑阻塞推理，占住 GIL，事件循环拿不到时间片，
于是队列里的分片发不出去；等 ASR 结束、GIL 释放，才一次性发出去。

方法：并发期间**轮询一个极轻量的健康接口**（/api/v1/health），测其响应延迟。
  若健康接口也被拖到秒级 → 事件循环确实被饿死（GIL 抢占）
  若健康接口始终毫秒级 → 事件循环正常，问题在别处

用法:
    python bench/measure_eventloop_starvation.py --base http://127.0.0.1:8000
"""
import argparse
import json
import socket
import struct
import threading
import time
from urllib.parse import urlparse


def simple_get(base, path, timeout=30):
    u = urlparse(base + path)
    t0 = time.perf_counter()
    s = socket.create_connection((u.hostname, u.port), timeout=timeout)
    s.sendall(f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nConnection: close\r\n\r\n".encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = s.recv(4096)
        if not d:
            break
        buf += d
    dt = (time.perf_counter() - t0) * 1000
    s.close()
    return dt


def make_wav(seconds=30, sr=16000):
    import math
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        f = 180 + 120 * math.sin(2 * math.pi * 0.7 * i / sr)
        frames += struct.pack("<h", int(12000 * math.sin(2 * math.pi * f * i / sr)))
    data = bytes(frames)
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
    hdr += struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    hdr += b"data" + struct.pack("<I", len(data))
    return hdr + data


def start_asr(base, wav, out):
    boundary = "----benchboundary"
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n".encode(),
        wav,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    u = urlparse(base + "/api/v1/asr/transcribe")
    t0 = time.perf_counter()
    s = socket.create_connection((u.hostname, u.port), timeout=600)
    s.sendall(
        f"POST {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
        f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
    )
    try:
        while s.recv(65536):
            pass
    except Exception:
        pass
    s.close()
    out["asr_s"] = time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--asr-seconds", type=float, default=30.0)
    args = ap.parse_args()

    print("=== 事件循环饿死探测 ===\n")
    print("【基线】空闲时健康接口延迟（5 次）")
    idle = [simple_get(args.base, "/api/v1/health") for _ in range(5)]
    print(f"    空闲: {[f'{x:.0f}ms' for x in idle]}  最大={max(idle):.0f}ms\n")

    print(f"【并发】起 ASR（{args.asr_seconds}s 音频），期间每 300ms 探一次健康接口")
    wav = make_wav(args.asr_seconds)
    out = {}
    th = threading.Thread(target=start_asr, args=(args.base, wav, out), daemon=True)
    th.start()
    time.sleep(0.5)
    lags = []
    t_end = time.time() + 12
    while time.time() < t_end and th.is_alive():
        lags.append(simple_get(args.base, "/api/v1/health"))
        time.sleep(0.3)
    th.join(timeout=120)
    print(f"    ASR 用时: {out.get('asr_s', -1):.2f}s")
    if lags:
        print(f"    并发期健康接口: {[f'{x:.0f}ms' for x in lags]}")
        print(f"    最大={max(lags):.0f}ms  中位={sorted(lags)[len(lags)//2]:.0f}ms")
    print()
    print("=== 判定 ===")
    if lags and max(lags) > 1000:
        print(f"  ❌ 事件循环被明显拖慢（最大 {max(lags):.0f}ms vs 空闲 {max(idle):.0f}ms）")
        print("     → ASR 占住 GIL，事件循环无法及时发送 TTS 分片")
    else:
        print(f"  ✅ 事件循环未被拖慢（最大 {max(lags) if lags else -1:.0f}ms）")


if __name__ == "__main__":
    main()
