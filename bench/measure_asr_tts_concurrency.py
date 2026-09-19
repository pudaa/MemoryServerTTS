"""测量 ASR 与 TTS 是否互相阻塞（判断"一次只能处理单个任务"的真实范围）

背景：项目里 app.state.model_lock 只被 TTS 路径使用，ASR 没有加锁。
那么"任务排队"到底发生在哪？本脚本实测：

  [A] 单独发流式 TTS 请求 → 基准 TTFA
  [B] 并发：先起 ASR 转写（较长音频），立刻发流式 TTS 请求 → 看 TTS 的 TTFA
      若与基准接近 → ASR 与 TTS 互不影响（可并行）
      若明显变长 → 二者互相阻塞

用法（需服务已启动）:
    python bench/measure_asr_tts_concurrency.py --base http://127.0.0.1:8000
"""
import argparse
import json
import socket
import struct
import threading
import time
from urllib.parse import urlparse

PCM_TEXT = "Hello there, this is a concurrency test."


def http_post(base, path, body_bytes, content_type="application/json", timeout=600):
    u = urlparse(base + path)
    s = socket.create_connection((u.hostname, u.port), timeout=timeout)
    s.sendall(
        f"POST {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
        f"Content-Type: {content_type}\r\nContent-Length: {len(body_bytes)}\r\n"
        f"Connection: close\r\n\r\n".encode() + body_bytes
    )
    return s


def read_head(s):
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = s.recv(4096)
        if not d:
            break
        buf += d
    head, _, rest = buf.partition(b"\r\n\r\n")
    lines = head.decode("latin1").split("\r\n")
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return lines[0], headers, rest


def tts_ttfa(base, label):
    body = json.dumps({"text": PCM_TEXT}).encode()
    t0 = time.perf_counter()
    s = http_post(base, "/api/v1/tts/synthesize-stream", body)
    status, headers, _ = read_head(s)
    ttfa = (time.perf_counter() - t0) * 1000
    print(f"    [{time.strftime('%H:%M:%S')}] {label}: {status.split()[1]} TTFA={ttfa:.0f}ms", flush=True)
    try:
        while s.recv(65536):
            pass
    except Exception:
        pass
    s.close()
    return ttfa


def make_wav(seconds=8, sr=16000):
    """生成一段**有声**的 wav（正弦波上扫），确保 ASR 真的做推理而不是秒退。

    注意：一开始用静音生成，ASR 会因内容为空几乎立即返回（0.03s），
    测不出并发关系——必须给真实信号。
    """
    import math
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        # 变频正弦，避免被判定为纯静音/单音而被快速跳过
        f = 180 + 120 * math.sin(2 * math.pi * 0.7 * i / sr)
        v = int(12000 * math.sin(2 * math.pi * f * i / sr))
        frames += struct.pack("<h", v)
    data = bytes(frames)
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
    hdr += struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    hdr += b"data" + struct.pack("<I", len(data))
    return hdr + data


def asr_task(base, wav, results, idx):
    boundary = "----benchboundary"
    # 字段名必须是 audio（见 src/asr/router.py），不是 file
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
                 f"Content-Type: audio/wav\r\n\r\n".encode())
    parts.append(wav)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    t0 = time.perf_counter()
    try:
        s = http_post(base, "/api/v1/asr/transcribe", body,
                      content_type=f"multipart/form-data; boundary={boundary}")
        status, headers, rest = read_head(s)
        # 读完
        try:
            while s.recv(65536):
                pass
        except Exception:
            pass
        s.close()
        results[idx] = time.perf_counter() - t0
        print(f"    [{time.strftime('%H:%M:%S')}] ASR 完成: {status.split()[1]} 用时={results[idx]:.2f}s", flush=True)
    except Exception as e:
        results[idx] = -1
        print(f"    ASR 异常: {type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--asr-seconds", type=float, default=8.0)
    args = ap.parse_args()

    print("=== ASR 与 TTS 并发关系测量 ===\n")

    print("【0】预热 TTS")
    tts_ttfa(args.base, "warmup")
    print()

    print("【1】基准：单独 TTS（3 次）")
    base_ttfa = [tts_ttfa(args.base, f"baseline{i+1}") for i in range(3)]
    avg = sum(base_ttfa) / len(base_ttfa)
    print(f"    基准均值 = {avg:.0f}ms\n")

    print(f"【2】并发：先起 ASR（{args.asr_seconds}s 音频），再立刻发 TTS")
    wav = make_wav(args.asr_seconds)
    results = {}
    th = threading.Thread(target=asr_task, args=(args.base, wav, results, 0), daemon=True)
    th.start()
    time.sleep(0.3)          # 让 ASR 先进入推理
    conc_ttfa = tts_ttfa(args.base, "concurrent-TTS")
    th.join(timeout=300)
    print()

    print("=== 判定 ===")
    ratio = conc_ttfa / avg if avg else float("nan")
    print(f"  基准 TTS TTFA {avg:.0f}ms → ASR 并发时 {conc_ttfa:.0f}ms（{ratio:.2f}×）")
    if ratio < 1.5:
        print("  ✅ ASR 与 TTS **互不阻塞**（可并行）")
    else:
        print("  ❌ ASR 与 TTS 互相阻塞（同一资源排队）")
    print(f"  ASR 自身用时: {results.get(0, -1):.2f}s（用于对照是否被 TTS 拖慢）")


if __name__ == "__main__":
    main()
