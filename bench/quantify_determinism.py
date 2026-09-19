"""量化"同文本两次生成"的实际差异（因为 Predictor 采样被捕获进 CUDA Graph，
无法注入 generator，需要先知道残差有多大再决定是否值得改架构）

对比维度：
  - 音频时长、字节数
  - 波形相关性 / 最大逐样本偏差 / RMS 差
  - 用 ASR 回读确认内容是否一致
若差异只是数值噪声（相关性 >0.99、听感一致），就不必为它重构图捕获；
若差异明显（相关性低），则说明"重复点击会听到明显不同的朗读"。

用法（服务已启动）:
    python bench/quantify_determinism.py --url http://127.0.0.1:8000/api/v1/tts/synthesize-stream
"""
import argparse
import json
import socket
from urllib.parse import urlparse

import numpy as np

TEXT = "That is a really thoughtful question, and I am glad you asked it."


def fetch_pcm(url, text, timeout=300.0):
    u = urlparse(url)
    host, port = u.hostname, u.port or 80
    payload = json.dumps({"text": text}).encode()
    s = socket.create_connection((host, port), timeout=timeout)
    s.sendall(
        f"POST {u.path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n".encode() + payload
    )
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = s.recv(4096)
        if not d:
            break
        buf += d
    head, _, pending = buf.partition(b"\r\n\r\n")
    headers = {}
    for ln in head.decode("latin1").split("\r\n")[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    chunked = "chunked" in headers.get("transfer-encoding", "").lower()
    body = bytearray()

    def feed(d):
        if d:
            body.extend(d)

    if chunked:
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
    return np.frombuffer(bytes(body), dtype="<i2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    print(f"=== 量化同文本重复生成的差异: {args.url}\n")
    runs = []
    for i in range(args.rounds):
        pcm = fetch_pcm(args.url, TEXT)
        runs.append(pcm)
        print(f"  run{i+1}: samples={len(pcm)}  时长={len(pcm)/24000:.3f}s")

    base = runs[0].astype(np.float64)
    print(f"\n{'对比':10s} {'长度一致':>8s} {'相关性':>9s} {'最大偏差':>9s} {'RMS差':>9s} {'dBFS差':>8s}")
    print("-" * 66)
    for i in range(1, len(runs)):
        other = runs[i].astype(np.float64)
        n = min(len(base), len(other))
        a, b = base[:n], other[:n]
        if a.std() > 0 and b.std() > 0:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = float("nan")
        maxdev = float(np.abs(a - b).max())
        rmsdiff = float(np.sqrt(np.mean((a - b) ** 2)))
        rms_a = float(np.sqrt(np.mean(a ** 2)))
        rms_b = float(np.sqrt(np.mean(b ** 2)))
        db = 20 * np.log10(rms_a / rms_b) if rms_b > 0 else float("nan")
        print(f"run1 vs {i+1} {str(len(base)==len(other)):>8s} {corr:9.5f} {maxdev:9.1f} "
              f"{rmsdiff:9.1f} {db:8.3f}")

    print("\n解读：")
    print("  相关性 >0.99 且 RMS 差远小于信号 RMS → 差异属数值噪声，听感基本一致")
    print("  相关性明显 <0.99 → 两次朗读确有可感知差异，需要处理")
    sig_rms = float(np.sqrt(np.mean(base ** 2)))
    print(f"  （参考：信号 RMS = {sig_rms:.0f}，int16 满量程 32767）")


if __name__ == "__main__":
    main()
