"""验证流式端点确定性：同文本重复请求 → 音频应可复现；不同文本 → 应不同

对应需求："重复点击同一条回复朗读按钮"固定 seed，做到可复现、不浪费存储。

**关于判定口径（重要）**：
`faster_qwen3_tts` 的 Predictor 采样（15 个 codebook）是在 **CUDA Graph 内部**
执行的（`predictor_graph._full_loop` 被 `torch.cuda.graph()` 捕获），因此
**无法给它注入 torch.Generator**——图重放会复用捕获时的 RNG 状态。

实测结果：长度 100% 一致，波形相关性 **0.99999**，RMS 差约为信号的 0.3%
（即约 -70dB 量级的数值噪声），**听感等同一致**。
因此本脚本按"相关性 + 长度"判定，而**不**要求逐字节相同。

用法（先起服务，再探测）:
    python bench/verify_stream_determinism.py --url http://127.0.0.1:8000/api/v1/tts/synthesize-stream
"""
import argparse
import hashlib
import json
import socket
import time
from urllib.parse import urlparse

import numpy as np

TEXT_A = "That's a really thoughtful question, and I'm glad you asked it."
TEXT_B = "Let me walk through it step by step, starting with the basics."
TEXT_C = "That's a really thoughtful question, and I'm glad you asked it."  # 与 A 相同

# 判定阈值：相关性 >= 0.99 视为"听感等同"
CORR_THRESHOLD = 0.99


def fetch(url, text, timeout=300.0):
    """返回 (pcm_bytes, ttfa_ms, chunks)"""
    u = urlparse(url)
    host, port = u.hostname, u.port or 80
    path = u.path + (("?" + u.query) if u.query else "")
    payload = json.dumps({"text": text}).encode()

    t0 = time.perf_counter()
    s = socket.create_connection((host, port), timeout=timeout)
    s.sendall(
        f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
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
    first_at = None
    nchunk = 0

    def feed(d):
        nonlocal first_at, nchunk
        if not d:
            return
        if first_at is None:
            first_at = time.perf_counter() - t0
        nchunk += 1
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
    return bytes(body), (first_at or 0) * 1000, nchunk


def h(b):
    return hashlib.sha256(b).hexdigest()[:16]


def corr(a_bytes, b_bytes):
    """两段 PCM 的波形相关性（长度不同则取较短公共部分）"""
    a = np.frombuffer(a_bytes, dtype="<i2").astype(np.float64)
    b = np.frombuffer(b_bytes, dtype="<i2").astype(np.float64)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    a, b = a[:n], b[:n]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    print(f"=== 确定性验证: {args.url}")
    print(f"    判定口径: 长度一致 且 波形相关性 >= {CORR_THRESHOLD}\n")

    print("【1】同文本多次请求（应可复现）")
    pcms = []
    for i in range(args.rounds):
        pcm, ttfa, n = fetch(args.url, TEXT_A)
        pcms.append(pcm)
        print(f"   A 第{i+1}次: sha={h(pcm)}  bytes={len(pcm)}  首片={ttfa:.0f}ms  分片={n}")
    corrs = [corr(pcms[0], pcms[i]) for i in range(1, len(pcms))]
    same_len = all(len(p) == len(pcms[0]) for p in pcms)
    ok1 = same_len and all(c >= CORR_THRESHOLD for c in corrs)
    print(f"   长度一致={same_len}  相关性={[f'{c:.5f}' for c in corrs]}")
    print(f"   => {'✅ 可复现（听感等同）' if ok1 else '❌ 不一致'}\n")

    print("【2】不同文本（应不同，保证语调多样性）")
    pcm_b, _, _ = fetch(args.url, TEXT_B)
    c_ab = corr(pcms[0], pcm_b)
    print(f"   B: sha={h(pcm_b)}  bytes={len(pcm_b)}  与A相关性={c_ab:.5f}")
    ok2 = (h(pcm_b) != h(pcms[0])) and (c_ab < CORR_THRESHOLD)
    print(f"   => {'✅ 与 A 明显不同' if ok2 else '❌ 与 A 相同/过于接近'}\n")

    print("【3】再次同文本 A（跨其他请求后仍应稳定）")
    pcm_c, _, _ = fetch(args.url, TEXT_C)
    c_ac = corr(pcms[0], pcm_c)
    print(f"   A: sha={h(pcm_c)}  bytes={len(pcm_c)}  与首次相关性={c_ac:.5f}")
    ok3 = len(pcm_c) == len(pcms[0]) and c_ac >= CORR_THRESHOLD
    print(f"   => {'✅ 仍与首次一致' if ok3 else '❌ 被其他请求影响'}\n")

    print(f"【4】字节级完全一致？ {'是' if h(pcm_c)==h(pcms[0]) else '否（存在图内采样的数值噪声，见脚本文档）'}")

    ok = ok1 and ok2 and ok3
    print(f"\n=== 总体: {'通过' if ok else '未通过'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
