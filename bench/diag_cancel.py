"""精确诊断：取消后服务端到底卡在哪

用 curl 子进程模拟"读 1 秒就断开"的客户端，同时在 Python 侧持续探测
'下一个请求能否立刻拿到锁'，并把服务端日志按时间对齐。

用法:
    python bench/diag_cancel.py --base http://127.0.0.1:8000 --log bench/_srv.log
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time

SMALL = "Hi there, this is a short one."


def long_text(words=200):
    base = ("This is a long paragraph used to keep the speech engine busy for a while, "
            "so that we can measure what happens when the client disconnects early. ")
    s = ""
    while len(s.split()) < words:
        s += base
    return s.strip()


def quick_probe(base_url, label):
    """发一个小请求，返回 TTFA(ms)"""
    from urllib.parse import urlparse
    u = urlparse(base_url + "/api/v1/tts/synthesize-stream")
    payload = json.dumps({"text": SMALL}).encode()
    t0 = time.perf_counter()
    s = socket.create_connection((u.hostname, u.port), timeout=300)
    s.sendall(
        f"POST {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n".encode() + payload
    )
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = s.recv(4096)
        if not d:
            break
        buf += d
    ttfa = (time.perf_counter() - t0) * 1000
    try:
        while True:
            if not s.recv(65536):
                break
    except Exception:
        pass
    s.close()
    print(f"    [{time.strftime('%H:%M:%S')}] {label}: TTFA={ttfa:.0f}ms", flush=True)
    return ttfa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--log", default="bench/_srv.log")
    ap.add_argument("--read-seconds", type=float, default=1.0)
    args = ap.parse_args()

    # 记录日志起点，便于事后只取本次区间
    log_start = 0
    if os.path.exists(args.log):
        log_start = os.path.getsize(args.log)

    print("=== 诊断：取消后的服务端行为 ===\n")
    print("【1】基线")
    quick_probe(args.base, "baseline")

    print(f"\n【2】发长文本请求，读 {args.read_seconds}s 后断开")
    body = json.dumps({"text": long_text()})
    curl = subprocess.Popen(
        ["curl", "-s", "-N", "-X", "POST", args.base + "/api/v1/tts/synthesize-stream",
         "-H", "Content-Type: application/json", "-d", body],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    time.sleep(args.read_seconds)
    t_kill = time.time()
    curl.kill()          # 强杀，等价于客户端 RST 断开
    curl.wait(timeout=10)
    print(f"    [{time.strftime('%H:%M:%S')}] 已断开 curl")

    # 立即连续探测，看多久才恢复
    print("\n【3】断开后立刻连续探测（每 0.5s 一次）")
    recovered_at = None
    for i in range(30):
        t0 = time.time()
        ttfa = quick_probe(args.base, f"probe{i+1} (断开后 {t0 - t_kill:.1f}s)")
        if ttfa < 1500 and recovered_at is None:
            recovered_at = t0 - t_kill
        if recovered_at is not None and i >= 1:
            break
        time.sleep(0.5)

    print(f"\n=== 结论 ===")
    if recovered_at is None:
        print("  ❌ 30 次探测内未恢复")
    else:
        print(f"  ✅ 断开后 {recovered_at:.1f}s 恢复（TTFA 回到 1.5s 以内）")
        print(f"     即：取消后服务端还会占用约 {recovered_at:.1f}s 才释放")

    # 打印本次区间内的服务端日志
    print(f"\n=== 服务端日志（本次区间）===")
    if os.path.exists(args.log):
        with open(args.log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(log_start)
            for line in f:
                line = line.rstrip()
                if any(k in line for k in ("流式生成开始", "流式 PCM 首片", "流式 PCM 完成",
                                           "取消", "cancel", "Error")):
                    print("   ", line)


if __name__ == "__main__":
    main()
