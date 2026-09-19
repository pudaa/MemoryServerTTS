"""链路首字节探测：对任意 URL 做 chunked 流式探测（裸 socket，不依赖 httpx/requests）

用于分层定位首字节延迟：
  Python  : http://127.0.0.1:8000/api/v1/tts/synthesize-stream
  Java    : http://127.0.0.1:8080/tts/synthesize-stream   （Step 2 透传层）

用法:
    python bench/probe_chain.py --url <url> [--label "层名"] [--text "..."] [--repeat 2]
"""
import argparse
import json
import socket
import time
from urllib.parse import urlparse

SHORT = "Hello there, my friend."
LONG = (
    "That's a really thoughtful question, and I'm glad you asked it. Let me walk through it step by step. "
    "First, it helps to understand what you're actually trying to achieve, because the best method depends "
    "entirely on your goal. If your aim is fluency, then daily practice matters far more than long study "
    "sessions once a week. Second, try to make the practice enjoyable, since you'll stick with something "
    "you like much longer than something that feels like a chore. You could watch short videos, read simple "
    "articles, or talk with a partner about topics you genuinely care about."
)


def probe_once(url, text, timeout=300.0):
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
    head, _, rest = buf.partition(b"\r\n\r\n")
    lines = head.decode("latin1").split("\r\n")
    status = lines[0]
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    sr = int(headers.get("x-audio-sample-rate", "24000"))
    chunked = "chunked" in headers.get("transfer-encoding", "").lower()

    first_at = None
    total = 0
    nchunk = 0
    pending = rest

    def feed(data):
        nonlocal first_at, total, nchunk
        if not data:
            return
        if first_at is None:
            first_at = time.perf_counter() - t0
        nchunk += 1
        total += len(data)

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

    elapsed = time.perf_counter() - t0
    return {
        "status": status,
        "ttfa_ms": None if first_at is None else first_at * 1000,
        "total_s": elapsed,
        "chunks": nchunk,
        "bytes": total,
        "audio_s": total / 2 / sr,
        "server_ttfa": headers.get("x-tts-ttfa-ms"),
        "content_type": headers.get("content-type"),
        "chunked": chunked,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args()

    cases = [("长文本(110词)", LONG)] if args.long_only else [("短句", SHORT), ("长文本(110词)", LONG)]
    print(f"=== 目标: {args.label or args.url}")
    print(f"    {args.url}\n")
    print(f"{'case':16s} {'状态':>10s} {'TTFA':>9s} {'服务端自报':>10s} {'分片':>5s} "
          f"{'音频':>7s} {'总耗时':>8s}")
    print("-" * 76)
    for cname, text in cases:
        for i in range(args.repeat):
            try:
                r = probe_once(args.url, text)
                ttfa = f"{r['ttfa_ms']:.0f}ms" if r["ttfa_ms"] is not None else "N/A"
                srv = f"{r['server_ttfa']}ms" if r["server_ttfa"] else "-"
                print(f"{cname:16s} {r['status'][9:]:>10s} {ttfa:>9s} {srv:>10s} "
                      f"{r['chunks']:5d} {r['audio_s']:6.2f}s {r['total_s']:7.2f}s")
            except Exception as e:
                print(f"{cname:16s}  FAILED: {type(e).__name__}: {e}")
    print()


if __name__ == "__main__":
    main()
