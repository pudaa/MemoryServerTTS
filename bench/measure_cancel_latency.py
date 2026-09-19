"""测量"客户端取消后，服务端还要占多久 GPU 锁"

真机反馈：停止播放后再点朗读，会有明显等待。怀疑是——客户端断开只停了播放，
服务端仍在继续生成（并持有 model_lock），于是下一次请求必须排队等它结束。

本脚本直接量化：
  1) 基线 TTFA：正常请求（无干扰）
  2) 取消后再请求：发一个大文本请求 → 读 ~1s 后**直接关闭连接**（模拟停止播放）
     → 立刻再发一个小请求 → 测第二个请求的 TTFA
     若取消生效，第二个请求 TTFA 应接近基线；若锁未释放，会明显变长。

用法:
    python bench/measure_cancel_latency.py --url http://127.0.0.1:8000/api/v1/tts/synthesize-stream
"""
import argparse
import json
import socket
import time
from urllib.parse import urlparse

SMALL = "Hi there, this is a short one."


def long_text(words=200):
    """构造一个需要较长时间生成的长文本"""
    base = ("This is a long paragraph used to keep the speech engine busy for a while, "
            "so that we can measure what happens when the client disconnects early. ")
    s = ""
    while len(s.split()) < words:
        s += base
    return s.strip()


def post_stream(url, text, read_seconds=None, timeout=300.0):
    """POST 请求；read_seconds 不为 None 时只读若干秒就关闭连接（模拟取消）。

    返回 (ttfa_ms, closed_early)
    """
    u = urlparse(url)
    host, port = u.hostname, u.port or 80
    payload = json.dumps({"text": text}).encode()
    t0 = time.perf_counter()
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
    ttfa = None
    if b"200" in head.split(b"\r\n")[0]:
        ttfa = (time.perf_counter() - t0) * 1000

    if read_seconds is None:
        # 读完整响应
        try:
            while True:
                d = s.recv(65536)
                if not d:
                    break
        except Exception:
            pass
        s.close()
        return ttfa, False

    # 只读一小段时间，然后**强制关闭**（模拟用户点"停止"）
    deadline = time.perf_counter() + read_seconds
    got = 0
    try:
        while time.perf_counter() < deadline:
            s.settimeout(max(0.05, deadline - time.perf_counter()))
            d = s.recv(65536)
            if not d:
                break
            got += len(d)
    except Exception:
        pass
    # 关键：立刻关闭 socket（不发 FIN 的优雅关闭也走 close → RST/FIN）
    try:
        s.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    s.close()
    return ttfa, got > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--read-seconds", type=float, default=1.0,
                    help="模拟取消前先读多少秒")
    ap.add_argument("--settle", type=float, default=0.2,
                    help="取消后到发下一个请求之间的间隔")
    args = ap.parse_args()

    print(f"=== 取消延迟测量: {args.url}\n")

    print("【0】预热一次")
    post_stream(args.url, SMALL)
    print("     done\n")

    print("【1】基线：直接请求小文本（3 次）")
    base = []
    for i in range(3):
        ttfa, _ = post_stream(args.url, SMALL)
        base.append(ttfa)
        print(f"     TTFA = {ttfa:.0f} ms")
    base_avg = sum(base) / len(base)
    print(f"     基线均值 = {base_avg:.0f} ms\n")

    print(f"【2】取消后再请求：先发长文本请求，读 {args.read_seconds}s 后强制断开，")
    print(f"     间隔 {args.settle}s 再发小文本请求")
    ttfa_long, got = post_stream(args.url, long_text(), read_seconds=args.read_seconds)
    print(f"     长文本请求首片 = {ttfa_long:.0f} ms（已收到数据={got}），随后断开连接")
    time.sleep(args.settle)

    t = time.perf_counter()
    ttfa2, _ = post_stream(args.url, SMALL)
    print(f"     第二个请求 TTFA = {ttfa2:.0f} ms\n")

    print("=== 判定 ===")
    ratio = ttfa2 / base_avg if base_avg else float("nan")
    print(f"  基线 {base_avg:.0f} ms → 取消后 {ttfa2:.0f} ms（{ratio:.2f}×）")
    if ratio < 2.0:
        print("  ✅ 取消后服务端很快释放（锁未被长期占用）")
    else:
        print("  ❌ 取消后下一个请求被明显阻塞 → 服务端未及时停止生成 / 未释放 model_lock")


if __name__ == "__main__":
    main()
