"""探测远端 Java 服务是否包含流式端点（401=存在需鉴权；404=不存在）"""
import socket

HOST, PORT = "frp-fit.com", 60966


def probe(method, path):
    try:
        s = socket.create_connection((HOST, PORT), timeout=15)
    except Exception as e:
        return f"CONNECT_FAIL {type(e).__name__}"
    body = b'{"text":"x"}' if method == "POST" else b""
    req = (f"{method} {path} HTTP/1.1\r\nHost: {HOST}:{PORT}\r\n"
           f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
           f"Connection: close\r\n\r\n").encode() + body
    try:
        s.sendall(req)
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = s.recv(4096)
            if not d:
                break
            buf += d
        return buf.split(b"\r\n")[0].decode("latin1").strip()
    finally:
        s.close()


for m, p in [
    ("GET", "/actuator/health"),
    ("GET", "/tts/voices"),
    ("POST", "/tts/synthesize"),
    ("POST", "/tts/synthesize-stream"),   # 新增端点
    ("POST", "/tts/definitely-not-exist"),  # 对照：不存在
]:
    print(f"{m:5s} {p:32s} -> {probe(m, p)}")
