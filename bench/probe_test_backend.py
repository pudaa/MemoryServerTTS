"""检查 TEST 后端（frp-fit.com:60966）的基础设施状况

目的：判断"正式版指向 TEST 后端"是否可行。
- Java 服务在不在：/actuator/health
- 它下游的 TTS 服务（Python，通常 localhost:8000）在不在：看 /tts/synthesize 是否 5xx
  （401=需鉴权但路由存在；500/503=路由在但下游不可用）
"""
import json
import socket

HOST, PORT = "frp-fit.com", 60966


def req(method, path, body=None):
    try:
        s = socket.create_connection((HOST, PORT), timeout=20)
    except Exception as e:
        return None, f"CONNECT_FAIL {type(e).__name__}: {e}", b""
    data = json.dumps(body).encode() if body is not None else b""
    head = (f"{method} {path} HTTP/1.1\r\nHost: {HOST}:{PORT}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\n"
            f"Connection: close\r\n\r\n").encode()
    try:
        s.sendall(head + data)
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = s.recv(4096)
            if not d:
                break
            buf += d
        h, _, rest = buf.partition(b"\r\n\r\n")
        lines = h.decode("latin1").split("\r\n")
        status = lines[0]
        ch = {}
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                ch[k.strip().lower()] = v.strip()
        payload = rest
        if ch.get("transfer-encoding", "").lower() == "chunked":
            try:
                n = int(payload.split(b"\r\n")[0], 16)
                payload = payload.split(b"\r\n", 1)[1][:n]
            except Exception:
                pass
        return status, ch, payload
    finally:
        s.close()


print("=== TEST 后端基础设施探测 ===")
st, hd, body = req("GET", "/actuator/health")
print(f"\n[1] GET /actuator/health -> {st}")
if body:
    print("    body:", body[:300].decode("utf-8", "replace"))

st, hd, body = req("POST", "/tts/synthesize", {"text": "hello", "language": "english"})
print(f"\n[2] POST /tts/synthesize（未带 token）-> {st}")
print("    （401=路由存在需鉴权；404=路由不存在；5xx=路由在但下游 TTS 异常）")

st, hd, body = req("POST", "/tts/synthesize-stream", {"text": "hello"})
print(f"\n[3] POST /tts/synthesize-stream（未带 token）-> {st}")
print("    （若与 [2] 表现一致，无法据此区分是否已部署新端点——见文档说明）")
