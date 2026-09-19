"""探测 /tts-audio/** 是否支持 MediaPlayer 直连播放所需的能力

方案 A（MediaPlayer + Authorization 头直连播放）依赖三件事：
  1. 带 Bearer 头能取到音频（鉴权没问题）
  2. 支持 HTTP Range → MediaPlayer 才能渐进缓冲 / 拖动进度
  3. 返回正确的 audio/wav Content-Type 与 Content-Length

本脚本用裸 socket 逐项验证（本机 httpx/requests 受代理环境变量影响不可靠）。

用法:
    python bench/probe_audio_http.py --url "http://frp-fit.com:60966/tts-audio/<file>.wav" --token <accessToken>
    # 或不带 token 先看 401/403 行为
"""
import argparse
import socket
from urllib.parse import urlparse


def request(host, port, path, method="GET", headers=None, read_body=2048, timeout=20):
    """发一个原始 HTTP 请求，返回 (状态行, 响应头 dict, 收到的 body 字节)"""
    hdrs = {"Host": f"{host}:{port}", "Connection": "close"}
    if headers:
        hdrs.update(headers)
    req = f"{method} {path} HTTP/1.1\r\n" + "".join(
        f"{k}: {v}\r\n" for k, v in hdrs.items()) + "\r\n"
    s = socket.create_connection((host, port), timeout=timeout)
    s.settimeout(timeout)          # 逐次 recv 的超时，避免服务端不关连接时挂死
    s.sendall(req.encode())
    buf = b""
    try:
        while b"\r\n\r\n" not in buf:
            d = s.recv(4096)
            if not d:
                break
            buf += d
    except socket.timeout:
        s.close()
        return "(响应头读取超时)", {}, b""
    head, _, rest = buf.partition(b"\r\n\r\n")
    lines = head.decode("latin1").split("\r\n")
    status = lines[0] if lines else "(空响应)"
    h = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            h[k.strip().lower()] = v.strip()
    body = rest
    # 若已知长度，按长度读满；否则尽力读到 EOF 或读满 read_body
    want = read_body
    if "content-length" in h:
        try:
            want = min(read_body, int(h["content-length"]))
        except ValueError:
            pass
    try:
        while len(body) < want:
            d = s.recv(4096)
            if not d:
                break
            body += d
    except socket.timeout:
        pass
    s.close()
    return status, h, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", default="")
    args = ap.parse_args()

    u = urlparse(args.url)
    host, port = u.hostname, u.port or 80
    path = u.path
    print(f"=== 目标 {args.url}")
    print(f"    host={host}:{port} path={path}\n")

    def show(label, status, h, body):
        print(f"--- {label}")
        print(f"    {status}")
        for k in ("content-type", "content-length", "accept-ranges",
                  "content-range", "cache-control", "transfer-encoding"):
            if k in h:
                print(f"    {k}: {h[k]}")
        print(f"    body: {len(body)} bytes, 前 16 字节 = {body[:16]!r}")
        is_wav = body[:4] == b"RIFF"
        print(f"    RIFF/WAV 头 = {is_wav}")
        print()

    # 1) 不带 token：确认鉴权确实拦截
    st, h, b = request(host, port, path)
    show("1) 无 Authorization（预期 401/403）", st, h, b)
    ok_auth_required = ("401" in st or "403" in st)

    # 2) 带 token：确认能取到音频
    if not args.token:
        print("未提供 --token，跳过鉴权与 Range 测试。")
        return
    auth = {"Authorization": "Bearer " + args.token}
    st, h, b = request(host, port, path, headers=auth)
    show("2) 带 Bearer（预期 200 + RIFF）", st, h, b)
    ok_200 = "200" in st

    # 3) Range 请求：MediaPlayer 拖动进度/渐进缓冲的关键
    st, h, b = request(host, port, path, headers={**auth, "Range": "bytes=0-1023"})
    show("3) Range: bytes=0-1023（MediaPlayer 关键能力）", st, h, b)
    ok_range = "206" in st and "content-range" in h

    print("=== 结论 ===")
    if "超时" in st or not st.startswith("HTTP/"):
        print("  ⚠️ 没拿到有效 HTTP 响应（连接超时/被拒/被反代吞掉）——本次探测**无结论**，")
        print("     不要据此判断鉴权或 Range。请改用可达的地址（如本机 127.0.0.1:8080）重测。")
        return
    print(f"  鉴权拦截生效      : {'✅' if ok_auth_required else '❌ 无 token 也能取到，鉴权有问题'}")
    print(f"  带 token 可取音频 : {'✅' if ok_200 else '❌'}")
    print(f"  支持 Range(206)   : {'✅ 可拖动进度/渐进缓冲' if ok_range else '❌ 不支持 Range —— MediaPlayer 仍能播，但可能无法 seek'}")
    if ok_200 and not ok_range:
        print("  → 建议：给 /tts-audio/** 开启 Range 支持（Spring 静态资源默认支持，"
              "若被自定义 ResourceHandler 覆盖则需检查）")


if __name__ == "__main__":
    main()
