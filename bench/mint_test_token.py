"""生成一个测试用 access token（HS256），供探测直连播放的 HTTP 能力使用。

用途：验证 /tts-audio/** 在「带 Bearer」下是否可取、是否支持 Range。
这不是绕过鉴权——用的是服务端**真实的** MEMORY_AUTH_JWT_SECRET（从环境变量读），
只是省去走一遍 /auth/login（测试账号密码是 PBKDF2 哈希，不可逆）。

用法:
    python bench/mint_test_token.py --user-id 1
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import time
import uuid


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--role", default="USER")
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--secret", default=None,
                    help="默认从 env MEMORY_AUTH_JWT_SECRET 或 Windows User 环境读取")
    args = ap.parse_args()

    secret = args.secret or os.environ.get("MEMORY_AUTH_JWT_SECRET")
    if not secret:
        # 尝试 Windows 用户级环境变量
        try:
            import subprocess
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[Environment]::GetEnvironmentVariable('MEMORY_AUTH_JWT_SECRET','User')"],
                capture_output=True, text=True, timeout=20).stdout.strip()
            secret = out or None
        except Exception:
            secret = None
    if not secret:
        raise SystemExit("找不到 MEMORY_AUTH_JWT_SECRET（可用 --secret 传入）")

    # JwtTokenService 用的是 configuredSecret.getBytes(UTF_8) —— **原始 UTF-8 字节**，
    # 不是 base64 解码后的字节。搞错会得到 401。
    key = secret.encode("utf-8")

    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(args.user_id),
        "role": args.role,
        "typ": "access",
        "fam": uuid.uuid4().hex,
        "iat": now,
        "exp": now + args.seconds,
    }
    signing_input = f"{b64u(json.dumps(header, separators=(',', ':')).encode())}." \
                    f"{b64u(json.dumps(payload, separators=(',', ':')).encode())}"
    sig = hmac.new(key, signing_input.encode(), hashlib.sha256).digest()
    print(f"{signing_input}.{b64u(sig)}")


if __name__ == "__main__":
    main()
