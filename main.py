import os
import uvicorn

if __name__ == "__main__":
    # 调试时设置 RELOAD=0 避免父子进程问题
    use_reload = os.environ.get("RELOAD", "0").lower() in ("1", "true", "yes")
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=use_reload)

    # 启动：python main.py 