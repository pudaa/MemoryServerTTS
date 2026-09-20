import os
import sys

# 【重要】解决 SSL 证书验证失败问题 (针对内网或代理环境)
# 注意：生产环境建议配置正确的 CA 证书，此处仅为调试方便
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['CURL_CA_BUNDLE'] = '' 
# 强制禁用 HuggingFace Hub 的 SSL 验证
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# 尝试在导入 huggingface_hub 之前应用 monkey patch 到 requests (如果 hf_hub 使用它)
try:
    import requests
    from urllib3.exceptions import InsecureRequestWarning
    import urllib3
    urllib3.disable_warnings(InsecureRequestWarning)
    # 注意：这不会直接影响 hf_hub 的内部会话，但有助于依赖 requests 的其他库
except ImportError:
    pass

import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# 检查 CUDA 是否可用，如果不可用则提示用户安装正确的 PyTorch 版本
if not torch.cuda.is_available():
    print("\n[ERROR] CUDA is not available! You are likely using the CPU version of PyTorch.")
    print("Please reinstall PyTorch with CUDA support:")
    print("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)


# 测试生成带情感的英文音频，并自动播放
try:
    # 推荐优先使用本地大模型
    model_path = "./models/qwen-1.7b"
    if not os.path.exists(model_path):
        print(f"[警告] 未找到本地大模型目录 {model_path}，将尝试 0.6B 小模型。")
        model_path = "./models/qwen-0.6b"
    print(f"[INFO] 加载模型: {model_path}")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa"
    )
    print("[INFO] 模型加载完成！")

    # 测试文本，带情感指令
    test_text = """
    Hello, welcome to Memory English Learning App! Let's practice speaking with a happy and encouraging tone!
    你好，欢迎来到记忆英语学习应用！让我们用愉快和鼓励的语气来练习口语吧！
    """
    instructions = "Speak with a happy, warm, and encouraging tone, suitable for English teaching."
    print(f"[INFO] 合成文本: {test_text}\n情感指令: {instructions}")
    wavs, sr = model.generate_custom_voice(
        text=test_text,
        language="English",
        speaker="ono_anna",
        instruct=instructions
    )
    out_path = "test_emotion_output.wav"
    sf.write(out_path, wavs[0], sr)
    print(f"[INFO] 已保存音频: {out_path}")

    # 自动播放音频（跨平台）
    import platform
    import subprocess
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(out_path)
        elif system == "Darwin":
            subprocess.run(["open", out_path])
        else:
            subprocess.run(["aplay", out_path])
    except Exception as e:
        print(f"[WARN] 自动播放失败: {e}")
        print(f"请手动打开 {out_path} 试听效果。")

except Exception as e:
    print(f"\n[ERROR] Failed to load or run model: {e}")
    print("\n--- Troubleshooting Suggestions ---")
    print("1. 请确认 models/qwen-1.7b 或 models/qwen-0.6b 已离线下载。")
    print("2. 若显存不足可切换到小模型，或调整 dtype 为 float16/bfloat16。")
    print("3. SoX Warning: Install SoX from http://sox.sourceforge.net/ and add it to PATH to移除警告 (可选)。")