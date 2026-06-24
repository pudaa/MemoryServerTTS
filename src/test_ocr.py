"""
OCR 模块验证脚本 —— 测试 PaddleOCR 安装和图片识别功能
运行方式: python src/test_ocr.py
"""
import os
import sys
import time
from pathlib import Path

# 确保项目路径在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_import():
    """测试 1: 检查 paddleocr 是否可导入"""
    print("=" * 60)
    print("测试 1: 检查 paddleocr 安装")
    print("=" * 60)
    try:
        import paddleocr
        print(f"  ✅ paddleocr 版本: {paddleocr.__version__}")
        return True
    except ImportError as e:
        print(f"  ❌ paddleocr 未安装: {e}")
        print("  请运行: pip install paddleocr onnxruntime-gpu")
        return False


def test_onnxruntime():
    """测试 2: 检查 onnxruntime-gpu 是否可用"""
    print("\n" + "=" * 60)
    print("测试 2: 检查 onnxruntime-gpu")
    print("=" * 60)
    try:
        import onnxruntime
        print(f"  ✅ onnxruntime 版本: {onnxruntime.__version__}")
        providers = onnxruntime.get_available_providers()
        print(f"  可用执行提供器: {providers}")
        if "CUDAExecutionProvider" in providers:
            print("  ✅ CUDA GPU 加速可用")
        elif "DmlExecutionProvider" in providers:
            print("  ⚠️ DirectML 可用 (Windows GPU)")
        else:
            print("  ⚠️ 仅 CPU 可用")
        return True
    except ImportError as e:
        print(f"  ❌ onnxruntime 未安装: {e}")
        print("  请运行: pip install onnxruntime-gpu")
        return False


def test_ocr_engine_init():
    """测试 3: 初始化 OCR 引擎"""
    print("\n" + "=" * 60)
    print("测试 3: 初始化 OCR 引擎")
    print("=" * 60)
    try:
        from src.ocr_engine import OCREngine

        engine = OCREngine(engine="onnxruntime", lang="en", device="gpu")
        print(f"  ✅ OCREngine 创建成功 (engine={engine.engine}, lang={engine.lang})")

        print("  正在加载 OCR 模型（首次运行会自动下载，约 20-50MB）...")
        t0 = time.perf_counter()
        engine.load()
        elapsed = time.perf_counter() - t0
        print(f"  ✅ 模型加载成功，耗时 {elapsed:.1f}s")
        return engine
    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        return None


def test_ocr_predict(engine):
    """测试 4: 用示例图片测试 OCR 识别"""
    print("\n" + "=" * 60)
    print("测试 4: OCR 识别测试")
    print("=" * 60)

    # 使用 PaddleOCR 官方示例图片
    test_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"

    # 先尝试下载测试图片
    import tempfile
    import urllib.request

    test_img_path = None
    try:
        print(f"  下载测试图片: {test_url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            urllib.request.urlretrieve(test_url, tmp.name)
            test_img_path = tmp.name
        print(f"  ✅ 测试图片已下载: {test_img_path}")
    except Exception as e:
        print(f"  ⚠️ 无法下载在线图片: {e}")
        print("  请手动准备一张英文文本图片后，使用以下代码测试：")
        print("  >>> from src.ocr_engine import OCREngine")
        print("  >>> engine = OCREngine(engine='onnxruntime', lang='en')")
        print("  >>> engine.load()")
        print("  >>> result = engine.predict_image('your_image.png')")
        print("  >>> print(result['text'])")
        return

    try:
        result = engine.predict_image(test_img_path)

        if result.get("success"):
            print(f"  ✅ 识别成功！")
            print(f"  处理耗时: {result['processing_time_ms']:.1f}ms")
            print(f"  平均置信度: {result['avg_confidence']:.2%}")
            print(f"  识别行数: {len(result['lines'])}")
            print(f"  识别文本:")
            print(f"  ---")
            for i, line in enumerate(result["lines"], 1):
                conf = result["confidences"][i-1] if i-1 < len(result["confidences"]) else 0
                print(f"  [{i}] ({conf:.2%}) {line}")
            print(f"  ---")
        else:
            print(f"  ❌ 识别失败: {result.get('error')}")
    except Exception as e:
        print(f"  ❌ 识别异常: {e}")
    finally:
        if test_img_path and os.path.exists(test_img_path):
            os.remove(test_img_path)


def test_fallback_engines():
    """测试 5: 尝试其他引擎作为备选"""
    print("\n" + "=" * 60)
    print("测试 5: 引擎备选方案检查")
    print("=" * 60)

    engines_to_try = []

    # 检查 transformers 引擎是否可用
    try:
        import transformers
        engines_to_try.append("transformers")
        print(f"  ✅ transformers 引擎可用 (已安装 v{transformers.__version__})")
    except ImportError:
        print("  ⚠️ transformers 不可用")

    # 检查 PaddlePaddle 是否可用
    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            engines_to_try.append("paddle_static")
            print(f"  ✅ paddle_static 引擎可用 (GPU)")
        else:
            print(f"  ⚠️ PaddlePaddle 已安装但无 CUDA 支持")
    except ImportError:
        print("  ⚠️ PaddlePaddle 未安装 (若 onnxruntime 不可用可尝试安装)")

    print(f"\n  可用引擎列表: {engines_to_try if engines_to_try else '无 (仅 onnxruntime)'}")

    return engines_to_try


def main():
    print("\n" + "█" * 60)
    print("  PaddleOCR 模块验证工具")
    print("  项目: MemoryServerTTS")
    print("█" * 60 + "\n")

    results = {}

    # 测试 1
    results["import"] = test_import()
    if not results["import"]:
        print("\n❌ 基础依赖未安装，请先运行:")
        print("   pip install -r requirements-ocr.txt")
        return False

    # 测试 2
    results["onnxruntime"] = test_onnxruntime()

    # 测试 3
    engine = test_ocr_engine_init()
    results["init"] = engine is not None

    # 测试 4
    if engine is not None:
        test_ocr_predict(engine)
    else:
        results["predict"] = False

    # 测试 5
    fallbacks = test_fallback_engines()

    # 总结
    print("\n" + "=" * 60)
    print("  验证总结")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    if all(results.values()):
        print("\n🎉 所有测试通过！OCR 模块可正常使用。")
        return True
    else:
        print("\n⚠️ 部分测试未通过，请根据提示修复。")
        print("\n📋 快速修复指南:")
        print("  1. 安装 OCR 依赖:")
        print("     pip install -r requirements-ocr.txt")
        print("  2. 如果 onnxruntime-gpu 不可用，尝试 CPU 版本:")
        print("     pip install onnxruntime  (替换 onnxruntime-gpu)")
        print("  3. 如果 paddleocr 与 PyTorch 冲突，考虑使用 Docker 部署 OCR 服务:")
        print("     docker run -it --gpus all --network host \\")
        print("       ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
