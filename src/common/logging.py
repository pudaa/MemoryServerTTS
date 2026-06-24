"""
统一日志模块 —— 为所有子模块提供一致的 [MODULE] 前缀输出

特性:
  - [时间戳] [模块名] 消息格式
  - ANSI 彩色输出 (INFO=默认, WARN=黄, ERROR=红, DEBUG=灰, SUCCESS=绿)
  - 同时输出到标准 logging 模块 (供外部工具捕获)
  - 日志级别控制 (环境变量 LOG_LEVEL=DEBUG|INFO|WARN|ERROR)

用法:
  from src.common.logging import get_logger

  logger = get_logger("TTS")
  logger.info("加载模型...")
  logger.success("模型加载完成")
  logger.warning("降级到 CPU")
  logger.error(f"加载失败: {e}")
  logger.debug("详细调试信息")
"""
import sys
import time
import logging
import os
from typing import Optional

# ── ANSI 颜色 ──
_RESET = "\033[0m"
_COLORS = {
    "DEBUG":   "\033[90m",   # 灰色
    "INFO":    "",           # 默认
    "SUCCESS": "\033[32m",   # 绿色
    "WARN":    "\033[33m",   # 黄色
    "ERROR":   "\033[31m",   # 红色
}
_ICONS = {
    "DEBUG":   "·",
    "INFO":    "→",
    "SUCCESS": "✓",
    "WARN":    "⚠",
    "ERROR":   "✗",
}

# ── 日志级别 ──
_LEVELS = {"DEBUG": 10, "INFO": 20, "SUCCESS": 25, "WARN": 30, "ERROR": 40}
_DEFAULT_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
if _DEFAULT_LEVEL not in _LEVELS:
    _DEFAULT_LEVEL = "INFO"

# ── 标准 logging 桥接 (仅作兼容，默认不输出到控制台) ──
_STD_LOGGER = logging.getLogger("MemoryServerTTS")
_STD_LOGGER.setLevel(logging.DEBUG)
# 默认不添加 handler，避免与彩色输出重复。如需文件日志，由调用方自行添加 handler。


class ModuleLogger:
    """模块级日志器 —— 统一 [MODULE] 前缀输出"""

    def __init__(self, module: str, min_level: Optional[str] = None):
        self._module = module
        self._min_level = _LEVELS.get(min_level or _DEFAULT_LEVEL, 20)

    def _should_log(self, level: str) -> bool:
        return _LEVELS.get(level, 20) >= self._min_level

    def _log(self, level: str, message: str):
        if not self._should_log(level):
            return

        color = _COLORS.get(level, "")
        icon = _ICONS.get(level, "·")
        timestamp = time.strftime("%H:%M:%S")

        # 彩色控制台输出
        formatted = f"{color}[{timestamp}] [{self._module}] {icon} {message}{_RESET}"
        print(formatted, flush=True)

        # 同时输出到标准 logging
        py_level = {"DEBUG": logging.DEBUG, "SUCCESS": logging.INFO}.get(level, getattr(logging, level, logging.INFO))
        _STD_LOGGER.log(py_level, f"[{self._module}] {message}")

    def debug(self, message: str):
        self._log("DEBUG", message)

    def info(self, message: str):
        self._log("INFO", message)

    def success(self, message: str):
        self._log("SUCCESS", message)

    def warning(self, message: str):
        self._log("WARN", message)

    def error(self, message: str):
        self._log("ERROR", message)

    # 打印分隔线
    def divider(self, char: str = "─", width: int = 50):
        self.info(char * width)

    # 打印键值对
    def kv(self, key: str, value, indent: int = 2):
        self.info(f"{' ' * indent}{key}: {value}")

    # 打印标题块
    def title(self, text: str):
        self.info("=" * 50)
        self.info(f"  {text}")
        self.info("=" * 50)


# ── 模块日志器缓存 ──
_loggers: dict[str, ModuleLogger] = {}


def get_logger(module: str) -> ModuleLogger:
    """获取模块日志器（懒加载单例）"""
    if module not in _loggers:
        _loggers[module] = ModuleLogger(module)
    return _loggers[module]
