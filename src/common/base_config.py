"""
统一配置基类 —— 所有模块配置的公共父类
"""
import os
from pathlib import Path
from typing import Any, Optional
from src.common.logging import get_logger

_logger = get_logger("CONFIG")


class BaseConfig:
    """配置基类，子类需定义 env_prefix 和 config_path"""

    # 子类必须覆盖
    env_prefix: str = "APPCONF_"       # 环境变量前缀
    _default_config_path: Optional[Path] = None  # 默认 YAML 路径

    def __init__(self, config_path: Optional[str | Path] = None):
        self._config_path = Path(config_path) if config_path else self._default_config_path
        self._data: dict = {}

        if self._config_path and self._config_path.exists():
            self._load_yaml()
        else:
            if self._config_path:
                _logger.debug(f"配置文件不存在，使用默认值: {self._config_path}")
            self._loaded = True

    # ── YAML ──

    def _load_yaml(self):
        try:
            import yaml
        except ImportError:
            _logger.warning("PyYAML 未安装，跳过配置文件解析")
            self._loaded = True
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            self._loaded = True
            _logger.debug(f"配置已加载: {self._config_path}")
        except Exception as e:
            _logger.error(f"配置文件解析失败 ({self._config_path}): {e}")
            self._loaded = True

    def reload(self):
        """重新加载配置"""
        self._data = {}
        if self._config_path and self._config_path.exists():
            self._load_yaml()

    # ── 环境变量 ──

    def _env(self, key: str, default: Any = None) -> Any:
        """读取环境变量 <env_prefix>_<KEY>"""
        env_key = f"{self.env_prefix}{key.upper().replace('.', '_')}"
        return os.environ.get(env_key, default)

    # ── 点分隔路径访问 ──

    def _get(self, dotted_key: str, default: Any = None) -> Any:
        """从配置字典获取值，支持 'a.b.c' 路径"""
        keys = dotted_key.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict):
                node = node.get(k)
                if node is None:
                    return default
            else:
                return default
        return node

    def _set(self, dotted_key: str, value: Any):
        """设置配置值（仅内存，不写回文件）"""
        keys = dotted_key.split(".")
        node = self._data
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    # ── 摘要 ──

    def summary(self) -> dict:
        """返回配置摘要（子类应覆盖）"""
        return {
            "config_path": str(self._config_path) if self._config_path else None,
            "data_keys": list(self._data.keys()),
        }

    def print_summary(self):
        """打印配置摘要"""
        for k, v in self.summary().items():
            print(f"  {k}: {v}")
