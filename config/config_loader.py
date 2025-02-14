"""
配置加载器
"""
from pathlib import Path
import yaml
from typing import Dict, Any

class Config:
    """配置加载器"""
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        config_path = Path(__file__).parent / 'config.yml'
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def _get_absolute_path(self, relative_path: str) -> Path:
        """将相对路径转换为绝对路径"""
        return Path(__file__).parent.parent / relative_path
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_path(self, key: str) -> Path:
        """获取路径配置"""
        path = self.get(f"paths.{key}")
        if path is None:
            raise KeyError(f"找不到路径配置：{key}")
        return self._get_absolute_path(path)

# 创建全局配置实例
config = Config()
