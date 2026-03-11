"""
配置基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import yaml
import json
import os


class ConfigBase(ABC):
    """配置基类"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        初始化配置
        
        Args:
            config_dict: 配置字典，如果为None则使用默认配置
        """
        if config_dict is None:
            config_dict = self.get_default_config()
        
        self._config = config_dict
        self._load_from_dict(config_dict)
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        pass
    
    def _load_from_dict(self, config_dict: Dict):
        """从字典加载配置到类属性"""
        for key, value in config_dict.items():
            # 将嵌套字典转换为Config对象
            if isinstance(value, dict) and hasattr(self, key):
                existing_obj = getattr(self, key)
                if isinstance(existing_obj, ConfigBase):
                    existing_obj._load_from_dict(value)
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
    
    def update(self, new_config: Dict, deep_update: bool = True):
        """更新配置
        
        Args:
            new_config: 新配置字典
            deep_update: 是否深度更新（递归更新嵌套字典）
        """
        if deep_update:
            self._deep_update_dict(self._config, new_config)
        else:
            self._config.update(new_config)
        
        # 重新加载到属性
        self._load_from_dict(self._config)
    
    def _deep_update_dict(self, original: Dict, new: Dict):
        """深度更新字典"""
        for key, value in new.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update_dict(original[key], value)
            else:
                original[key] = value
    
    def to_dict(self) -> Dict:
        """将配置转换为字典"""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, ConfigBase):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def to_yaml(self) -> str:
        """将配置转换为YAML字符串"""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def to_json(self) -> str:
        """将配置转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, filepath: str, format: str = 'yaml'):
        """保存配置到文件
        
        Args:
            filepath: 文件路径
            format: 格式 ('yaml' 或 'json')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        elif format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, filepath: str):
        """从文件加载配置
        
        Args:
            filepath: 文件路径
            
        Returns:
            配置对象
        """
        filepath_str = str(filepath)
        with open(filepath, 'r') as f:
            if filepath_str.endswith('.yaml') or filepath_str.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            elif filepath_str.endswith('.json'):
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
        
        return cls(config_dict)
    
    def __getitem__(self, key):
        """支持字典式访问"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """支持字典式设置"""
        setattr(self, key, value)
        if key in self._config:
            self._config[key] = value
    
    def __contains__(self, key):
        """检查键是否存在"""
        return hasattr(self, key)
    
    def __str__(self):
        """字符串表示"""
        return f"{self.__class__.__name__}: {self.to_dict()}"
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        # 子类可以重写此方法进行验证
        return True