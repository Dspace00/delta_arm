"""
模块功能说明：
定义控制器的抽象基类。

所有具体控制器应继承此类并实现 update 方法。
"""

from abc import ABC, abstractmethod

class BaseController(ABC):
    def __init__(self, config):
        """
        初始化控制器
        :param config: 配置字典 (包含kp, ki, kd, limits等)
        """
        self.config = config

    @abstractmethod
    def update(self, setpoint, measurement):
        """
        计算控制输出
        :param setpoint: 目标值
        :param measurement: 当前测量值
        :return: 控制输出
        """
        pass

    @abstractmethod
    def reset(self):
        """重置内部状态"""
        pass
