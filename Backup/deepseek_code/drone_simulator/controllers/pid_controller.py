"""
通用PID控制器
支持1D, 2D, 3D控制
"""
import numpy as np
from typing import Union, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from .base import ControllerBase


@dataclass
class PIDParams:
    """PID参数数据类"""
    kp: Union[float, List[float], np.ndarray]
    ki: Union[float, List[float], np.ndarray]
    kd: Union[float, List[float], np.ndarray]
    dt: float
    integral_limit: Union[float, List[float], np.ndarray] = None
    output_limit: Tuple[float, float] = (-float('inf'), float('inf'))
    anti_windup: str = 'clamping'  # 'clamping', 'back_calculation'
    feedforward_gain: float = 0.0
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保参数为numpy数组
        self.kp = np.asarray(self.kp)
        self.ki = np.asarray(self.ki)
        self.kd = np.asarray(self.kd)
        
        # 设置积分限制
        if self.integral_limit is None:
            self.integral_limit = np.inf * np.ones_like(self.kp)
        else:
            self.integral_limit = np.asarray(self.integral_limit)
        
        # 验证参数
        self._validate()
    
    def _validate(self):
        """验证参数有效性"""
        if np.any(self.kp < 0):
            raise ValueError("KP must be non-negative")
        if np.any(self.ki < 0):
            raise ValueError("KI must be non-negative")
        if np.any(self.kd < 0):
            raise ValueError("KD must be non-negative")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.anti_windup not in ['clamping', 'back_calculation']:
            raise ValueError("anti_windup must be 'clamping' or 'back_calculation'")


class PIDController(ControllerBase):
    """通用PID控制器类"""
    
    def __init__(self, params: PIDParams, name: str = "PID Controller"):
        """
        初始化PID控制器
        
        Args:
            params: PID参数
            name: 控制器名称
        """
        super().__init__(name)
        
        # 参数
        self.params = params
        
        # 状态变量
        self.dim = self._get_dimension(params.kp)
        self.integral = np.zeros(self.dim)
        self.previous_error = np.zeros(self.dim)
        self.previous_output = np.zeros(self.dim)
        self.previous_measurement = np.zeros(self.dim)
        self.derivative_filtered = np.zeros(self.dim)
        
        # 滤波器参数（用于微分项）
        self.derivative_filter_alpha = 0.1
        self.derivative_initialized = False
        
    def _get_dimension(self, gain: np.ndarray) -> int:
        """获取控制维度"""
        if gain.ndim == 0:
            return 1
        elif gain.ndim == 1:
            return len(gain)
        else:
            raise ValueError("Gain must be scalar or 1D array")
    
    def update(self, setpoint: Union[float, np.ndarray], 
               measurement: Union[float, np.ndarray],
               feedforward: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
        """
        更新PID控制器
        
        Args:
            setpoint: 目标值
            measurement: 测量值
            feedforward: 前馈项
            
        Returns:
            控制输出
        """
        # 转换为numpy数组
        setpoint = np.asarray(setpoint)
        measurement = np.asarray(measurement)
        
        # 确保维度一致
        if setpoint.shape != measurement.shape:
            raise ValueError("Setpoint and measurement must have same shape")
        
        # 计算误差
        error = setpoint - measurement
        
        # 比例项
        proportional = self.params.kp * error
        
        # 积分项（考虑积分限制）
        self.integral += error * self.params.dt
        self._apply_integral_limits()
        integral = self.params.ki * self.integral
        
        # 微分项（使用测量值微分以减少设定值突变的影响）
        if self.params.dt > 0:
            if self.derivative_initialized:
                derivative = (measurement - self.previous_measurement) / self.params.dt
                # 应用低通滤波器
                self.derivative_filtered = (
                    self.derivative_filter_alpha * derivative +
                    (1 - self.derivative_filter_alpha) * self.derivative_filtered
                )
            else:
                self.derivative_filtered = np.zeros_like(measurement)
                self.derivative_initialized = True
        else:
            self.derivative_filtered = np.zeros_like(measurement)
        
        derivative = -self.params.kd * self.derivative_filtered  # 负号因为使用测量值微分
        
        # 前馈项
        if feedforward is not None:
            feedforward = np.asarray(feedforward)
            feedforward_term = self.params.feedforward_gain * feedforward
        else:
            feedforward_term = 0
        
        # 计算总输出
        output = proportional + integral + derivative + feedforward_term
        
        # 应用输出限制
        output = self._limit_output(output)
        
        # 更新状态
        self.previous_error = error
        self.previous_measurement = measurement
        self.previous_output = output
        
        return output
    
    def _apply_integral_limits(self):
        """应用积分限制"""
        # 积分钳位抗饱和
        if self.params.anti_windup == 'clamping':
            # 当输出饱和时停止积分
            # 这里简化处理：检查上一次输出是否饱和
            output_saturated = (
                (self.previous_output <= self.params.output_limit[0]) |
                (self.previous_output >= self.params.output_limit[1])
            )
            self.integral[output_saturated] = self.previous_measurement[output_saturated]
        elif self.params.anti_windup == 'back_calculation':
            # 回算抗饱和（需要知道执行器饱和值）
            pass
        
        # 应用积分限幅
        self.integral = np.clip(self.integral, -self.params.integral_limit, self.params.integral_limit)
    
    def _limit_output(self, output: np.ndarray) -> np.ndarray:
        """限制输出范围"""
        if (self.params.output_limit[0] != -float('inf') or 
            self.params.output_limit[1] != float('inf')):
            return np.clip(output, self.params.output_limit[0], self.params.output_limit[1])
        return output
    
    def reset(self):
        """重置控制器状态"""
        self.integral = np.zeros(self.dim)
        self.previous_error = np.zeros(self.dim)
        self.previous_output = np.zeros(self.dim)
        self.previous_measurement = np.zeros(self.dim)
        self.derivative_filtered = np.zeros(self.dim)
        self.derivative_initialized = False
    
    def set_parameters(self, kp=None, ki=None, kd=None):
        """动态设置PID参数"""
        if kp is not None:
            self.params.kp = np.asarray(kp)
        if ki is not None:
            self.params.ki = np.asarray(ki)
        if kd is not None:
            self.params.kd = np.asarray(kd)
    
    def get_state(self) -> dict:
        """获取控制器状态"""
        return {
            'integral': self.integral.copy(),
            'previous_error': self.previous_error.copy(),
            'previous_output': self.previous_output.copy(),
            'derivative_filtered': self.derivative_filtered.copy()
        }
    
    def __str__(self):
        return f"PIDController(name={self.name}, dim={self.dim}, KP={self.params.kp}, KI={self.params.ki}, KD={self.params.kd})"