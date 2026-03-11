# DeltaVisualizer.py
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import socket
import json
import time
import threading

class DeltaVisualizer:
    """Delta机械臂3D轨迹可视化器"""
    
    def __init__(self, max_points=500):
        self.running = True
        self.max_points = max_points
        self.trajectory = deque(maxlen=max_points)
        self.current_position = np.array([0, 0, -0.15])
        self.target_position = np.array([0, 0, -0.15])
        self.is_moving = False
        
        # 创建图形界面
        self._setup_plot()
        
        # Socket连接
        self.socket = None
        self.buffer = ""
    
    def _setup_plot(self):
        """设置3D绘图界面"""
        plt.switch_backend('TkAgg')
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Delta Robot End-Effector Trajectory')
        
        # 设置初始视图范围
        self.ax.set_xlim([-0.2, 0.2])
        self.ax.set_ylim([-0.2, 0.2])
        self.ax.set_zlim([-0.3, 0])
        
        # 创建图形对象
        self.trajectory_scatter = self.ax.scatter([], [], [], c='blue', marker='.', s=10, alpha=0.6)
        self.current_point = self.ax.scatter([], [], [], c='red', marker='o', s=50)
        self.target_point = self.ax.scatter([], [], [], c='green', marker='x', s=100, linewidth=2)
        
        # 添加图例和网格
        self.ax.legend(['Trajectory', 'Current Position', 'Target Position'])
        self.ax.grid(True)
        self.ax.set_box_aspect([1, 1, 1])
        
        # 添加工作空间指示
        self._draw_workspace_indicator()
    
    def _draw_workspace_indicator(self):
        """绘制工作空间指示"""
        # 绘制工作空间边界
        theta = np.linspace(0, 2*np.pi, 100)
        r = 0.15
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z_upper = np.full_like(x, -0.05)
        z_lower = np.full_like(x, -0.25)
        
        self.ax.plot(x, y, z_upper, 'r--', alpha=0.3, label='Workspace')
        self.ax.plot(x, y, z_lower, 'r--', alpha=0.3)
    
    def connect_to_server(self, host='localhost', port=12346):
        """连接到可视化服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((host, port))
            print(f"已连接到可视化服务器 {host}:{port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def receive_data(self):
        """从服务器接收数据"""
        try:
            self.socket.settimeout(0.01)
            
            data = b""
            while True:
                try:
                    chunk = self.socket.recv(1024)
                    if not chunk:
                        break
                    data += chunk
                except socket.timeout:
                    break
            
            if data:
                self.buffer += data.decode()
                lines = self.buffer.split('\n')
                
                if lines:
                    self.buffer = lines[-1]
                    complete_lines = lines[:-1]
                    
                    for line in complete_lines:
                        if line.strip():
                            try:
                                return json.loads(line)
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"接收数据错误: {e}")
            self.running = False
        
        return None
    
    def update_visualization(self):
        """更新可视化显示"""
        data = self.receive_data()
        
        if data is not None:
            position = data.get("position", [0, 0, -0.15])
            target_pos = data.get("target_position", [0, 0, -0.15])
            is_moving = data.get("is_moving", False)
            
            self.current_position = np.array(position)
            self.target_position = np.array(target_pos)
            self.is_moving = is_moving
            
            # 添加到轨迹
            self.trajectory.append(self.current_position.copy())
        
        # 更新图形
        if len(self.trajectory) > 0:
            traj_array = np.array(self.trajectory)
            
            # 更新轨迹散点
            self.trajectory_scatter._offsets3d = (
                traj_array[:, 0], traj_array[:, 1], traj_array[:, 2]
            )
            
            # 更新当前位置点
            self.current_point._offsets3d = (
                [self.current_position[0]], 
                [self.current_position[1]], 
                [self.current_position[2]]
            )
            
            # 更新目标位置点
            self.target_point._offsets3d = (
                [self.target_position[0]], 
                [self.target_position[1]], 
                [self.target_position[2]]
            )
            
            # 动态调整视图范围
            self._adjust_view_limits(traj_array)
            
            # 更新标题显示状态
            status = "MOVING" if self.is_moving else "STATIONARY"
            self.ax.set_title(f'Delta Robot Trajectory - Status: {status}')
        
        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return True
    
    def _adjust_view_limits(self, traj_array):
        """动态调整视图范围"""
        if len(traj_array) > 10:
            x_data = traj_array[:, 0]
            y_data = traj_array[:, 1]
            z_data = traj_array[:, 2]
            
            # 计算数据范围
            x_range = np.max(x_data) - np.min(x_data)
            y_range = np.max(y_data) - np.min(y_data)
            z_range = np.max(z_data) - np.min(z_data)
            
            # 添加边距
            margin = 0.02
            x_margin = max(x_range * 0.1, margin)
            y_margin = max(y_range * 0.1, margin)
            z_margin = max(z_range * 0.1, margin)
            
            self.ax.set_xlim([np.min(x_data) - x_margin, np.max(x_data) + x_margin])
            self.ax.set_ylim([np.min(y_data) - y_margin, np.max(y_data) + y_margin])
            self.ax.set_zlim([np.min(z_data) - z_margin, np.max(z_data) + z_margin])
    
    def run_visualization(self):
        """运行可视化主循环"""
        if not self.connect_to_server():
            return
        
        try:
            plt.ion()  # 开启交互模式
            plt.show(block=False)
            
            print("3D轨迹可视化已启动")
            
            while self.running:
                if not self.update_visualization():
                    break
                time.sleep(0.05)  # 20Hz 更新频率
                
        except KeyboardInterrupt:
            print("可视化被用户中断")
        except Exception as e:
            print(f"可视化错误: {e}")
        finally:
            self.close()
    
    def close(self):
        """关闭可视化"""
        self.running = False
        if self.socket:
            self.socket.close()
        plt.close(self.fig)
        print("可视化已结束")