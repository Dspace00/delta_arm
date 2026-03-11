import socket
import json
import time
import os
import numpy as np

class DeltaStatusMonitor:
    """Delta机械臂状态监控器"""
    
    def __init__(self):
        self.running = True
        self.socket = None
        self.buffer = ""
        
        # 状态数据
        self.status_data = {
            'joint_angles': [0, 0, 0],
            'joint_angles_deg': [0, 0, 0],
            'ee_position': [0, 0, 0],
            'ee_velocity': [0, 0, 0],
            'target_angles': [0, 0, 0],
            'target_position': [0, 0, 0],
            'is_moving': False
        }
    
    def connect_to_server(self, host='localhost', port=12348):
        """连接到状态服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((host, port))
            print(f"已连接到状态服务器 {host}:{port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def receive_data(self):
        """从服务器接收数据"""
        try:
            self.socket.settimeout(0.1)
            
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
    
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_status(self):
        """显示状态信息"""
        self.clear_screen()
        
        print("=" * 60)
        print("           DELTA ROBOT STATUS MONITOR")
        print("=" * 60)
        
        # 显示关节角度
        print("\n--- JOINT ANGLES ---")
        for i in range(3):
            rad = self.status_data['joint_angles'][i]
            deg = self.status_data['joint_angles_deg'][i]
            target_rad = self.status_data['target_angles'][i]
            target_deg = np.degrees(target_rad)
            
            print(f"Motor {i+1}: {rad:7.4f} rad ({deg:7.2f}°) | "
                  f"Target: {target_rad:7.4f} rad ({target_deg:7.2f}°)")
        
        # 显示末端位置
        print("\n--- END EFFECTOR POSITION ---")
        ee_pos = self.status_data['ee_position']
        target_pos = self.status_data['target_position']
        ee_vel = self.status_data['ee_velocity']
        
        print(f"Current: X={ee_pos[0]:7.4f}, Y={ee_pos[1]:7.4f}, Z={ee_pos[2]:7.4f}")
        print(f"Target:  X={target_pos[0]:7.4f}, Y={target_pos[1]:7.4f}, Z={target_pos[2]:7.4f}")
        print(f"Velocity: VX={ee_vel[0]:7.4f}, VY={ee_vel[1]:7.4f}, VZ={ee_vel[2]:7.4f}")
        
        # 显示运动状态
        print("\n--- STATUS ---")
        status = "MOVING" if self.status_data['is_moving'] else "STATIONARY"
        print(f"Robot Status: {status}")
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to exit")
    
    def run_monitor(self):
        """运行状态监控"""
        if not self.connect_to_server():
            return
        
        try:
            print("状态监控已启动")
            
            while self.running:
                data = self.receive_data()
                if data is not None:
                    self.status_data = data
                    self.display_status()

                # time.sleep(0.001)  # 100Hz 更新频率

        except KeyboardInterrupt:
            print("\n状态监控被用户中断")
        except Exception as e:
            print(f"状态监控错误: {e}")
        finally:
            self.close()
    
    def close(self):
        """关闭监控"""
        self.running = False
        if self.socket:
            self.socket.close()
        print("状态监控已结束")


if __name__ == "__main__":
    monitor = DeltaStatusMonitor()
    monitor.run_monitor()