# DeltaCommandClient.py
import socket
import json
import time
import numpy as np

class DeltaCommandClient:
    """Delta机械臂命令客户端"""
    
    def __init__(self, host='localhost', port=12347):
        self.host = host
        self.port = port
        self.socket = None
        
        # 轨迹参数
        self.trajectory_params = {
            'circle': {'center': [0, 0, -0.15], 'radius': 0.05, 'duration': 10.0},
            'square': {'center': [0, 0, -0.15], 'side': 0.08, 'duration': 12.0},
            'figure8': {'center': [0, 0, -0.15], 'size': 0.04, 'tilt': 30, 'duration': 8.0}
        }
    
    def connect(self):
        """连接到仿真器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"已连接到仿真器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接仿真器失败: {e}")
            return False
    
    def send_command(self, command, **kwargs):
        """发送命令到仿真器"""
        if self.socket is None:
            print("未连接到仿真器")
            return None
        
        message = {"command": command, **kwargs}
        
        try:
            self.socket.sendall((json.dumps(message) + "\n").encode())
            
            # 等待响应
            response = b""
            while True:
                chunk = self.socket.recv(1024)
                if not chunk:
                    break
                response += chunk
                if b"\n" in chunk:
                    break
            
            if response:
                return json.loads(response.decode().strip())
            else:
                return None
                
        except Exception as e:
            print(f"发送命令错误: {e}")
            return None
    
    def move_to_position(self, x, y, z, speed=1.0):
        """移动到指定位置"""
        target_pos = [x, y, z]
        return self.send_command("move_to", target_position=target_pos, speed=speed)
    
    def set_joint_angles(self, angle1, angle2, angle3):
        """直接设置关节角度"""
        joint_angles = [angle1, angle2, angle3]
        return self.send_command("set_joint_angles", joint_angles=joint_angles)
    
    def get_status(self):
        """获取当前状态"""
        return self.send_command("get_status")
    
    def stop_movement(self):
        """停止运动"""
        return self.send_command("stop")
    
    def execute_trajectory(self, trajectory_type, **params):
        """执行预定义轨迹"""
        if trajectory_type == "circle":
            center = params.get('center', self.trajectory_params['circle']['center'])
            radius = params.get('radius', self.trajectory_params['circle']['radius'])
            duration = params.get('duration', self.trajectory_params['circle']['duration'])
            speed = params.get('speed', 1.0)
            
            return self._execute_circle_trajectory(center, radius, speed, duration)
        
        elif trajectory_type == "square":
            center = params.get('center', self.trajectory_params['square']['center'])
            side = params.get('side', self.trajectory_params['square']['side'])
            duration = params.get('duration', self.trajectory_params['square']['duration'])
            speed = params.get('speed', 1.0)
            
            return self._execute_square_trajectory(center, side, speed, duration)
        
        elif trajectory_type == "figure8":
            center = params.get('center', self.trajectory_params['figure8']['center'])
            size = params.get('size', self.trajectory_params['figure8']['size'])
            tilt = params.get('tilt', self.trajectory_params['figure8']['tilt'])
            duration = params.get('duration', self.trajectory_params['figure8']['duration'])
            speed = params.get('speed', 1.0)
            
            return self._execute_figure8_trajectory(center, size, tilt, speed, duration)
        
        else:
            print(f"未知轨迹类型: {trajectory_type}")
            return False
    
    def _execute_circle_trajectory(self, center, radius, speed=1.0, duration=10.0):
        """执行圆形轨迹"""
        print(f"开始圆形轨迹: 中心={center}, 半径={radius}")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            t = (time.time() - start_time) / duration
            angle = 2 * np.pi * t
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            
            response = self.move_to_position(x, y, z, speed)
            if response is None or response.get("status") != "success":
                print("圆形轨迹执行失败")
                return False
            
            time.sleep(0.02)
        
        return True
    
    def _execute_square_trajectory(self, center, side_length, speed=1.0, duration=12.0):
        """执行正方形轨迹"""
        print(f"开始正方形轨迹: 中心={center}, 边长={side_length}")
        
        half_side = side_length / 2
        corners = [
            [center[0] - half_side, center[1] - half_side, center[2]],
            [center[0] + half_side, center[1] - half_side, center[2]],
            [center[0] + half_side, center[1] + half_side, center[2]],
            [center[0] - half_side, center[1] + half_side, center[2]]
        ]
        
        start_time = time.time()
        corner_time = duration / 4
        
        for i in range(4):
            corner_start_time = time.time()
            while time.time() - corner_start_time < corner_time:
                t = (time.time() - corner_start_time) / corner_time
                
                current_corner = corners[i]
                next_corner = corners[(i + 1) % 4]
                
                target_pos = [
                    current_corner[0] + (next_corner[0] - current_corner[0]) * t,
                    current_corner[1] + (next_corner[1] - current_corner[1]) * t,
                    current_corner[2] + (next_corner[2] - current_corner[2]) * t
                ]
                
                response = self.move_to_position(target_pos[0], target_pos[1], target_pos[2], speed)
                if response is None or response.get("status") != "success":
                    print("正方形轨迹执行失败")
                    return False
                
                time.sleep(0.02)
        
        return True
    
    def _execute_figure8_trajectory(self, center, size, tilt_angle=30, speed=1.0, duration=8.0):
        """执行8字形轨迹"""
        print(f"开始8字形轨迹: 中心={center}, 大小={size}, 倾斜角={tilt_angle}度")
        
        tilt_rad = np.radians(tilt_angle)
        start_time = time.time()
        
        while time.time() - start_time < duration:
            t = (time.time() - start_time) / duration
            angle = 4 * np.pi * t
            
            u = size * np.sin(angle)
            v = size * np.sin(angle) * np.cos(angle)
            
            x = center[0] + u
            y = center[1] + v * np.cos(tilt_rad)
            z = center[2] + v * np.sin(tilt_rad)
            
            response = self.move_to_position(x, y, z, speed)
            if response is None or response.get("status") != "success":
                print("8字形轨迹执行失败")
                return False
            
            time.sleep(0.02)
        
        return True
    
    def interactive_control(self):
        """交互式控制界面"""
        print("\n" + "="*50)
        print("      DELTA ROBOT INTERACTIVE CONTROLLER")
        print("="*50)
        
        while True:
            print("\n控制选项:")
            print("1. 移动到位置")
            print("2. 设置关节角度") 
            print("3. 执行轨迹")
            print("4. 获取状态")
            print("5. 停止运动")
            print("6. 退出")
            
            choice = input("\n请选择操作 (1-6): ").strip()
            
            if choice == "1":
                self._handle_position_control()
            elif choice == "2":
                self._handle_joint_control()
            elif choice == "3":
                self._handle_trajectory_control()
            elif choice == "4":
                self._handle_status_query()
            elif choice == "5":
                self.stop_movement()
                print("运动已停止")
            elif choice == "6":
                print("退出控制程序")
                break
            else:
                print("无效选择，请重新输入")
    
    def _handle_position_control(self):
        """处理位置控制"""
        try:
            print("\n输入目标位置 (单位: 米):")
            x = float(input("X: "))
            y = float(input("Y: ")) 
            z = float(input("Z: "))
            speed = float(input("速度 (0.1-2.0, 默认1.0): ") or "1.0")
            
            response = self.move_to_position(x, y, z, speed)
            if response and response.get("status") == "success":
                print("位置命令发送成功")
            else:
                print("位置命令发送失败")
                
        except ValueError:
            print("输入无效，请输入数字")
        except Exception as e:
            print(f"错误: {e}")
    
    def _handle_joint_control(self):
        """处理关节控制"""
        try:
            print("\n输入关节角度 (单位: 弧度):")
            angle1 = float(input("关节1: "))
            angle2 = float(input("关节2: "))
            angle3 = float(input("关节3: "))
            
            response = self.set_joint_angles(angle1, angle2, angle3)
            if response and response.get("status") == "success":
                print("关节角度命令发送成功")
            else:
                print("关节角度命令发送失败")
                
        except ValueError:
            print("输入无效，请输入数字")
        except Exception as e:
            print(f"错误: {e}")
    
    def _handle_trajectory_control(self):
        """处理轨迹控制"""
        print("\n选择轨迹类型:")
        print("1. 圆形轨迹")
        print("2. 正方形轨迹") 
        print("3. 8字形轨迹")
        
        choice = input("请选择轨迹 (1-3): ").strip()
        
        try:
            if choice == "1":
                self.execute_trajectory("circle")
            elif choice == "2":
                self.execute_trajectory("square")
            elif choice == "3":
                self.execute_trajectory("figure8")
            else:
                print("无效选择")
        except Exception as e:
            print(f"轨迹执行错误: {e}")
    
    def _handle_status_query(self):
        """处理状态查询"""
        response = self.get_status()
        if response and response.get("status") == "success":
            data = response.get("data", {})
            print("\n当前状态:")
            print(f"  关节角度: {data.get('joint_angles_deg', [0,0,0])}°")
            print(f"  末端位置: {data.get('ee_position', [0,0,0])}")
            print(f"  运动状态: {'移动中' if data.get('is_moving') else '静止'}")
        else:
            print("获取状态失败")
    
    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()