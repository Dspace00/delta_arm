# DeltaSimulator.py
import mujoco
import mujoco.viewer
import numpy as np
import socket
import json
import threading
import time

from DeltaController import DeltaController
from DeltaRobotModel import DeltaRobotModel

class DeltaSimulator:
    """Delta机械臂MuJoCo仿真器"""
    
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化组件
        self.robot_model = DeltaRobotModel()
        self.controller = DeltaController(self.robot_model)
        
        # 获取模型索引
        self._setup_model_indices()
        
        # 控制状态
        self.current_trajectory = None
        self.is_moving = False
        self.move_start_time = 0
        
        # Socket通信
        self.control_socket = None
        self.viz_socket = None
        self.status_socket = None
        self.running = True
        
        # 状态数据
        self.status_data = {
            'joint_angles': [0, 0, 0],
            'joint_angles_deg': [0, 0, 0],
            'ee_position': [0, 0, 0],
            'target_angles': [0, 0, 0],
            'target_position': [0, 0, 0],
            'is_moving': False
        }
    
    def _setup_model_indices(self):
        """设置模型组件索引"""
        # 执行器索引
        self.motor_actuator_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor1'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor2'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor3')
        ]
        
        # 关节索引
        self.motor_joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'motor_arm_joint1'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'motor_arm_joint2'), 
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'motor_arm_joint3')
        ]
        
        # 关节位置地址
        self.motor_qpos_adr = [self.model.joint(i).qposadr[0] for i in self.motor_joint_indices]
        
        # 末端平台body ID
        self.end_platform_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'end_platform')
    
    def get_joint_angles(self):
        """获取当前关节角度"""
        angles = []
        for adr in self.motor_qpos_adr:
            angles.append(self.data.qpos[adr])
        return np.array(angles)
    
    def set_joint_angles(self, angles):
        """设置关节角度控制信号"""
        for i, angle in enumerate(angles):
            if i < len(self.motor_actuator_indices):
                self.data.ctrl[self.motor_actuator_indices[i]] = angle
    
    def get_end_effector_position(self):
        """获取末端执行器位置"""
        return self.data.xpos[self.end_platform_id].copy()
    
    def update_control(self):
        """更新控制信号"""
        if not self.is_moving or self.current_trajectory is None:
            return
        
        elapsed_time = self.data.time - self.move_start_time
        
        if elapsed_time >= self.current_trajectory['duration']:
            # 移动完成
            self.set_joint_angles(self.current_trajectory['target_angles'])
            self.is_moving = False
            self.current_trajectory = None
            print("移动完成")
            return
        
        # 获取当前目标角度
        current_target_angles = self.controller.get_current_target_angles(
            self.current_trajectory, elapsed_time
        )
        self.set_joint_angles(current_target_angles)
    
    def move_to_position(self, target_pos, speed=1.0):
        """移动到目标位置"""
        current_angles = self.get_joint_angles()
        trajectory = self.controller.move_to_position(target_pos, current_angles, speed)
        
        if trajectory is not None:
            self.current_trajectory = trajectory
            self.move_start_time = self.data.time
            self.is_moving = True
            return True
        return False
    
    def set_direct_joint_control(self, joint_angles):
        """直接设置关节角度控制"""
        self.is_moving = False
        self.current_trajectory = None
        self.set_joint_angles(joint_angles)
        return True
    
    def update_status_data(self):
        """更新状态数据"""
        self.status_data['joint_angles'] = self.get_joint_angles().tolist()
        self.status_data['joint_angles_deg'] = np.degrees(self.get_joint_angles()).tolist()
        self.status_data['ee_position'] = self.get_end_effector_position().tolist()
        self.status_data['is_moving'] = self.is_moving
        
        if self.is_moving and self.current_trajectory is not None:
            self.status_data['target_angles'] = self.current_trajectory['target_angles']
            self.status_data['target_position'] = self.current_trajectory['target_position']
        else:
            self.status_data['target_angles'] = self.get_joint_angles().tolist()
            self.status_data['target_position'] = self.get_end_effector_position().tolist()
    
    def start_control_server(self, host='localhost', port=12347):
        """启动控制服务器"""
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.bind((host, port))
        self.control_socket.listen(5)
        
        print(f"控制服务器已启动在 {host}:{port}")
        
        control_thread = threading.Thread(target=self._handle_control_connections)
        control_thread.daemon = True
        control_thread.start()
    
    def _handle_control_connections(self):
        """处理控制连接"""
        while self.running:
            try:
                conn, addr = self.control_socket.accept()
                print(f"控制客户端连接: {addr}")
                
                client_thread = threading.Thread(
                    target=self._handle_control_messages, 
                    args=(conn,)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"控制连接错误: {e}")
    
    def _handle_control_messages(self, conn):
        """处理控制消息"""
        try:
            while self.running:
                data = b""
                while True:
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in chunk:
                        break
                
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode().strip())
                    response = self._process_control_command(message)
                    conn.sendall((json.dumps(response) + "\n").encode())
                    
                except json.JSONDecodeError:
                    response = {"status": "error", "message": "JSON解析错误"}
                    conn.sendall((json.dumps(response) + "\n").encode())
                    
        except Exception as e:
            print(f"控制消息处理错误: {e}")
        finally:
            conn.close()
    
    def _process_control_command(self, command):
        """处理控制命令"""
        cmd_type = command.get("command")
        
        if cmd_type == "move_to":
            target_pos = command.get("target_position")
            speed = command.get("speed", 1.0)
            success = self.move_to_position(target_pos, speed)
            return {"status": "success" if success else "error"}
        
        elif cmd_type == "set_joint_angles":
            joint_angles = command.get("joint_angles")
            success = self.set_direct_joint_control(joint_angles)
            return {"status": "success" if success else "error"}
        
        elif cmd_type == "get_status":
            self.update_status_data()
            return {"status": "success", "data": self.status_data}
        
        elif cmd_type == "stop":
            self.is_moving = False
            self.current_trajectory = None
            return {"status": "stopped"}
        
        elif cmd_type == "trajectory_circle":
            center = command.get("center", [0, 0, -0.15])
            radius = command.get("radius", 0.05)
            speed = command.get("speed", 1.0)
            duration = command.get("duration", 10.0)
            
            points = self.controller.trajectory_circle(center, radius, speed, duration)
            # 这里可以执行轨迹跟踪
            return {"status": "success", "points": points}
        
        else:
            return {"status": "error", "message": "未知命令"}
    
    def start_viz_server(self, host='localhost', port=12346):
        """启动可视化服务器"""
        self.viz_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.viz_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.viz_socket.bind((host, port))
        self.viz_socket.listen(5)
        
        print(f"可视化服务器已启动在 {host}:{port}")
        
        viz_thread = threading.Thread(target=self._handle_viz_connections)
        viz_thread.daemon = True
        viz_thread.start()
    
    def _handle_viz_connections(self):
        """处理可视化连接"""
        while self.running:
            try:
                conn, addr = self.viz_socket.accept()
                print(f"可视化客户端连接: {addr}")
                
                while self.running:
                    self.update_status_data()
                    
                    viz_data = {
                        "timestamp": time.time(),
                        "position": self.status_data['ee_position'],
                        "target_position": self.status_data['target_position'],
                        "is_moving": self.status_data['is_moving']
                    }
                    
                    try:
                        conn.sendall((json.dumps(viz_data) + "\n").encode())
                    except:
                        print("可视化连接断开")
                        break
                    
                    time.sleep(0.05)  # 20Hz 更新频率
                
                conn.close()
                
            except Exception as e:
                if self.running:
                    print(f"可视化连接错误: {e}")
    
    def start_status_server(self, host='localhost', port=12348):
        """启动状态服务器"""
        self.status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.status_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.status_socket.bind((host, port))
        self.status_socket.listen(5)
        
        print(f"状态服务器已启动在 {host}:{port}")
        
        status_thread = threading.Thread(target=self._handle_status_connections)
        status_thread.daemon = True
        status_thread.start()
    
    def _handle_status_connections(self):
        """处理状态连接"""
        while self.running:
            try:
                conn, addr = self.status_socket.accept()
                print(f"状态客户端连接: {addr}")
                
                while self.running:
                    self.update_status_data()
                    
                    try:
                        conn.sendall((json.dumps(self.status_data) + "\n").encode())
                    except:
                        print("状态连接断开")
                        break
                    
                    time.sleep(0.1)  # 10Hz 更新频率
                
                conn.close()
                
            except Exception as e:
                if self.running:
                    print(f"状态连接错误: {e}")
    
    def run_simulation(self):
        """运行仿真主循环"""
        print("启动Delta机械臂仿真...")
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                print("MuJoCo查看器已启动")
                
                # 仿真循环
                while viewer.is_running and self.running:
                    step_start = time.time()
                    
                    # 更新控制
                    self.update_control()
                    
                    # 执行仿真步
                    mujoco.mj_step(self.model, self.data)
                    
                    # 同步查看器
                    viewer.sync()
                    
                    # 控制仿真速度
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                    
        except Exception as e:
            print(f"仿真错误: {e}")
        finally:
            self.running = False
            print("仿真已结束")
    
    def shutdown(self):
        """关闭仿真器"""
        self.running = False
        if self.control_socket:
            self.control_socket.close()
        if self.viz_socket:
            self.viz_socket.close()
        if self.status_socket:
            self.status_socket.close()