"""
模块功能说明：
可视化模块，负责数据记录和绘图。

重要功能：
- update: 记录每一步的仿真数据
- plot: 仿真结束后绘制曲线图 (自动处理macOS兼容性)
"""

import numpy as np
import matplotlib
import platform
import os

# Cross-platform check
if platform.system() == 'Darwin':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        self.history = {
            'time': [],
            'position': [],
            'target_position': [],
            'velocity': [],
            'target_velocity': [],
            'attitude': [], # Euler
            'target_attitude': [], # Euler
            'rate': [],
            'target_rate': [],
            'force': [],
            'torque': []
        }
        
    def update(self, time, pos, target_pos, vel, target_vel, 
               att_euler, target_att_euler, rate, target_rate, 
               force, torque):
        self.history['time'].append(time)
        self.history['position'].append(pos)
        self.history['target_position'].append(target_pos)
        self.history['velocity'].append(vel)
        self.history['target_velocity'].append(target_vel)
        self.history['attitude'].append(att_euler)
        self.history['target_attitude'].append(target_att_euler)
        self.history['rate'].append(rate)
        self.history['target_rate'].append(target_rate)
        self.history['force'].append(force)
        self.history['torque'].append(torque)

    def plot(self, save_path='simulation_results.png'):
        # Convert lists to arrays
        for key in self.history:
            self.history[key] = np.array(self.history[key])
            
        time = self.history['time']
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        
        # 1. Position
        ax = axes[0, 0]
        ax.plot(time, self.history['position'][:,0], 'r', label='x')
        ax.plot(time, self.history['target_position'][:,0], 'r--', label='ref')
        ax.plot(time, self.history['position'][:,1], 'g', label='y')
        ax.plot(time, self.history['target_position'][:,1], 'g--', label='ref')
        ax.plot(time, self.history['position'][:,2], 'b', label='z')
        ax.plot(time, self.history['target_position'][:,2], 'b--', label='ref')
        ax.set_title('Position')
        ax.legend()
        ax.grid(True)
        
        # 2. Velocity
        ax = axes[1, 0]
        ax.plot(time, self.history['velocity'][:,0], 'r', label='vx')
        ax.plot(time, self.history['target_velocity'][:,0], 'r--', label='ref')
        ax.plot(time, self.history['velocity'][:,1], 'g', label='vy')
        ax.plot(time, self.history['target_velocity'][:,1], 'g--', label='ref')
        ax.plot(time, self.history['velocity'][:,2], 'b', label='vz')
        ax.plot(time, self.history['target_velocity'][:,2], 'b--', label='ref')
        ax.set_title('Velocity')
        ax.grid(True)
        
        # 3. Attitude (Euler)
        ax = axes[2, 0]
        ax.plot(time, self.history['attitude'][:,0], 'r', label='roll')
        ax.plot(time, self.history['target_attitude'][:,0], 'r--', label='ref')
        ax.plot(time, self.history['attitude'][:,1], 'g', label='pitch')
        ax.plot(time, self.history['target_attitude'][:,1], 'g--', label='ref')
        ax.plot(time, self.history['attitude'][:,2], 'b', label='yaw')
        ax.plot(time, self.history['target_attitude'][:,2], 'b--', label='ref')
        ax.set_title('Attitude (Euler deg)')
        ax.grid(True)
        
        # 4. Rate
        ax = axes[3, 0]
        ax.plot(time, self.history['rate'][:,0], 'r', label='p')
        ax.plot(time, self.history['target_rate'][:,0], 'r--', label='ref')
        ax.plot(time, self.history['rate'][:,1], 'g', label='q')
        ax.plot(time, self.history['target_rate'][:,1], 'g--', label='ref')
        ax.plot(time, self.history['rate'][:,2], 'b', label='r')
        ax.plot(time, self.history['target_rate'][:,2], 'b--', label='ref')
        ax.set_title('Angular Rate')
        ax.grid(True)
        
        # 5. Force (Target Force in World Frame)
        ax = axes[0, 1]
        ax.plot(time, self.history['force'][:,0], 'r', label='Fx')
        ax.plot(time, self.history['force'][:,1], 'g', label='Fy')
        ax.plot(time, self.history['force'][:,2], 'b', label='Fz')
        ax.set_title('Force Command (World)')
        ax.legend()
        ax.grid(True)
        
        # 6. Torque
        ax = axes[1, 1]
        ax.plot(time, self.history['torque'][:,0], 'r', label='Mx')
        ax.plot(time, self.history['torque'][:,1], 'g', label='My')
        ax.plot(time, self.history['torque'][:,2], 'b', label='Mz')
        ax.set_title('Torque Command')
        ax.legend()
        ax.grid(True)
        
        # 3D Plot
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        ax.plot(self.history['position'][:,0], self.history['position'][:,1], self.history['position'][:,2], label='Path')
        ax.set_title('3D Trajectory')
        
        plt.tight_layout()
        
        # Cross-platform logic
        if platform.system() == 'Darwin':
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            try:
                plt.show()
            except:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
