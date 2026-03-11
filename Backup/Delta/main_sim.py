# main_sim.py (主程序入口)
import multiprocessing as mp
import time
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DeltaSimulator import DeltaSimulator
from DeltaVisualizer import DeltaVisualizer
from DeltaStatusMonitor import DeltaStatusMonitor
from DeltaCommandClient import DeltaCommandClient

def run_simulator():
    """运行仿真器进程"""
    simulator = DeltaSimulator("scene.xml")
    
    # 启动服务器
    simulator.start_control_server('localhost', 12347)
    simulator.start_viz_server('localhost', 12346)
    simulator.start_status_server('localhost', 12348)
    
    # 运行仿真
    simulator.run_simulation()

def run_visualizer():
    """运行可视化进程"""
    visualizer = DeltaVisualizer()
    visualizer.run_visualization()

def run_status_monitor():
    """运行状态监控进程"""
    monitor = DeltaStatusMonitor()
    monitor.run_monitor()

def run_command_client():
    """运行命令客户端进程"""
    client = DeltaCommandClient()
    if client.connect():
        client.interactive_control()
    client.close()

if __name__ == "__main__":
    print("Delta机械臂仿真系统启动中...")
    
    # 创建进程
    processes = []
    
    try:
        # 启动仿真器进程
        print("启动仿真器进程...")
        sim_process = mp.Process(target=run_simulator)
        processes.append(sim_process)
        sim_process.start()
        
        time.sleep(2)  # 等待仿真器启动
        
        # 启动可视化进程
        print("启动可视化进程...")
        viz_process = mp.Process(target=run_visualizer)
        processes.append(viz_process)
        viz_process.start()
        
        time.sleep(1)
        
        # 启动状态监控进程
        print("启动状态监控进程...")
        monitor_process = mp.Process(target=run_status_monitor)
        processes.append(monitor_process)
        monitor_process.start()
        
        time.sleep(1)
        
        # 在主进程运行命令客户端（这样可以接收键盘输入）
        print("启动命令客户端...")
        run_command_client()
        
    except KeyboardInterrupt:
        print("\n系统关闭中...")
    except Exception as e:
        print(f"系统错误: {e}")
    finally:
        # 终止所有进程
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        
        print("Delta机械臂仿真系统已关闭")