#!/usr/bin/env python3
# simulator_app.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DeltaSimulator import DeltaSimulator
from DeltaVisualizer import DeltaVisualizer
import multiprocessing as mp
import time

def run_simulator():
    simulator = DeltaSimulator("scene.xml")
    simulator.start_control_server('localhost', 12347)
    simulator.start_viz_server('localhost', 12346)
    simulator.start_status_server('localhost', 12348)
    simulator.run_simulation()

def run_visualizer():
    visualizer = DeltaVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    print("Delta机械臂仿真器")
    print("=" * 40)
    
    processes = []
    
    sim_process = mp.Process(target=run_simulator)
    processes.append(sim_process)
    sim_process.start()
    
    time.sleep(3)
    
    viz_process = mp.Process(target=run_visualizer)
    processes.append(viz_process)
    viz_process.start()
    
    # 等待进程结束
    for process in processes:
        if process.is_alive():
            process.join()