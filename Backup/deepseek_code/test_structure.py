#!/usr/bin/env python3
"""
测试新目录结构
"""
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有模块导入"""
    print("Testing imports...")
    
    modules_to_test = [
        "drone_simulator",
        "drone_simulator.models",
        "drone_simulator.simulation",
        "drone_simulator.controllers",
        "experiments"
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
    
    print("\nAll imports tested!")

if __name__ == "__main__":
    test_imports()