#!/usr/bin/env python3
# status_monitor_app.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DeltaStatusMonitor import DeltaStatusMonitor

if __name__ == "__main__":
    print("Delta机械臂状态监控器")
    print("=" * 40)
    monitor = DeltaStatusMonitor()
    monitor.run_monitor()