#!/usr/bin/env python3
# controller_app.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DeltaCommandClient import DeltaCommandClient

if __name__ == "__main__":
    print("Delta机械臂控制器")
    print("=" * 40)
    client = DeltaCommandClient()
    if client.connect():
        client.interactive_control()
    client.close()