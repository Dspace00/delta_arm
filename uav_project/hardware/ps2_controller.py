"""
PS2 2.4G Wireless Gamepad Controller.

Provides a driver interface for reading PS2-style wireless gamepad inputs
and mapping them to Delta robot workspace coordinates.

Hardware Info:
    - Connection: 2.4G USB receiver
    - Device Name: USB WirelessGamepad
    - Library: pygame.joystick (cross-platform HID support)
"""

import os
import sys
import time
import numpy as np

# pygame is required for joystick support
try:
    import pygame
    import pygame.joystick
except ImportError:
    raise ImportError(
        "pygame is required for joystick support. "
        "Install with: pip install pygame"
    )


class PS2Controller:
    """
    PS2 2.4G Wireless Gamepad Controller.
    
    Reads joystick and button inputs from a PS2-style wireless gamepad
    and maps them to Delta robot workspace coordinates.
    
    Control Mapping:
        - Left Stick X/Y: Control end-effector X/Y position
        - Right Stick Y: Control end-effector Z position
        - START: Exit simulation
        - SELECT: Reset to center position
    
    Example:
        >>> controller = PS2Controller()
        >>> controller.connect()
        >>> while running:
        ...     controller.read_input()
        ...     target_pos = controller.get_position()
        ...     if controller.get_button('start'):
        ...         break
        >>> controller.close()
    """
    
    # Button mapping for standard PS2 gamepad
    # Note: Actual indices may vary depending on the device
    BUTTON_MAP = {
        'cross': 0,      # X button (bottom)
        'circle': 1,     # O button (right)
        'square': 2,     # □ button (left)
        'triangle': 3,   # △ button (top)
        'L1': 4,
        'R1': 5,
        'L2': 6,
        'R2': 7,
        'select': 8,
        'start': 9,
        'L3': 10,        # Left stick press
        'R3': 11,        # Right stick press
    }
    
    # Axis mapping for standard PS2 gamepad
    # Note: Actual indices may vary depending on the device
    AXIS_MAP = {
        'left_x': 0,     # Left stick horizontal (-1: left, +1: right)
        'left_y': 1,     # Left stick vertical (-1: up, +1: down)
        'right_x': 2,    # Right stick horizontal
        'right_y': 3,    # Right stick vertical
        'L2': 4,         # L2 trigger (-1: released, +1: fully pressed)
        'R2': 5,         # R2 trigger
    }
    
    def __init__(self, 
                 joystick_id=0,
                 workspace_radius=0.10,
                 z_center=-0.15,
                 z_range=0.05,
                 deadzone=0.15,
                 invert_y=True):
        """
        Initialize PS2 Controller.
        
        Args:
            joystick_id: Joystick index (default 0 for first connected).
            workspace_radius: Maximum XY workspace radius in meters.
            z_center: Center Z position relative to base in meters.
            z_range: Z position range (±) from center in meters.
            deadzone: Joystick deadzone threshold (0.0 to 1.0).
            invert_y: If True, invert Y axis (common for gamepad conventions).
        """
        # Joystick settings
        self.joystick_id = joystick_id
        self.joystick = None
        self.connected = False
        
        # Workspace mapping
        self.workspace_radius = workspace_radius
        self.z_center = z_center
        self.z_range = z_range
        
        # Input processing
        self.deadzone = deadzone
        self.invert_y = invert_y
        
        # Current state
        self._position = np.array([0.0, 0.0, z_center])  # [x, y, z]
        self._axes = np.zeros(6)
        self._buttons = np.zeros(16, dtype=bool)  # PS2 has up to 16 buttons
        
        # Calibration offset (for drift correction)
        self._axis_offset = np.zeros(6)
        
    def connect(self):
        """
        Connect to the PS2 gamepad.
        
        Initializes pygame and connects to the specified joystick.
        
        Returns:
            bool: True if connection successful.
        
        Raises:
            RuntimeError: If no joystick is found.
        """
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        # Check for connected joysticks
        num_joysticks = pygame.joystick.get_count()
        if num_joysticks == 0:
            raise RuntimeError(
                "No joystick found! Please connect your PS2 gamepad receiver."
            )
        
        print(f"Found {num_joysticks} joystick(s)")
        
        # Connect to specified joystick
        self.joystick = pygame.joystick.Joystick(self.joystick_id)
        self.joystick.init()
        
        # Print joystick info
        print(f"Connected to: {self.joystick.get_name()}")
        print(f"  Axes: {self.joystick.get_numaxes()}")
        print(f"  Buttons: {self.joystick.get_numbuttons()}")
        
        self.connected = True
        return True
    
    def calibrate(self, samples=100):
        """
        Calibrate joystick axes to correct for drift.
        
        Should be called with joysticks at rest position.
        
        Args:
            samples: Number of samples to average for calibration.
        """
        if not self.connected:
            raise RuntimeError("Joystick not connected. Call connect() first.")
        
        print("Calibrating... Keep joysticks at rest position.")
        
        axis_sums = np.zeros(6)
        
        for _ in range(samples):
            pygame.event.pump()
            for i in range(min(6, self.joystick.get_numaxes())):
                axis_sums[i] += self.joystick.get_axis(i)
            time.sleep(0.001)
        
        self._axis_offset = axis_sums / samples
        print(f"Calibration complete. Offsets: {self._axis_offset}")
    
    def read_input(self):
        """
        Read current joystick and button state.
        
        Should be called once per frame in the main loop.
        Updates internal state and computes target position.
        """
        if not self.connected:
            return
        
        # Process pygame events (required for joystick updates)
        pygame.event.pump()
        
        # Read axes
        for i in range(min(6, self.joystick.get_numaxes())):
            raw_value = self.joystick.get_axis(i)
            # Apply calibration offset
            value = raw_value - self._axis_offset[i]
            # Apply deadzone
            if abs(value) < self.deadzone:
                value = 0.0
            self._axes[i] = value
        
        # Read buttons
        for i in range(self.joystick.get_numbuttons()):
            self._buttons[i] = bool(self.joystick.get_button(i))
        
        # Compute target position from joystick axes
        self._update_position()
    
    def _update_position(self):
        """
        Update target position based on joystick axes.
        
        Mapping:
            - Left stick X: X axis (-workspace_radius to +workspace_radius)
            - Left stick Y: Y axis (-workspace_radius to +workspace_radius)
            - Right stick Y: Z axis (z_center ± z_range)
        """
        # Left stick controls X/Y
        x_input = self._axes[self.AXIS_MAP['left_x']]
        y_input = self._axes[self.AXIS_MAP['left_y']]
        
        # Apply Y inversion if needed (some gamepads have inverted Y)
        if self.invert_y:
            y_input = -y_input
        
        # Map to workspace
        x = x_input * self.workspace_radius
        y = y_input * self.workspace_radius
        
        # Right stick controls Z
        z_input = -self._axes[self.AXIS_MAP['right_y']]  # Invert: up = higher Z
        z = self.z_center + z_input * self.z_range
        
        self._position = np.array([x, y, z])
    
    def get_position(self):
        """
        Get current target position.
        
        Returns:
            np.ndarray: Target position [x, y, z] relative to base in meters.
        """
        return self._position.copy()
    
    def get_axis(self, axis_name):
        """
        Get raw axis value by name.
        
        Args:
            axis_name: Name of axis (e.g., 'left_x', 'right_y').
        
        Returns:
            float: Axis value (-1.0 to 1.0).
        """
        if axis_name in self.AXIS_MAP:
            return self._axes[self.AXIS_MAP[axis_name]]
        return 0.0
    
    def get_button(self, button_name):
        """
        Get button state by name.
        
        Args:
            button_name: Name of button (e.g., 'start', 'select', 'cross').
        
        Returns:
            bool: True if button is pressed.
        """
        if button_name in self.BUTTON_MAP:
            idx = self.BUTTON_MAP[button_name]
            if idx < len(self._buttons):
                return self._buttons[idx]
        return False
    
    def get_all_buttons(self):
        """
        Get all button states.
        
        Returns:
            np.ndarray: Boolean array of button states.
        """
        return self._buttons.copy()
    
    def get_all_axes(self):
        """
        Get all axis values.
        
        Returns:
            np.ndarray: Float array of axis values.
        """
        return self._axes.copy()
    
    def reset_position(self):
        """
        Reset target position to center.
        """
        self._position = np.array([0.0, 0.0, self.z_center])
        print(f"Position reset to center: {self._position}")
    
    def close(self):
        """
        Disconnect and cleanup.
        """
        if self.joystick:
            self.joystick.quit()
        pygame.joystick.quit()
        pygame.quit()
        self.connected = False
        print("Joystick disconnected.")
    
    def print_state(self):
        """
        Print current state for debugging.
        """
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        
        print("=== PS2 Controller State ===")
        print(f"Position: {self._position}")
        print(f"Left Stick:  X={self._axes[0]: .3f} Y={self._axes[1]: .3f}")
        print(f"Right Stick: X={self._axes[2]: .3f} Y={self._axes[3]: .3f}")
        
        pressed_buttons = [name for name, idx in self.BUTTON_MAP.items() 
                          if idx < len(self._buttons) and self._buttons[idx]]
        if pressed_buttons:
            print(f"Pressed: {', '.join(pressed_buttons)}")
        print("============================")
