from uav_project.controllers.cascade_controller import CascadeController
from uav_project.controllers.delta_controller import DeltaController

class CombinedController:
    """
    Combines the UAV CascadeController and the DeltaController.
    """
    def __init__(self, uav_model):
        self.uav_controller = CascadeController(uav_model)
        self.uav = uav_model
        
        # Only initialize DeltaController if the model has a Delta arm
        if getattr(self.uav, 'has_delta', False):
            self.delta_controller = DeltaController(uav_model)
        else:
            self.delta_controller = None

    def update(self, sim_time: float) -> None:
        """
        Updates both controllers.
        
        Args:
            sim_time (float): Current simulation time.
        """
        self.uav_controller.update(sim_time)
        if self.delta_controller is not None:
            self.delta_controller.update(sim_time)

    def set_target_position(self, pos: list) -> None:
        """
        Sets target for UAV controller.
        Delta controller has its own internal trajectory generator.
        
        Args:
            pos (list): Target position [x, y, z]
        """
        self.uav_controller.set_target_position(pos)

    def get_log_data(self):
        """
        Returns log data.
        """
        uav_data = self.uav_controller.get_log_data()
        if self.delta_controller is not None:
            delta_des = self.delta_controller.current_des_pos_log
            delta_act = self.delta_controller.current_actual_pos_log
            return uav_data + (delta_des, delta_act)
        else:
            return uav_data + (None, None)

    def print_state(self):
        """
        Prints the state of the system.
        """
        self.uav_controller.print_state()
