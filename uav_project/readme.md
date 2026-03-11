# UAV Project - Delta Sim

This folder contains the main codebase for the UAV Delta Robot simulation project. It is the foundation for ongoing research.

## Environment Setup

This project uses **Conda** for environment management. Follow the steps below to set up the environment on your machine.

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.
- Git installed.

### Installation Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Zang153/Project_uav.git
    cd Project_uav
    ```

2.  **Create Conda Environment**

    Create a new environment named `delta_sim` with Python 3.11:

    ```bash
    conda create -n delta_sim python=3.11 -y
    ```

3.  **Activate Environment**

    ```bash
    conda activate delta_sim
    ```

    > **Note for Windows PowerShell Users:**
    > If you encounter errors activating the environment, you may need to initialize conda for PowerShell and set the execution policy:
    > ```powershell
    > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    > conda init powershell
    > ```
    > Restart your terminal after running these commands.

4.  **Install Dependencies**

    Install the required packages. Note that `mujoco` is installed via `conda-forge` to ensure compatibility on Windows.

    ```bash
    # Install core scientific packages
    conda install numpy scipy matplotlib -y
    
    # Install MuJoCo from conda-forge (Recommended for Windows)
    conda install -c conda-forge mujoco -y
    
    # Install other dependencies via pip
    pip install numpy-quaternion
    ```

### Running the Simulation

To run the main simulation script:

```bash
cd uav_project
python main.py
```

## Project Structure

- `controllers/`: Control algorithms (PID, Cascade, etc.)
- `meshes/`: STL files and XML models for the robot.
- `models/`: Robot model definitions and mixer logic.
- `simulation/`: MuJoCo simulator interface.
- `utils/`: Helper functions, kinematics, and logging.
- `main.py`: Entry point for the simulation.
