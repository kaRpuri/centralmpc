# Centralized MPC

A centralized Model Predictive Control (MPC) implementation for multi-agent path planning and collision avoidance in 3D space. This project implements a centralized controller that optimizes trajectories for multiple agents simultaneously while avoiding collisions and tracking dynamic goals.

## Overview

This repository contains a centralized MPC approach for coordinating multiple agents in a shared workspace. The controller solves a single large-scale quadratic program (QP) that considers all agents simultaneously, ensuring optimal collision-free trajectories to their respective goals.

## Features

- **Centralized Optimization**: Single QP formulation for all agents
- **Dynamic Goal Tracking**: Support for multiple goal motion patterns:
  - Static goals
  - Circular motion
  - Linear translation
  - Combined circular and translating motion
- **Collision Avoidance**: Built-in constraints to prevent inter-agent collisions
- **3D Path Planning**: Full 3D workspace support with position and velocity constraints
- **Multiple Scenario Support**: Configurable test scenarios with varying complexity
- **Performance Analysis**: Comprehensive trajectory analysis and visualization tools

## Key Components

### Core MPC Controller (`central_mpc.py`)

The main controller implementing:
- State space model with position and velocity states (6-DOF per agent)
- Convex quadratic programming formulation using CVXPY
- Dynamic constraints (position bounds, velocity limits, acceleration limits)
- Cost function with position tracking, velocity tracking, and control effort terms

### Scenario Runner (`run_scenarios_central.py`)

Executes multiple test scenarios with configurable parameters:
- Loads scenario configurations from JSON files
- Converts distributed MPC configs to centralized format
- Runs multiple trials per scenario for statistical analysis
- Saves trajectory and goal data for post-processing

### Scaling Tests (`run_scenario_scale_central.py`)

Evaluates scalability by testing with increasing numbers of agents:
- Tests system performance from 4 to 32 agents
- Measures computation time and success rates
- Generates scaling performance data

### Analysis Tools

Multiple analysis scripts for evaluating performance:
- `analyze_traj_central.py`: Analyzes centralized MPC trajectories
- `analyze_traj_distributed.py`: Compares with distributed approaches
- `analyze_trajectories_all.py`: Comprehensive multi-method comparison
- `analyze_trajectories_scaled_all.py`: Scaling performance analysis
- `analyze_trajectories_base.py`: Base analysis utilities

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- CVXPY
- Matplotlib (for visualization)
- A compatible QP solver (OSQP, ECOS, or similar)

### Setup

```bash
# Clone the repository
git clone https://github.com/kaRpuri/centralmpc.git
cd centralmpc

# Install dependencies
pip install numpy cvxpy matplotlib
```

## Usage

### Running Single Scenarios

```bash
python run_scenarios_central.py
```

This will execute predefined scenarios and save results to the `experiments/` directory.

### Running Scaling Tests

```bash
python run_scenario_scale_central.py
```

Or use the provided shell script:

```bash
bash run_scale_tests.sh
```

### Analyzing Results

After running experiments, analyze the trajectories:

```bash
# Analyze centralized MPC results
python analyze_traj_central.py

# Compare multiple methods
python analyze_trajectories_all.py

# Analyze scaling performance
python analyze_trajectories_scaled_all.py
```

## Configuration

The MPC controller accepts the following configuration parameters:

```python
config = {
    "num_agents": 8,           # Number of agents
    "horizon": 30,             # MPC prediction horizon
    "dt": 0.01,                # Control timestep (seconds)
    "pos_min": [-10, -10, 0],  # Workspace minimum bounds
    "pos_max": [10, 10, 5],    # Workspace maximum bounds
    "vel_max": 1.5,            # Maximum velocity (m/s)
    "acc_max": 1.0,            # Maximum acceleration (m/s²)
    "Q_pos": 50.0,             # Position tracking weight
    "Q_vel": 1.0,              # Velocity tracking weight
    "R": 0.1,                  # Control effort weight
    "Q_terminal": 20.0,        # Terminal cost weight
    "goal_tolerance": 0.05,    # Goal reached threshold (meters)
    "motion_type": "static"    # Goal motion pattern
}
```

## Project Structure

```
centralmpc/
├── central_mpc.py                      # Core MPC controller
├── run_scenarios_central.py            # Scenario execution script
├── run_scenario_scale_central.py       # Scaling test script
├── run_scale_tests.sh                  # Shell script for scaling tests
├── analyze_traj_central.py             # Centralized trajectory analysis
├── analyze_traj_distributed.py         # Distributed comparison analysis
├── analyze_trajectories_all.py         # Multi-method comparison
├── analyze_trajectories_scaled_all.py  # Scaling analysis
├── analyze_trajectories_base.py        # Base analysis utilities
├── experiments/                        # Experimental results directory
├── results/                            # Additional results storage
├── results_central/                    # Centralized MPC results
├── goals.txt                           # Goal position data
└── trajectories.txt                    # Trajectory data
```

## Performance Metrics

The analysis tools compute the following metrics:

- **Tracking Error**: Position error from desired goals over time
- **Collision Count**: Number of inter-agent collisions
- **Minimum Distance**: Closest approach between agents
- **Computation Time**: QP solve time per iteration
- **Success Rate**: Percentage of agents reaching goals

## License

This project is available for academic and research purposes.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
