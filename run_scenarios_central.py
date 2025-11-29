"""
Centralized MPC Scenario Runner
Runs all 9 scenarios from ../online_dmpc/cpp/config/ using the centralized MPC approach
"""

import os
import json
import numpy as np
from typing import Dict, Any
import time

# Parameters
RUNS = range(1, 4)  # 1 through 3 runs per scenario

# Import classes from central_mpc.py
from central_mpc import CentralizedMPC, GoalManager, Simulator


def load_scenario_config(scenario_num: int) -> Dict[str, Any]:
    """Load scenario configuration from JSON file"""
    config_path = f"../online_dmpc/cpp/config/scenario_{scenario_num}.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Scenario config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def convert_config_for_central_mpc(scenario_config: Dict[str, Any], scenario_num: int) -> Dict[str, Any]:
    """Convert scenario config to format expected by central MPC"""
    
    # Determine motion type based on scenario number
    if scenario_num <= 6:
        motion_type = "static"
    elif scenario_num == 7:
        motion_type = "translating" 
    elif scenario_num == 8:
        motion_type = "circular"
    elif scenario_num == 9:
        motion_type = "circular_translating"
    else:
        motion_type = "static"  # default fallback
    
    # Create central MPC config
    central_config = {
        # Basic parameters
        "num_agents": scenario_config["N"],
        "dim": scenario_config["dim"],
        "horizon": 30,  # MPC horizon (can be adjusted)
        "dt": scenario_config["ts"],  # Control timestep
        "dt_plan": scenario_config["h"],  # Planning timestep
        "sim_duration": scenario_config["simulation_duration"],
        
        # Workspace bounds
        "pos_min": scenario_config["pmin"],
        "pos_max": scenario_config["pmax"],
        "vel_max": 1.5,  # Safe velocity limit
        "acc_max": 1.0,  # Conservative acceleration limit (from amax bounds)
        
        # Cost function weights (tuned for good performance)
        "Q_pos": 50.0,
        "Q_vel": 1.0,
        "R": 0.1,
        "Q_terminal": 20.0,
        
        # Goal tracking tolerance
        "goal_tolerance": scenario_config["goal_tolerance"],
        "noise_std": scenario_config.get("std_position", 0.001),
        
        # Goal motion parameters
        "motion_type": motion_type,
        "circular_radius": scenario_config.get("goal_circular_radius", 2.0),
        "circular_omega": scenario_config.get("goal_circular_omega", 0.3),
        "translation_velocity": scenario_config.get("goal_translation_velocity", 0.2),
        
        # Initial and goal positions
        "initial_positions": scenario_config["po"],
        "goal_positions": scenario_config["pf"]
    }
    
    return central_config


def run_scenario_single_run(scenario_num: int, run_num: int) -> bool:
    """Run a single run of a scenario with centralized MPC"""
    
    print(f"\n{'='*15} SCENARIO {scenario_num} - RUN {run_num} {'='*15}")
    
    try:
        # Load scenario configuration
        scenario_config = load_scenario_config(scenario_num)
        print(f"✓ Loaded config for scenario {scenario_num}, run {run_num}")
        
        # Convert to central MPC format
        central_config = convert_config_for_central_mpc(scenario_config, scenario_num)
        motion_type = central_config["motion_type"]
        
        # Add run-specific variations for statistical validity
        # Slight variations in noise, initial conditions, or goal tolerance
        np.random.seed(42 + scenario_num * 100 + run_num)  # Reproducible but different seeds
        
        # Add small noise to initial positions (±5cm)
        initial_pos = np.array(central_config["initial_positions"])
        noise_scale = 0.05  # 5cm standard deviation
        position_noise = np.random.normal(0, noise_scale, initial_pos.shape)
        central_config["initial_positions"] = (initial_pos + position_noise).tolist()
        
        # Vary noise level slightly between runs
        base_noise = central_config.get("noise_std", 0.001)
        noise_variations = [0.8, 1.0, 1.2]  # 80%, 100%, 120% of base noise
        central_config["noise_std"] = base_noise * noise_variations[run_num - 1]
        
        # Vary goal tolerance slightly
        base_tolerance = central_config.get("goal_tolerance", 0.15)
        tolerance_variations = [0.9, 1.0, 1.1]  # 90%, 100%, 110% of base tolerance
        central_config["goal_tolerance"] = base_tolerance * tolerance_variations[run_num - 1]
        
        print(f"  Motion type: {motion_type}")
        print(f"  Agents: {central_config['num_agents']}")
        print(f"  Duration: {central_config['sim_duration']}s")
        print(f"  Noise std: {central_config['noise_std']:.6f}")
        print(f"  Goal tolerance: {central_config['goal_tolerance']:.3f}")
        
        # Create output directory inside experiments folder
        experiments_dir = "experiments"
        os.makedirs(experiments_dir, exist_ok=True)
        scenario_dir = os.path.join(experiments_dir, f"scenario_{scenario_num}_central")
        os.makedirs(scenario_dir, exist_ok=True)
        output_dir = os.path.join(scenario_dir, f"run_{run_num}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run simulation
        print(f"  Running centralized MPC simulation...")
        start_time = time.time()
        
        sim = Simulator(central_config)
        sim.simulate()
        
        # Save results
        traj_path = os.path.join(output_dir, 'trajectories.txt')
        goals_path = os.path.join(output_dir, 'goals.txt')
        
        sim.save_trajectories(traj_path)
        sim.save_goals(goals_path)
        
        # Save configuration for reference
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(central_config, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"  ✓ Scenario {scenario_num}, run {run_num} completed in {elapsed:.1f}s")
        print(f"    Results saved to: {output_dir}/")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in scenario {scenario_num}, run {run_num}: {e}")
        return False


def run_scenario(scenario_num: int) -> bool:
    """Run all runs for a single scenario with centralized MPC"""
    
    print(f"\n{'='*20} SCENARIO {scenario_num} {'='*20}")
    
    successful_runs = 0
    total_runs = len(RUNS)
    
    for run_num in RUNS:
        if run_scenario_single_run(scenario_num, run_num):
            successful_runs += 1
    
    success_rate = successful_runs / total_runs
    print(f"\n✓ Scenario {scenario_num} summary: {successful_runs}/{total_runs} runs successful ({success_rate:.1%})")
    
    return successful_runs > 0  # Consider scenario successful if at least one run succeeded


def main():
    """Run all 9 scenarios with centralized MPC"""
    
    print("\n" + "="*80)
    print("CENTRALIZED MPC - MULTI-SCENARIO ANALYSIS")
    print("="*80)
    print("Running 9 scenarios with different motion patterns:")
    print("  Scenarios 1-6: Static goals")
    print("  Scenario 7:    Translating goals") 
    print("  Scenario 8:    Circular goals")
    print("  Scenario 9:    Circular + Translating goals")
    print(f"Each scenario will be run {len(RUNS)} times for statistical validity")
    print("="*80)
    
    scenarios = list(range(1, 10))  # Scenarios 1 through 9
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for scenario_num in scenarios:
        if run_scenario(scenario_num):
            successful += 1
        else:
            failed += 1
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("SCENARIO ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total runtime: {total_time:.1f}s")
    print()
    
    print("Generated directories:")
    for scenario_num in scenarios:
        scenario_dir = os.path.join("experiments", f"scenario_{scenario_num}_central")
        if os.path.exists(scenario_dir):
            run_count = len([d for d in os.listdir(scenario_dir) if d.startswith('run_') and os.path.isdir(os.path.join(scenario_dir, d))])
            print(f"  ✓ {scenario_dir}/ ({run_count} runs)")
        else:
            print(f"  ✗ {scenario_dir}/ (failed)")
    
    print("\nEach run directory contains:")
    print("  - trajectories.txt: Agent trajectories")
    print("  - goals.txt: Time-varying goal trajectories")
    print("  - config.json: Configuration used for this run")
    
    print("="*80)
    
    if successful == len(scenarios):
        print("🎉 All scenarios completed successfully!")
    elif successful > 0:
        print(f"⚠️  {successful}/{len(scenarios)} scenarios completed successfully")
    else:
        print("❌ No scenarios completed successfully")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
