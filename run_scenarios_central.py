"""
Centralized MPC Scenario Runner
Runs all 9 scenarios from ../online_dmpc/cpp/config/ using the centralized MPC approach
"""

import os
import json
import numpy as np
from typing import Dict, Any
import time

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


def run_scenario(scenario_num: int) -> bool:
    """Run a single scenario with centralized MPC"""
    
    print(f"\n{'='*20} SCENARIO {scenario_num} {'='*20}")
    
    try:
        # Load scenario configuration
        scenario_config = load_scenario_config(scenario_num)
        print(f"✓ Loaded config for scenario {scenario_num}")
        
        # Convert to central MPC format
        central_config = convert_config_for_central_mpc(scenario_config, scenario_num)
        motion_type = central_config["motion_type"]
        print(f"  Motion type: {motion_type}")
        print(f"  Agents: {central_config['num_agents']}")
        print(f"  Duration: {central_config['sim_duration']}s")
        print(f"  Workspace: {central_config['pos_min']} to {central_config['pos_max']}")
        
        # Create output directory inside experiments folder
        experiments_dir = "experiments"
        os.makedirs(experiments_dir, exist_ok=True)
        output_dir = os.path.join(experiments_dir, f"scenario_{scenario_num}_central")
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
        print(f"  ✓ Scenario {scenario_num} completed in {elapsed:.1f}s")
        print(f"    Results saved to: {output_dir}/")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in scenario {scenario_num}: {e}")
        return False


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
        output_dir = os.path.join("experiments", f"scenario_{scenario_num}_central")
        if os.path.exists(output_dir):
            print(f"  ✓ {output_dir}/")
        else:
            print(f"  ✗ {output_dir}/ (failed)")
    
    print("\nEach directory contains:")
    print("  - trajectories.txt: Agent trajectories")
    print("  - goals.txt: Time-varying goal trajectories")
    print("  - config.json: Configuration used for this scenario")
    
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
