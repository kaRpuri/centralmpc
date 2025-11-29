"""
Centralized MPC Scalability Test Runner
Runs scalability tests with 4, 8, 16, 32, 64 agents using configurations from ../online_dmpc/cpp/config/
Run using: python3 run_scenario_scale_central.py --agents 4 8 16 32 64 --runs 1 2 3
"""

import os
import json
import numpy as np
from typing import Dict, Any, List
import time
import argparse

# Parameters
RUNS = range(1, 4)  # 1 through 3 runs per agent count
AGENT_COUNTS = [4, 8, 16, 32, 64]  # Scalability test configurations

# Import classes from central_mpc.py
from central_mpc import CentralizedMPC, GoalManager, Simulator


def load_scale_config(agent_count: int) -> Dict[str, Any]:
    """Load scalability configuration from JSON file"""
    config_path = f"../online_dmpc/cpp/config/scenario_scale_{agent_count}.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Scale config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def convert_to_central_config(dmpc_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert DMPC config to centralized MPC format"""
    central_config = {
        "num_agents": dmpc_config["N"],
        "dim": dmpc_config["dim"],
        "horizon": dmpc_config.get("k_hor", 30),
        "dt": dmpc_config.get("ts", 0.01),
        "dt_plan": dmpc_config.get("h", 0.1),
        "sim_duration": dmpc_config.get("simulation_duration", 30),
        
        # Position and velocity limits
        "pos_min": dmpc_config.get("pmin", [-0.5, -0.5, 0.2]),
        "pos_max": dmpc_config.get("pmax", [4.5, 4.5, 4.0]),
        "vel_max": 1.5,
        "acc_max": dmpc_config.get("amax", [1.0, 1.0, 1.0])[0],
        
        # MPC weights (tuned for larger systems)
        "Q_pos": 50.0,
        "Q_vel": 1.0,
        "R": 0.1,
        "Q_terminal": 20.0,
        
        # Noise parameters (same as distributed)
        "std_position": dmpc_config.get("std_position", 0.00228682),
        "std_velocity": dmpc_config.get("std_velocity", 0.0109302),
        
        # Goal tracking
        "goal_tolerance": dmpc_config.get("goal_tolerance", 0.15),
        "motion_type": dmpc_config.get("motion_type", "static"),
        "circular_radius": dmpc_config.get("goal_circular_radius", 2.0),
        "circular_omega": dmpc_config.get("goal_circular_omega", 0.2),
        "translation_velocity": dmpc_config.get("goal_translation_velocity", 0.2),
        
        # Extract initial and goal positions
        "initial_positions": dmpc_config.get("po", []),
        "goal_positions": dmpc_config.get("pf", [])
    }
    
    # Validate positions
    if len(central_config["initial_positions"]) != central_config["num_agents"]:
        raise ValueError(f"Initial positions count mismatch: {len(central_config['initial_positions'])} != {central_config['num_agents']}")
    
    if len(central_config["goal_positions"]) != central_config["num_agents"]:
        raise ValueError(f"Goal positions count mismatch: {len(central_config['goal_positions'])} != {central_config['num_agents']}")
    
    return central_config


def run_scale_single_run(agent_count: int, run_num: int) -> bool:
    """Run a single run of a scalability test with centralized MPC"""
    try:
        print(f"\n=============== SCALABILITY {agent_count} AGENTS - RUN {run_num} ===============")
        
        # Load and convert configuration
        dmpc_config = load_scale_config(agent_count)
        central_config = convert_to_central_config(dmpc_config)
        
        # Add slight variations between runs
        np.random.seed(42 + agent_count * 100 + run_num)  # Reproducible but different seeds
        
        # Use exact noise parameters from distributed DMPC JSON configs
        std_position = dmpc_config.get("std_position", 0.00228682)
        std_velocity = dmpc_config.get("std_velocity", 0.0109302)
        
        # Apply noise parameters to central config
        central_config["std_position"] = std_position
        central_config["std_velocity"] = std_velocity
        
        # Add small variations between runs (±10% noise variation)
        noise_variations = [0.9, 1.0, 1.1]  # 90%, 100%, 110% of base noise
        central_config["std_position"] = std_position * noise_variations[run_num - 1]
        central_config["std_velocity"] = std_velocity * noise_variations[run_num - 1]
        
        # Vary goal tolerance slightly
        base_tolerance = central_config.get("goal_tolerance", 0.15)
        tolerance_variations = [0.9, 1.0, 1.1]  # 90%, 100%, 110% of base tolerance
        central_config["goal_tolerance"] = base_tolerance * tolerance_variations[run_num - 1]
        
        motion_type = central_config["motion_type"]
        
        print(f"✓ Loaded config for {agent_count} agents, run {run_num}")
        print(f"  Motion type: {motion_type}")
        print(f"  Agents: {central_config['num_agents']}")
        print(f"  Duration: {central_config['sim_duration']}s")
        print(f"  Position noise std: {central_config['std_position']:.6f}")
        print(f"  Velocity noise std: {central_config['std_velocity']:.6f}")
        print(f"  Goal tolerance: {central_config['goal_tolerance']:.3f}")
        print(f"  Running centralized MPC simulation...")
        
        # Create output directory inside experiments folder
        experiments_dir = "experiments"
        os.makedirs(experiments_dir, exist_ok=True)
        scale_dir = os.path.join(experiments_dir, f"scale_{agent_count}_agents_central")
        os.makedirs(scale_dir, exist_ok=True)
        
        output_dir = os.path.join(scale_dir, f"run_{run_num}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run simulation
        start_time = time.time()
        
        sim = Simulator(central_config)
        success = sim.simulate()
        
        if not success:
            print(f"✗ Simulation failed for {agent_count} agents, run {run_num}")
            return False
        
        # Save results
        traj_path = os.path.join(output_dir, 'trajectories.txt')
        goals_path = os.path.join(output_dir, 'goals.txt')
        
        sim.save_trajectories(traj_path)
        sim.save_goals(goals_path)
        
        # Clean up config before saving (remove old parameters)
        config_to_save = central_config.copy()
        if "noise_std" in config_to_save:
            del config_to_save["noise_std"]
        
        # Save configuration for reference
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"  ✓ {agent_count} agents, run {run_num} completed in {elapsed:.1f}s")
        print(f"    Results saved to: {output_dir}/")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in {agent_count} agents, run {run_num}: {e}")
        return False


def run_scale_all_runs(agent_count: int, runs: List[int]) -> Dict[str, Any]:
    """Run all runs for a given agent count"""
    print(f"\n🚀 Starting scalability test for {agent_count} agents...")
    
    results = {
        "agent_count": agent_count,
        "total_runs": len(runs),
        "successful_runs": 0,
        "failed_runs": 0,
        "run_details": {}
    }
    
    for run_num in runs:
        success = run_scale_single_run(agent_count, run_num)
        
        if success:
            results["successful_runs"] += 1
            results["run_details"][f"run_{run_num}"] = "success"
        else:
            results["failed_runs"] += 1
            results["run_details"][f"run_{run_num}"] = "failed"
    
    success_rate = (results["successful_runs"] / results["total_runs"]) * 100
    print(f"\n✓ {agent_count} agents summary: {results['successful_runs']}/{results['total_runs']} runs successful ({success_rate:.1f}%)")
    
    return results


def main():
    """Main function to run scalability tests"""
    parser = argparse.ArgumentParser(description='Run centralized MPC scalability tests')
    parser.add_argument('--agents', type=int, nargs='+', default=AGENT_COUNTS, 
                       choices=AGENT_COUNTS, help='Agent counts to test')
    parser.add_argument('--runs', type=int, nargs='+', default=list(RUNS), 
                       help='Run numbers to execute (default: 1 2 3)')
    parser.add_argument('--max-agents', type=int, default=64,
                       help='Maximum number of agents to test (default: 64)')
    
    args = parser.parse_args()
    
    # Filter agent counts by max-agents limit
    agent_counts = [count for count in args.agents if count <= args.max_agents]
    runs = args.runs
    
    print("================================================================================")
    print("CENTRALIZED MPC SCALABILITY TEST")
    print("================================================================================")
    print(f"Agent counts: {agent_counts}")
    print(f"Runs per agent count: {runs}")
    print(f"Total tests: {len(agent_counts) * len(runs)}")
    print("================================================================================")
    
    overall_start_time = time.time()
    all_results = []
    successful_configs = 0
    failed_configs = 0
    
    for agent_count in agent_counts:
        try:
            results = run_scale_all_runs(agent_count, runs)
            all_results.append(results)
            
            if results["failed_runs"] == 0:
                successful_configs += 1
            else:
                failed_configs += 1
                
        except Exception as e:
            print(f"\n✗ Critical error with {agent_count} agents: {e}")
            failed_configs += 1
            continue
    
    total_runtime = time.time() - overall_start_time
    
    print("\n" + "="*80)
    print("SCALABILITY TEST COMPLETE!")
    print("="*80)
    print(f"Total agent configurations: {len(agent_counts)}")
    print(f"Successful: {successful_configs}")
    print(f"Failed: {failed_configs}")
    print(f"Total runtime: {total_runtime:.1f}s")
    
    print(f"\nGenerated directories:")
    for agent_count in agent_counts:
        print(f"  ✓ experiments/scale_{agent_count}_agents_central/ ({len(runs)} runs)")
    
    print(f"\nEach run directory contains:")
    print(f"  - trajectories.txt: Agent trajectories")
    print(f"  - goals.txt: Time-varying goal trajectories")  
    print(f"  - config.json: Configuration used for this run")
    
    # Generate summary statistics
    total_agents_tested = 0
    total_successful_runs = 0
    total_runs = 0
    
    print(f"\n" + "="*80)
    print("SCALABILITY RESULTS SUMMARY")
    print("="*80)
    for result in all_results:
        agent_count = result["agent_count"] 
        success_rate = (result["successful_runs"] / result["total_runs"]) * 100
        print(f"  {agent_count:2d} agents: {result['successful_runs']:2d}/{result['total_runs']:2d} runs successful ({success_rate:5.1f}%)")
        
        total_agents_tested += agent_count * result["successful_runs"]
        total_successful_runs += result["successful_runs"]
        total_runs += result["total_runs"]
    
    overall_success_rate = (total_successful_runs / total_runs) * 100 if total_runs > 0 else 0
    print(f"\n  Overall: {total_successful_runs}/{total_runs} runs successful ({overall_success_rate:.1f}%)")
    print(f"  Total agent-simulations completed: {total_agents_tested}")
    
    print("="*80)
    if failed_configs == 0:
        print("🎉 All scalability tests completed successfully!")
    else:
        print(f"⚠️  {failed_configs} agent configurations had issues - check logs above")
    print("="*80)


if __name__ == "__main__":
    main()
