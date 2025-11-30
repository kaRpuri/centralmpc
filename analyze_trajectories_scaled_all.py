import os
import numpy as np
import matplotlib.pyplot as plt

# Import common functions from base analysis file
from analyze_trajectories_base import load_trajectories, load_goals, compute_errors, count_collisions, compute_min_distances

# ============================================================
# CONTROL VARIABLES - Set to True/False to include/exclude methods
# ============================================================
INCLUDE_CENTRALIZED_MPC = True
INCLUDE_STATIC_ON_DEMAND = True
INCLUDE_STATIC_BVC = True
INCLUDE_REACTIVE_ON_DEMAND = False
INCLUDE_REACTIVE_BVC = False
INCLUDE_PREDICTIVE_ON_DEMAND = False
INCLUDE_PREDICTIVE_BVC = False
# ============================================================

# Parameters
COLLISION_RADIUS = 0.2
TIME_LIMIT = 10.0  # seconds
AGENT_COUNTS = [4, 8, 16, 32, 64]  # Different agent counts for scalability analysis
RUNS = range(1, 4)  # 1 through 3


def load_centralized_data(agent_count):
    """Load centralized MPC data for a specific agent count (averaged over multiple runs)."""
    run_data = []
    
    for run in RUNS:
        traj_file = f'experiments/scale_{agent_count}_agents_central/run_{run}/trajectories.txt'
        goals_file = f'experiments/scale_{agent_count}_agents_central/run_{run}/goals.txt'
        
        if os.path.exists(traj_file) and os.path.exists(goals_file):
            try:
                dt = 0.01  # seconds
                trajs, N = load_trajectories(traj_file)
                goals = load_goals(goals_file, N)
                avg_error, errors = compute_errors(trajs, goals, dt, TIME_LIMIT)
                collisions_per_timestep, total_collisions = count_collisions(trajs, dt, TIME_LIMIT, COLLISION_RADIUS)
                min_distances = compute_min_distances(trajs, dt, TIME_LIMIT)
                
                run_data.append({
                    'avg_error': avg_error,
                    'collisions_per_timestep': collisions_per_timestep,
                    'min_distances': min_distances,
                    'total_collisions': total_collisions,
                    'mean_error': np.mean(avg_error),
                    'dt': dt
                })
            except Exception as e:
                print(f"Error loading centralized data for {agent_count} agents, run {run}: {e}")
    
    if not run_data:
        return None
    
    # Average across runs
    avg_errors = np.array([data['avg_error'] for data in run_data])
    collisions_per_timestep = np.array([data['collisions_per_timestep'] for data in run_data])
    min_distances_data = np.array([data['min_distances'] for data in run_data])
    
    # Ensure all arrays have the same length
    min_length = min(len(arr) for arr in avg_errors)
    avg_errors = np.array([arr[:min_length] for arr in avg_errors])
    collisions_per_timestep = np.array([arr[:min_length] for arr in collisions_per_timestep])
    min_distances_data = np.array([arr[:min_length] for arr in min_distances_data])
    
    return {
        'avg_error_mean': np.mean(avg_errors, axis=0),
        'avg_error_std': np.std(avg_errors, axis=0),
        'collisions_mean': np.mean(collisions_per_timestep, axis=0),
        'collisions_std': np.std(collisions_per_timestep, axis=0),
        'min_distances_mean': np.mean(min_distances_data, axis=0),
        'min_distances_std': np.std(min_distances_data, axis=0),
        'total_collisions_mean': np.mean([data['total_collisions'] for data in run_data]),
        'mean_error': np.mean([data['mean_error'] for data in run_data]),
        'dt': run_data[0]['dt'],
        'num_runs': len(run_data)
    }


def load_distributed_data(agent_count, reallocation_method, collision_method):
    """Load distributed DMPC data for a specific agent count and method combination."""
    run_data = []
    
    for run in RUNS:
        traj_file = f'../online_dmpc/cpp/results/scalability/scenario_scale_{agent_count}/{reallocation_method}/{collision_method}/run_{run}/trajectories.txt'
        goals_file = f'../online_dmpc/cpp/results/scalability/scenario_scale_{agent_count}/{reallocation_method}/{collision_method}/run_{run}/goals.txt'
        
        if os.path.exists(traj_file) and os.path.exists(goals_file):
            try:
                dt = 0.01  # seconds
                trajs, N = load_trajectories(traj_file)
                goals = load_goals(goals_file, N)
                avg_error, errors = compute_errors(trajs, goals, dt, TIME_LIMIT)
                collisions_per_timestep, total_collisions = count_collisions(trajs, dt, TIME_LIMIT, COLLISION_RADIUS)
                min_distances = compute_min_distances(trajs, dt, TIME_LIMIT)
                
                run_data.append({
                    'avg_error': avg_error,
                    'collisions_per_timestep': collisions_per_timestep,
                    'min_distances': min_distances,
                    'total_collisions': total_collisions,
                    'mean_error': np.mean(avg_error),
                    'dt': dt
                })
            except Exception as e:
                print(f"Error loading {reallocation_method}/{collision_method} run {run} for {agent_count} agents: {e}")
    
    if not run_data:
        return None
    
    # Average across runs
    avg_errors = np.array([data['avg_error'] for data in run_data])
    collisions_per_timestep = np.array([data['collisions_per_timestep'] for data in run_data])
    min_distances_data = np.array([data['min_distances'] for data in run_data])
    
    # Ensure all arrays have the same length
    min_length = min(len(arr) for arr in avg_errors)
    avg_errors = np.array([arr[:min_length] for arr in avg_errors])
    collisions_per_timestep = np.array([arr[:min_length] for arr in collisions_per_timestep])
    min_distances_data = np.array([arr[:min_length] for arr in min_distances_data])
    
    return {
        'avg_error_mean': np.mean(avg_errors, axis=0),
        'avg_error_std': np.std(avg_errors, axis=0),
        'collisions_mean': np.mean(collisions_per_timestep, axis=0),
        'collisions_std': np.std(collisions_per_timestep, axis=0),
        'min_distances_mean': np.mean(min_distances_data, axis=0),
        'min_distances_std': np.std(min_distances_data, axis=0),
        'total_collisions_mean': np.mean([data['total_collisions'] for data in run_data]),
        'mean_error': np.mean([data['mean_error'] for data in run_data]),
        'dt': run_data[0]['dt']
    }


def get_method_style(method_name):
    """Get plotting style (color, linestyle, marker) for each method."""
    styles = {
        'Centralized MPC': {'color': 'black', 'linestyle': '-', 'linewidth': 3},
        'Static/On-demand': {'color': 'blue', 'linestyle': '-', 'linewidth': 2},
        'Static/BVC': {'color': 'blue', 'linestyle': '--', 'linewidth': 2},
        'Reactive/On-demand': {'color': 'red', 'linestyle': '-', 'linewidth': 2},
        'Reactive/BVC': {'color': 'red', 'linestyle': '--', 'linewidth': 2},
        'Predictive/On-demand': {'color': 'green', 'linestyle': '-', 'linewidth': 2},
        'Predictive/BVC': {'color': 'green', 'linestyle': '--', 'linewidth': 2}
    }
    return styles.get(method_name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1})


def process_agent_count(agent_count):
    """Process and plot all enabled methods for a specific agent count."""
    print(f"\nProcessing {agent_count} agents...")
    
    print(f"  Scalability test with {agent_count} agents")
    
    # Collect all enabled methods data
    method_data = {}
    
    # Load centralized MPC data
    if INCLUDE_CENTRALIZED_MPC:
        central_data = load_centralized_data(agent_count)
        if central_data:
            method_data['Centralized MPC'] = central_data
            print(f"    ✓ Centralized MPC: Avg error = {central_data['mean_error']:.4f} m, Collisions = {central_data['total_collisions_mean']:.1f} (n={central_data['num_runs']})")
    
    # Load distributed DMPC data for enabled methods
    method_configs = [
        ('static', 'on-demand', 'Static/On-demand', INCLUDE_STATIC_ON_DEMAND),
        ('static', 'BVC', 'Static/BVC', INCLUDE_STATIC_BVC),
        ('reactive', 'on-demand', 'Reactive/On-demand', INCLUDE_REACTIVE_ON_DEMAND),
        ('reactive', 'BVC', 'Reactive/BVC', INCLUDE_REACTIVE_BVC),
        ('predictive', 'on-demand', 'Predictive/On-demand', INCLUDE_PREDICTIVE_ON_DEMAND),
        ('predictive', 'BVC', 'Predictive/BVC', INCLUDE_PREDICTIVE_BVC)
    ]
    
    for realloc_method, collision_method, label, include in method_configs:
        if include:
            dist_data = load_distributed_data(agent_count, realloc_method, collision_method)
            if dist_data:
                method_data[label] = dist_data
                print(f"    ✓ {label}: Avg error = {dist_data['mean_error']:.4f} m, Collisions = {dist_data['total_collisions_mean']:.1f}")
            else:
                print(f"    ✗ {label}: No data available")
    
    if not method_data:
        print(f"  No data available for {agent_count} agents")
        return False
    
    # Create results directory
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Determine time array from the first method with data
    first_method = next(iter(method_data.values()))
    dt = first_method['dt']
    time = np.arange(len(first_method['avg_error_mean'])) * dt
    
    # Plot 1: Average Error Comparison
    plt.figure(figsize=(14, 8))
    
    for method_name, data in method_data.items():
        style = get_method_style(method_name)
        plt.plot(time, data['avg_error_mean'], label=method_name, **style)
        
        # Add error bands for all methods (both distributed and centralized now have multiple runs)
        if np.any(data['avg_error_std'] > 0):
            plt.fill_between(time,
                            data['avg_error_mean'] - data['avg_error_std'],
                            data['avg_error_mean'] + data['avg_error_std'],
                            alpha=0.2, color=style['color'])
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Average Distance to Goal (m)', fontsize=12)
    plt.title(f'Goal Tracking Performance - {agent_count} Agents (Scalability Test)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    error_plot_path = os.path.join(results_dir, f'all_methods_error_{agent_count}_agents.png')
    plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Cumulative Collisions Comparison
    plt.figure(figsize=(14, 8))
    
    for method_name, data in method_data.items():
        style = get_method_style(method_name)
        cumulative_mean = np.cumsum(data['collisions_mean'])
        plt.plot(time, cumulative_mean, label=method_name, **style)
        
        # Add error bands for all methods (both distributed and centralized now have multiple runs)
        if np.any(data['collisions_std'] > 0):
            cumulative_std = np.cumsum(data['collisions_std'])
            plt.fill_between(time,
                            cumulative_mean - cumulative_std,
                            cumulative_mean + cumulative_std,
                            alpha=0.2, color=style['color'])
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Cumulative Collisions', fontsize=12)
    plt.title(f'Collision Safety - {agent_count} Agents (Scalability Test)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    collisions_plot_path = os.path.join(results_dir, f'all_methods_collisions_{agent_count}_agents.png')
    plt.savefig(collisions_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Minimum Distances Comparison
    plt.figure(figsize=(14, 8))
    
    for method_name, data in method_data.items():
        style = get_method_style(method_name)
        plt.plot(time, data['min_distances_mean'], label=method_name, **style)
        
        # Add error bands for all methods (both distributed and centralized now have multiple runs)
        if np.any(data['min_distances_std'] > 0):
            plt.fill_between(time,
                            data['min_distances_mean'] - data['min_distances_std'],
                            data['min_distances_mean'] + data['min_distances_std'],
                            alpha=0.2, color=style['color'])
    
    # Add horizontal line at 0.3m to show dangerous area
    plt.axhline(y=0.3, color='red', linestyle=':', linewidth=2, alpha=0.8, label='Dangerous Area (0.3m)')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Minimum Distance Between Any Two Bots (m)', fontsize=12)
    plt.title(f'Inter-Agent Safety - {agent_count} Agents (Scalability Test)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    min_distance_plot_path = os.path.join(results_dir, f'all_methods_min_distances_{agent_count}_agents.png')
    plt.savefig(min_distance_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Plots saved for {agent_count} agents")
    return True


def create_summary_comparison():
    """Create summary comparison across all agent counts and methods."""
    print(f"\nCreating summary comparison...")
    
    agent_count_summary = {}
    
    for agent_count in AGENT_COUNTS:
        agent_metrics = {}
        
        # Centralized MPC
        if INCLUDE_CENTRALIZED_MPC:
            central_data = load_centralized_data(agent_count)
            if central_data:
                agent_metrics['Centralized MPC'] = {
                    'mean_error': central_data['mean_error'],
                    'total_collisions': central_data['total_collisions_mean']
                }
        
        # Distributed DMPC methods
        method_configs = [
            ('static', 'on-demand', 'Static/On-demand', INCLUDE_STATIC_ON_DEMAND),
            ('static', 'BVC', 'Static/BVC', INCLUDE_STATIC_BVC),
            ('reactive', 'on-demand', 'Reactive/On-demand', INCLUDE_REACTIVE_ON_DEMAND),
            ('reactive', 'BVC', 'Reactive/BVC', INCLUDE_REACTIVE_BVC),
            ('predictive', 'on-demand', 'Predictive/On-demand', INCLUDE_PREDICTIVE_ON_DEMAND),
            ('predictive', 'BVC', 'Predictive/BVC', INCLUDE_PREDICTIVE_BVC)
        ]
        
        for realloc_method, collision_method, label, include in method_configs:
            if include:
                dist_data = load_distributed_data(agent_count, realloc_method, collision_method)
                if dist_data:
                    agent_metrics[label] = {
                        'mean_error': dist_data['mean_error'],
                        'total_collisions': dist_data['total_collisions_mean']
                    }
        
        if agent_metrics:
            agent_count_summary[agent_count] = {'agent_count': agent_count, 'methods': agent_metrics}
    
    if not agent_count_summary:
        print("  No data available for summary")
        return
    
    # Create summary plots
    results_dir = './results'
    
    # Summary plot: Mean Error by Agent Count and Method
    plt.figure(figsize=(16, 10))
    
    agent_counts_list = list(agent_count_summary.keys())
    method_names = set()
    for agent_data in agent_count_summary.values():
        method_names.update(agent_data['methods'].keys())
    method_names = sorted(list(method_names))
    
    x_positions = np.arange(len(agent_counts_list))
    bar_width = 0.8 / len(method_names) if method_names else 0.8
    
    for i, method in enumerate(method_names):
        errors = []
        for agent_count in agent_counts_list:
            if method in agent_count_summary[agent_count]['methods']:
                errors.append(agent_count_summary[agent_count]['methods'][method]['mean_error'])
            else:
                errors.append(0)
        
        style = get_method_style(method)
        plt.bar(x_positions + i * bar_width, errors, bar_width, 
               label=method, color=style['color'], alpha=0.7)
    
    plt.xlabel('Number of Agents', fontsize=12)
    plt.ylabel('Mean Goal Tracking Error (m)', fontsize=12)
    plt.title('Mean Goal Tracking Error - Scalability Analysis', fontsize=14, fontweight='bold')
    plt.xticks(x_positions + bar_width * (len(method_names)-1) / 2, agent_counts_list)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    summary_error_path = os.path.join(results_dir, 'scalability_summary_errors.png')
    plt.savefig(summary_error_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary plot: Total Collisions by Agent Count and Method
    plt.figure(figsize=(16, 10))
    
    for i, method in enumerate(method_names):
        collisions = []
        for agent_count in agent_counts_list:
            if method in agent_count_summary[agent_count]['methods']:
                collisions.append(agent_count_summary[agent_count]['methods'][method]['total_collisions'])
            else:
                collisions.append(0)
        
        style = get_method_style(method)
        plt.bar(x_positions + i * bar_width, collisions, bar_width, 
               label=method, color=style['color'], alpha=0.7)
    
    plt.xlabel('Number of Agents', fontsize=12)
    plt.ylabel('Total Collisions', fontsize=12)
    plt.title('Total Collisions - Scalability Analysis', fontsize=14, fontweight='bold')
    plt.xticks(x_positions + bar_width * (len(method_names)-1) / 2, agent_counts_list)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    summary_collisions_path = os.path.join(results_dir, 'scalability_summary_collisions.png')
    plt.savefig(summary_collisions_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Summary plots saved to {results_dir}/")


def main():
    """Main analysis function."""
    print("\n" + "="*90)
    print("SCALABILITY ANALYSIS - CENTRALIZED vs DISTRIBUTED")
    print("="*90)
    
    enabled_methods = []
    if INCLUDE_CENTRALIZED_MPC:
        enabled_methods.append("Centralized MPC")
    if INCLUDE_STATIC_ON_DEMAND:
        enabled_methods.append("Static/On-demand")
    if INCLUDE_STATIC_BVC:
        enabled_methods.append("Static/BVC")
    if INCLUDE_REACTIVE_ON_DEMAND:
        enabled_methods.append("Reactive/On-demand")
    if INCLUDE_REACTIVE_BVC:
        enabled_methods.append("Reactive/BVC")
    if INCLUDE_PREDICTIVE_ON_DEMAND:
        enabled_methods.append("Predictive/On-demand")
    if INCLUDE_PREDICTIVE_BVC:
        enabled_methods.append("Predictive/BVC")
    
    print(f"Enabled methods ({len(enabled_methods)}):")
    for method in enabled_methods:
        print(f"  ✓ {method}")
    
    if not enabled_methods:
        print("❌ No methods enabled! Please set at least one INCLUDE_* variable to True.")
        return
    
    print(f"\nAnalyzing {len(AGENT_COUNTS)} different agent counts:")
    for count in AGENT_COUNTS:
        print(f"  {count} agents")
    print("="*90)
    
    processed = 0
    successful = 0
    
    for agent_count in AGENT_COUNTS:
        processed += 1
        print(f"\nProcessing {processed}/{len(AGENT_COUNTS)}: {agent_count} agents")
        if process_agent_count(agent_count):
            successful += 1
    
    # Create summary comparison
    create_summary_comparison()
    
    print("\n" + "="*90)
    print("SCALABILITY ANALYSIS COMPLETE!")
    print("="*90)
    print(f"Total agent counts: {len(AGENT_COUNTS)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed/Skipped: {processed - successful}")
    print(f"Results saved in: ./results/")
    print("\nGenerated files:")
    print("  Per-agent-count comparisons:")
    print("    - all_methods_error_X_agents.png")
    print("    - all_methods_collisions_X_agents.png") 
    print("    - all_methods_min_distances_X_agents.png")
    print("  Cross-agent-count summaries:")
    print("    - scalability_summary_errors.png")
    print("    - scalability_summary_collisions.png")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
