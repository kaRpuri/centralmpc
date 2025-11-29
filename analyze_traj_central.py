import os
import numpy as np
import matplotlib.pyplot as plt

# Import common functions from base analysis file
from analyze_trajectories_base import load_trajectories, load_goals, compute_errors, count_collisions, compute_min_distances

# Parameters
COLLISION_RADIUS = 0.2
TIME_LIMIT = 10.0  # seconds
SCENARIOS = range(1, 10)  # 1 through 9
RUNS = range(1, 4)  # 1 through 3


def process_scenario_run(scenario, run):
    """Process a single scenario run. Returns metrics data."""
    traj_file = f'experiments/scenario_{scenario}_central/run_{run}/trajectories.txt'
    goals_file = f'experiments/scenario_{scenario}_central/run_{run}/goals.txt'
    
    # Check if files exist
    if not os.path.exists(traj_file) or not os.path.exists(goals_file):
        return None
    
    try:
        dt = 0.01  # seconds
        trajs, N = load_trajectories(traj_file)
        goals = load_goals(goals_file, N)
        avg_error, errors = compute_errors(trajs, goals, dt, TIME_LIMIT)
        collisions_per_timestep, total_collisions = count_collisions(trajs, dt, TIME_LIMIT, COLLISION_RADIUS)
        min_distances = compute_min_distances(trajs, dt, TIME_LIMIT)
        
        return {
            'avg_error': avg_error,
            'collisions_per_timestep': collisions_per_timestep,
            'min_distances': min_distances,
            'total_collisions': total_collisions,
            'mean_error': np.mean(avg_error),
            'dt': dt
        }
    except Exception as e:
        print(f"    Error processing run {run}: {e}")
        return None


def process_scenario(scenario):
    """Process all runs for a single scenario and create averaged plots."""
    print(f"\nProcessing Scenario {scenario}...")
    
    # Determine motion type
    if scenario <= 6:
        motion_type = "Static"
    elif scenario == 7:
        motion_type = "Translating"
    elif scenario == 8:
        motion_type = "Circular"
    elif scenario == 9:
        motion_type = "Circular+Translating"
    else:
        motion_type = "Unknown"
    
    print(f"  Motion type: {motion_type}")
    
    # Collect data from all runs
    run_data = []
    for run in RUNS:
        data = process_scenario_run(scenario, run)
        if data is not None:
            run_data.append(data)
    
    if not run_data:
        print(f"  No data found for scenario {scenario}")
        return False
    
    print(f"  Found data for {len(run_data)}/{len(RUNS)} runs")
    
    # Average across runs
    avg_errors = np.array([data['avg_error'] for data in run_data])
    collisions_per_timestep = np.array([data['collisions_per_timestep'] for data in run_data])
    min_distances_data = np.array([data['min_distances'] for data in run_data])
    
    # Ensure all arrays have the same length by taking the minimum
    min_length = min(len(arr) for arr in avg_errors)
    avg_errors = np.array([arr[:min_length] for arr in avg_errors])
    collisions_per_timestep = np.array([arr[:min_length] for arr in collisions_per_timestep])
    min_distances_data = np.array([arr[:min_length] for arr in min_distances_data])
    
    # Compute means and standard deviations
    avg_error_mean = np.mean(avg_errors, axis=0)
    avg_error_std = np.std(avg_errors, axis=0)
    collisions_mean = np.mean(collisions_per_timestep, axis=0)
    collisions_std = np.std(collisions_per_timestep, axis=0)
    min_distances_mean = np.mean(min_distances_data, axis=0)
    min_distances_std = np.std(min_distances_data, axis=0)
    
    # Summary statistics
    total_collisions_mean = np.mean([data['total_collisions'] for data in run_data])
    total_collisions_std = np.std([data['total_collisions'] for data in run_data])
    mean_error_avg = np.mean([data['mean_error'] for data in run_data])
    mean_error_std = np.std([data['mean_error'] for data in run_data])
    min_distance_overall = np.min(min_distances_mean)
    
    print(f"  Avg error = {mean_error_avg:.4f} ± {mean_error_std:.4f} m")
    print(f"  Total collisions = {total_collisions_mean:.1f} ± {total_collisions_std:.1f}")
    print(f"  Min distance = {min_distance_overall:.3f} m")
    
    # Create results directory
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Time array
    dt = run_data[0]['dt']
    time = np.arange(len(avg_error_mean)) * dt
    
    # Plot average error with error bands
    plt.figure(figsize=(10, 6))
    plt.plot(time, avg_error_mean, label=f'Centralized MPC - {motion_type}', 
            color='blue', linewidth=2)
    plt.fill_between(time, 
                    avg_error_mean - avg_error_std,
                    avg_error_mean + avg_error_std,
                    alpha=0.3, color='blue', label=f'±1 std dev (n={len(run_data)})')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Distance to Goal (m)')
    plt.title(f'Average Goal Error - Scenario {scenario} ({motion_type} Goals) - Centralized MPC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    error_plot_path = os.path.join(results_dir, f'central_average_error_scenario_{scenario}.png')
    plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot cumulative collisions with error bands
    plt.figure(figsize=(10, 6))
    cumulative_mean = np.cumsum(collisions_mean)
    cumulative_std = np.cumsum(collisions_std)
    plt.plot(time, cumulative_mean, color='red', linewidth=2,
            label=f'Centralized MPC - {motion_type}')
    plt.fill_between(time,
                    cumulative_mean - cumulative_std,
                    cumulative_mean + cumulative_std,
                    alpha=0.3, color='red', label=f'±1 std dev (n={len(run_data)})')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Collisions')
    plt.title(f'Cumulative Collisions (<{COLLISION_RADIUS:.2f}m) - Scenario {scenario} ({motion_type} Goals) - Centralized MPC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    collisions_plot_path = os.path.join(results_dir, f'central_cumulative_collisions_scenario_{scenario}.png')
    plt.savefig(collisions_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot minimum distances with error bands
    plt.figure(figsize=(10, 6))
    plt.plot(time, min_distances_mean, label=f'Centralized MPC - {motion_type}', 
            color='green', linewidth=2)
    plt.fill_between(time,
                    min_distances_mean - min_distances_std,
                    min_distances_mean + min_distances_std,
                    alpha=0.3, color='green', label=f'±1 std dev (n={len(run_data)})')
    
    # Add horizontal line at 0.3m to show dangerous area
    plt.axhline(y=0.3, color='red', linestyle=':', linewidth=2, alpha=0.8, 
               label='Dangerous Area (0.3m)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Minimum Distance Between Any Two Bots (m)')
    plt.title(f'Minimum Distance Between Bots - Scenario {scenario} ({motion_type} Goals) - Centralized MPC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    min_distance_plot_path = os.path.join(results_dir, f'central_minimum_distances_scenario_{scenario}.png')
    plt.savefig(min_distance_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def create_comparison_summary():
    """Create a summary comparison across all scenarios."""
    print(f"\nCreating comparison summary...")
    
    scenario_metrics = {}
    
    for scenario in SCENARIOS:
        # Collect data from all runs
        run_data = []
        for run in RUNS:
            traj_file = f'experiments/scenario_{scenario}_central/run_{run}/trajectories.txt'
            goals_file = f'experiments/scenario_{scenario}_central/run_{run}/goals.txt'
            
            if os.path.exists(traj_file) and os.path.exists(goals_file):
                try:
                    dt = 0.01
                    trajs, N = load_trajectories(traj_file)
                    goals = load_goals(goals_file, N)
                    avg_error, _ = compute_errors(trajs, goals, dt, TIME_LIMIT)
                    _, total_collisions = count_collisions(trajs, dt, TIME_LIMIT, COLLISION_RADIUS)
                    min_distances = compute_min_distances(trajs, dt, TIME_LIMIT)
                    
                    run_data.append({
                        'mean_error': np.mean(avg_error),
                        'total_collisions': total_collisions,
                        'min_distance': np.min(min_distances),
                        'final_error': avg_error[-1] if len(avg_error) > 0 else np.inf
                    })
                except Exception as e:
                    print(f"    Error processing scenario {scenario} run {run} for summary: {e}")
        
        if run_data:
            # Determine motion type
            if scenario <= 6:
                motion_type = "Static"
            elif scenario == 7:
                motion_type = "Translating"
            elif scenario == 8:
                motion_type = "Circular"
            elif scenario == 9:
                motion_type = "Circular+Translating"
            else:
                motion_type = "Unknown"
            
            # Compute averages and standard deviations
            scenario_metrics[scenario] = {
                'motion_type': motion_type,
                'mean_error': np.mean([data['mean_error'] for data in run_data]),
                'mean_error_std': np.std([data['mean_error'] for data in run_data]),
                'total_collisions': np.mean([data['total_collisions'] for data in run_data]),
                'total_collisions_std': np.std([data['total_collisions'] for data in run_data]),
                'min_distance': np.mean([data['min_distance'] for data in run_data]),
                'min_distance_std': np.std([data['min_distance'] for data in run_data]),
                'final_error': np.mean([data['final_error'] for data in run_data]),
                'final_error_std': np.std([data['final_error'] for data in run_data]),
                'num_runs': len(run_data)
            }
    
    if not scenario_metrics:
        print("  No data available for summary")
        return
    
    # Create summary plots
    results_dir = './results'
    
    # Summary bar plot - Mean Error by Scenario
    plt.figure(figsize=(12, 6))
    scenarios_list = list(scenario_metrics.keys())
    mean_errors = [scenario_metrics[s]['mean_error'] for s in scenarios_list]
    error_stds = [scenario_metrics[s]['mean_error_std'] for s in scenarios_list]
    motion_types = [scenario_metrics[s]['motion_type'] for s in scenarios_list]
    
    colors = {'Static': 'blue', 'Translating': 'green', 'Circular': 'orange', 'Circular+Translating': 'red'}
    bar_colors = [colors.get(mt, 'gray') for mt in motion_types]
    
    plt.bar(scenarios_list, mean_errors, yerr=error_stds, color=bar_colors, alpha=0.7, 
           capsize=5, error_kw={'linewidth': 2})
    plt.xlabel('Scenario')
    plt.ylabel('Mean Error (m)')
    plt.title(f'Centralized MPC - Mean Goal Tracking Error by Scenario (Averaged over {len(RUNS)} runs)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add motion type labels
    for i, (scenario, motion_type) in enumerate(zip(scenarios_list, motion_types)):
        plt.text(scenario, mean_errors[i] + error_stds[i] + 0.01, motion_type, 
                ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.tight_layout()
    summary_error_path = os.path.join(results_dir, 'central_summary_mean_errors.png')
    plt.savefig(summary_error_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary bar plot - Total Collisions by Scenario
    plt.figure(figsize=(12, 6))
    total_collisions_list = [scenario_metrics[s]['total_collisions'] for s in scenarios_list]
    collisions_stds = [scenario_metrics[s]['total_collisions_std'] for s in scenarios_list]
    
    plt.bar(scenarios_list, total_collisions_list, yerr=collisions_stds, color=bar_colors, alpha=0.7,
           capsize=5, error_kw={'linewidth': 2})
    plt.xlabel('Scenario')
    plt.ylabel('Total Collisions')
    plt.title(f'Centralized MPC - Total Collisions by Scenario (Averaged over {len(RUNS)} runs)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add motion type labels
    for i, (scenario, motion_type) in enumerate(zip(scenarios_list, motion_types)):
        plt.text(scenario, total_collisions_list[i] + collisions_stds[i] + 0.5, motion_type, 
                ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.tight_layout()
    summary_collisions_path = os.path.join(results_dir, 'central_summary_total_collisions.png')
    plt.savefig(summary_collisions_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Summary plots saved to {results_dir}/")
    print("\nSummary Statistics:")
    for scenario in scenarios_list:
        data = scenario_metrics[scenario]
        print(f"  Scenario {scenario} ({data['motion_type']}): "
              f"Error = {data['mean_error']:.4f}±{data['mean_error_std']:.4f}m, "
              f"Collisions = {data['total_collisions']:.1f}±{data['total_collisions_std']:.1f}, "
              f"Runs = {data['num_runs']}")
    
    return scenario_metrics


def main():
    """Process all centralized MPC scenarios."""
    print("\n" + "="*80)
    print("CENTRALIZED MPC - TRAJECTORY ANALYSIS")
    print("="*80)
    print(f"Analyzing {len(SCENARIOS)} scenarios from centralized MPC:")
    print("  Scenarios 1-6: Static goals")
    print("  Scenario 7:    Translating goals") 
    print("  Scenario 8:    Circular goals")
    print("  Scenario 9:    Circular + Translating goals")
    print(f"Each scenario averaged over {len(RUNS)} runs")
    print("="*80)
    
    processed = 0
    successful = 0
    
    for scenario in SCENARIOS:
        processed += 1
        print(f"\nProcessing scenario {processed}/{len(SCENARIOS)}: Scenario {scenario}")
        if process_scenario(scenario):
            successful += 1
    
    # Create comparison summary
    create_comparison_summary()
    
    print("\n" + "="*80)
    print("CENTRALIZED MPC ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Total scenarios: {len(SCENARIOS)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed/Skipped: {processed - successful}")
    print(f"Results saved in: ./results/")
    print("Generated files:")
    print("  Individual scenario plots (averaged over multiple runs):")
    for scenario in SCENARIOS:
        scenario_dir = f'experiments/scenario_{scenario}_central'
        if os.path.exists(scenario_dir):
            run_count = len([d for d in os.listdir(scenario_dir) if d.startswith('run_') and os.path.isdir(os.path.join(scenario_dir, d))])
            print(f"    - central_*_scenario_{scenario}.png (from {run_count} runs)")
    print("  Summary plots (with error bars):")
    print("    - central_summary_mean_errors.png")
    print("    - central_summary_total_collisions.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
