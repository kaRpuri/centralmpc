import os
import numpy as np
import matplotlib.pyplot as plt

# Import common functions from base analysis file
from analyze_trajectories_base import load_trajectories, load_goals, compute_errors, count_collisions, compute_min_distances

# Parameters
COLLISION_RADIUS = 0.2
TIME_LIMIT = 10.0  # seconds
SCENARIOS = range(1, 10)  # 1 through 9
REALLOCATION_METHODS = ['static', 'predictive', 'reactive']
COLLISION_METHODS = ['BVC', 'on-demand']
RUNS = range(1, 4)  # 1 through 3


def process_scenario_run(scenario, reallocation_method, collision_method, run):
    """Process a single scenario, reallocation method, collision method, and run combination. Returns metrics data."""
    traj_file = f'../online_dmpc/cpp/results/experiments/scenario_{scenario}/{reallocation_method}/{collision_method}/run_{run}/trajectories.txt'
    goals_file = f'../online_dmpc/cpp/results/experiments/scenario_{scenario}/{reallocation_method}/{collision_method}/run_{run}/goals.txt'
    
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
        print(f"Error processing scenario {scenario}, {reallocation_method}/{collision_method}, run {run}: {e}")
        return None

def process_scenario(scenario):
    """Process all methods and runs for a single scenario, create combined plots."""
    print(f"\nProcessing Scenario {scenario}...")
    
    # Store data for all method combinations
    method_data = {}
    
    for realloc_method in REALLOCATION_METHODS:
        for collision_method in COLLISION_METHODS:
            method_key = f"{realloc_method}_{collision_method}"
            run_data = []
            
            for run in RUNS:
                data = process_scenario_run(scenario, realloc_method, collision_method, run)
                if data is not None:
                    run_data.append(data)
            
            if run_data:
                # Average across runs
                avg_errors = np.array([data['avg_error'] for data in run_data])
                collisions_per_timestep = np.array([data['collisions_per_timestep'] for data in run_data])
                min_distances_data = np.array([data['min_distances'] for data in run_data])
                
                # Ensure all arrays have the same length by taking the minimum
                min_length = min(len(arr) for arr in avg_errors)
                avg_errors = np.array([arr[:min_length] for arr in avg_errors])
                collisions_per_timestep = np.array([arr[:min_length] for arr in collisions_per_timestep])
                min_distances_data = np.array([arr[:min_length] for arr in min_distances_data])
                
                method_data[method_key] = {
                    'avg_error_mean': np.mean(avg_errors, axis=0),
                    'avg_error_std': np.std(avg_errors, axis=0),
                    'collisions_mean': np.mean(collisions_per_timestep, axis=0),
                    'collisions_std': np.std(collisions_per_timestep, axis=0),
                    'min_distances_mean': np.mean(min_distances_data, axis=0),
                    'min_distances_std': np.std(min_distances_data, axis=0),
                    'total_collisions_mean': np.mean([data['total_collisions'] for data in run_data]),
                    'mean_error': np.mean([data['mean_error'] for data in run_data]),
                    'dt': run_data[0]['dt'],
                    'reallocation': realloc_method,
                    'collision': collision_method
                }
                print(f"  {realloc_method.capitalize()}/{collision_method}: Avg error = {method_data[method_key]['mean_error']:.4f} m, "
                      f"Total collisions = {method_data[method_key]['total_collisions_mean']:.1f}")
    
    if not method_data:
        print(f"  No data found for scenario {scenario}")
        return False
    
    # Create results directory
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Determine time array from the first method with data
    first_method = next(iter(method_data.values()))
    dt = first_method['dt']
    time = np.arange(len(first_method['avg_error_mean'])) * dt
    
    # Plot combined average error
    plt.figure(figsize=(12, 8))
    
    # Define colors and line styles for different method combinations
    realloc_colors = {'static': 'blue', 'predictive': 'green', 'reactive': 'red'}
    collision_styles = {'BVC': '-', 'on-demand': '--'}
    
    for method_key, data in method_data.items():
        realloc = data['reallocation']
        collision = data['collision']
        color = realloc_colors.get(realloc, 'black')
        linestyle = collision_styles.get(collision, '-')
        label = f'{realloc.capitalize()}/{collision}'
        
        plt.plot(time, data['avg_error_mean'], label=label, 
                color=color, linestyle=linestyle, linewidth=2)
        plt.fill_between(time, 
                        data['avg_error_mean'] - data['avg_error_std'],
                        data['avg_error_mean'] + data['avg_error_std'],
                        alpha=0.2, color=color)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Average Distance to Goal (m)')
    plt.title(f'Average Goal Error - Scenario {scenario} (Averaged over {len(RUNS)} runs)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    error_plot_path = os.path.join(results_dir, f'combined_average_error_scenario_{scenario}.png')
    plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot combined cumulative collisions
    plt.figure(figsize=(12, 8))
    
    for method_key, data in method_data.items():
        realloc = data['reallocation']
        collision = data['collision']
        color = realloc_colors.get(realloc, 'black')
        linestyle = collision_styles.get(collision, '-')
        label = f'{realloc.capitalize()}/{collision}'
        
        cumulative_mean = np.cumsum(data['collisions_mean'])
        cumulative_std = np.cumsum(data['collisions_std'])
        
        plt.plot(time, cumulative_mean, label=label, 
                color=color, linestyle=linestyle, linewidth=2)
        plt.fill_between(time, 
                        cumulative_mean - cumulative_std,
                        cumulative_mean + cumulative_std,
                        alpha=0.2, color=color)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Collisions')
    plt.title(f'Cumulative Collisions (<{COLLISION_RADIUS:.2f}m) - Scenario {scenario} (Averaged over {len(RUNS)} runs)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    collisions_plot_path = os.path.join(results_dir, f'combined_cumulative_collisions_scenario_{scenario}.png')
    plt.savefig(collisions_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot minimum distances between any two bots
    plt.figure(figsize=(12, 8))
    
    for method_key, data in method_data.items():
        realloc = data['reallocation']
        collision = data['collision']
        color = realloc_colors.get(realloc, 'black')
        linestyle = collision_styles.get(collision, '-')
        label = f'{realloc.capitalize()}/{collision}'
        
        plt.plot(time, data['min_distances_mean'], label=label, 
                color=color, linestyle=linestyle, linewidth=2)
        plt.fill_between(time, 
                        data['min_distances_mean'] - data['min_distances_std'],
                        data['min_distances_mean'] + data['min_distances_std'],
                        alpha=0.2, color=color)
    
    # Add horizontal line at 0.3m to show dangerous area
    plt.axhline(y=0.3, color='red', linestyle=':', linewidth=2, alpha=0.8, label='Dangerous Area (0.3m)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Minimum Distance Between Any Two Bots (m)')
    plt.title(f'Minimum Distance Between Bots - Scenario {scenario} (Averaged over {len(RUNS)} runs)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    min_distance_plot_path = os.path.join(results_dir, f'combined_minimum_distances_scenario_{scenario}.png')
    plt.savefig(min_distance_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    """Process all scenarios, averaging across runs and combining methods."""
    total_scenarios = len(SCENARIOS)
    processed = 0
    successful = 0
    
    print(f"Processing {total_scenarios} scenarios...")
    print(f"Scenarios: {list(SCENARIOS)}")
    print(f"Reallocation Methods: {REALLOCATION_METHODS}")
    print(f"Collision Methods: {COLLISION_METHODS}")
    print(f"Runs to average: {list(RUNS)}")
    print("="*50)
    
    for scenario in SCENARIOS:
        processed += 1
        print(f"\nProcessing scenario {processed}/{total_scenarios}: Scenario {scenario}")
        if process_scenario(scenario):
            successful += 1
    
    print("="*50)
    print(f"Processing complete!")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Successfully processed: {successful}")
    print(f"Failed/Skipped: {processed - successful}")
    print(f"Results saved in: ./results/")
    print(f"Note: Each scenario now has 3 combined plots (error + collisions + minimum distances) instead of {len(REALLOCATION_METHODS) * len(COLLISION_METHODS) * len(RUNS) * 2} individual plots")

if __name__ == "__main__":
    main()
