import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters
COLLISION_RADIUS = 0.2
TIME_LIMIT = 10.0  # seconds
SCENARIOS = range(1, 10)  # 1 through 9
METHODS = ['static', 'predictive', 'reactive']
RUNS = range(1, 4)  # 1 through 3


def load_trajectories(traj_file):
    with open(traj_file, 'r') as f:
        header = f.readline().split()
        N = int(header[0])
        pos_min = list(map(float, header[2:5]))
        pos_max = list(map(float, header[5:8]))
        # Initial positions (3 lines)
        init_pos = np.array([list(map(float, f.readline().split())) for _ in range(3)])
        # Goal positions (3 lines)
        goal_pos = np.array([list(map(float, f.readline().split())) for _ in range(3)])
        # Trajectories: N agents, each with 3 lines (x, y, z)
        trajs = []
        for i in range(N):
            agent_traj = np.array([list(map(float, f.readline().split())) for _ in range(3)])
            trajs.append(agent_traj)
        # Shape: (N, 3, T)
        trajs = np.array(trajs)
    return trajs, N


def load_goals(goals_file, N):
    import sys
    with open(goals_file, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        print(f"Error: {goals_file} is empty. Please run the simulation to generate this file.")
        sys.exit(1)
    if len(lines) % (N * 3) != 0:
        print(f"Error: {goals_file} does not have the expected number of lines. Check if the file is complete.")
        sys.exit(1)
    T = len(lines[0].split())
    goals = np.zeros((N, 3, T))
    for i in range(N):
        for d in range(3):
            vals = list(map(float, lines[i*3+d].split()))
            if len(vals) != T:
                print(f"Error: Line {i*3+d+1} in {goals_file} does not have the expected {T} values.")
                sys.exit(1)
            goals[i, d, :] = vals
    return goals

def compute_errors(trajs, goals, dt, time_limit):
    N, _, T = trajs.shape
    max_steps = min(int(time_limit / dt), T)
    errors = np.zeros((N, max_steps))
    for t in range(max_steps):
        for i in range(N):
            errors[i, t] = np.linalg.norm(trajs[i, :, t] - goals[i, :, t])
    avg_error = np.mean(errors, axis=0)
    return avg_error, errors

def count_collisions(trajs, dt, time_limit, collision_radius):
    N, _, T = trajs.shape
    max_steps = min(int(time_limit / dt), T)
    in_collision = np.zeros((N, N), dtype=bool)
    collision_events = set()
    collisions_per_timestep = []
    for t in range(max_steps):
        new_events = 0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(trajs[i, :, t] - trajs[j, :, t])
                if dist < collision_radius:
                    if not in_collision[i, j]:
                        # New collision event
                        collision_events.add((i, j))
                        in_collision[i, j] = True
                        new_events += 1
                else:
                    in_collision[i, j] = False
        collisions_per_timestep.append(new_events)
    return collisions_per_timestep, len(collision_events)

def process_scenario_run(scenario, method, run):
    """Process a single scenario, method, and run combination. Returns metrics data."""
    traj_file = f'../online_dmpc/cpp/results/experiments/scenario_{scenario}/{method}/run_{run}/trajectories.txt'
    goals_file = f'../online_dmpc/cpp/results/experiments/scenario_{scenario}/{method}/run_{run}/goals.txt'
    
    # Check if files exist
    if not os.path.exists(traj_file) or not os.path.exists(goals_file):
        return None
    
    try:
        dt = 0.01  # seconds
        trajs, N = load_trajectories(traj_file)
        goals = load_goals(goals_file, N)
        avg_error, errors = compute_errors(trajs, goals, dt, TIME_LIMIT)
        collisions_per_timestep, total_collisions = count_collisions(trajs, dt, TIME_LIMIT, COLLISION_RADIUS)
        
        return {
            'avg_error': avg_error,
            'collisions_per_timestep': collisions_per_timestep,
            'total_collisions': total_collisions,
            'mean_error': np.mean(avg_error),
            'dt': dt
        }
    except Exception as e:
        print(f"Error processing scenario {scenario}, method {method}, run {run}: {e}")
        return None

def process_scenario(scenario):
    """Process all methods and runs for a single scenario, create combined plots."""
    print(f"\nProcessing Scenario {scenario}...")
    
    # Store data for all methods
    method_data = {}
    
    for method in METHODS:
        run_data = []
        for run in RUNS:
            data = process_scenario_run(scenario, method, run)
            if data is not None:
                run_data.append(data)
        
        if run_data:
            # Average across runs
            avg_errors = np.array([data['avg_error'] for data in run_data])
            collisions_per_timestep = np.array([data['collisions_per_timestep'] for data in run_data])
            
            # Ensure all arrays have the same length by taking the minimum
            min_length = min(len(arr) for arr in avg_errors)
            avg_errors = np.array([arr[:min_length] for arr in avg_errors])
            collisions_per_timestep = np.array([arr[:min_length] for arr in collisions_per_timestep])
            
            method_data[method] = {
                'avg_error_mean': np.mean(avg_errors, axis=0),
                'avg_error_std': np.std(avg_errors, axis=0),
                'collisions_mean': np.mean(collisions_per_timestep, axis=0),
                'collisions_std': np.std(collisions_per_timestep, axis=0),
                'total_collisions_mean': np.mean([data['total_collisions'] for data in run_data]),
                'mean_error': np.mean([data['mean_error'] for data in run_data]),
                'dt': run_data[0]['dt']
            }
            print(f"  {method.capitalize()}: Avg error = {method_data[method]['mean_error']:.4f} m, "
                  f"Total collisions = {method_data[method]['total_collisions_mean']:.1f}")
    
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
    plt.figure(figsize=(10, 6))
    colors = {'static': 'blue', 'predictive': 'green', 'reactive': 'red'}
    
    for method, data in method_data.items():
        plt.plot(time, data['avg_error_mean'], label=f'{method.capitalize()}', 
                color=colors.get(method, 'black'), linewidth=2)
        plt.fill_between(time, 
                        data['avg_error_mean'] - data['avg_error_std'],
                        data['avg_error_mean'] + data['avg_error_std'],
                        alpha=0.3, color=colors.get(method, 'black'))
    
    plt.xlabel('Time (s)')
    plt.ylabel('Average Distance to Goal (m)')
    plt.title(f'Average Goal Error - Scenario {scenario} (Averaged over {len(RUNS)} runs)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    error_plot_path = os.path.join(results_dir, f'combined_average_error_scenario_{scenario}.png')
    plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot combined cumulative collisions
    plt.figure(figsize=(10, 6))
    
    for method, data in method_data.items():
        cumulative_mean = np.cumsum(data['collisions_mean'])
        cumulative_std = np.cumsum(data['collisions_std'])
        
        plt.plot(time, cumulative_mean, label=f'{method.capitalize()}', 
                color=colors.get(method, 'black'), linewidth=2)
        plt.fill_between(time, 
                        cumulative_mean - cumulative_std,
                        cumulative_mean + cumulative_std,
                        alpha=0.3, color=colors.get(method, 'black'))
    
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Collisions')
    plt.title(f'Cumulative Collisions (<{COLLISION_RADIUS:.2f}m) - Scenario {scenario} (Averaged over {len(RUNS)} runs)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    collisions_plot_path = os.path.join(results_dir, f'combined_cumulative_collisions_scenario_{scenario}.png')
    plt.savefig(collisions_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    """Process all scenarios, averaging across runs and combining methods."""
    total_scenarios = len(SCENARIOS)
    processed = 0
    successful = 0
    
    print(f"Processing {total_scenarios} scenarios...")
    print(f"Scenarios: {list(SCENARIOS)}")
    print(f"Methods: {METHODS}")
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
    print(f"Note: Each scenario now has 2 combined plots (error + collisions) instead of {len(METHODS) * len(RUNS) * 2} individual plots")

if __name__ == "__main__":
    main()
