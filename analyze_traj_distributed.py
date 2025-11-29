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

def process_scenario(scenario, method, run):
    """Process a single scenario, method, and run combination."""
    traj_file = f'../online_dmpc/cpp/results/experiments/scenario_{scenario}/{method}/run_{run}/trajectories.txt'
    goals_file = f'../online_dmpc/cpp/results/experiments/scenario_{scenario}/{method}/run_{run}/goals.txt'
    
    # Check if files exist
    if not os.path.exists(traj_file):
        print(f"Warning: {traj_file} not found, skipping...")
        return False
    if not os.path.exists(goals_file):
        print(f"Warning: {goals_file} not found, skipping...")
        return False
    
    try:
        # You may need to set dt to match your simulation
        dt = 0.01  # seconds
        trajs, N = load_trajectories(traj_file)
        goals = load_goals(goals_file, N)
        avg_error, errors = compute_errors(trajs, goals, dt, TIME_LIMIT)
        collisions_per_timestep, total_collisions = count_collisions(trajs, dt, TIME_LIMIT, COLLISION_RADIUS)
        time = np.arange(len(avg_error)) * dt

        print(f"Scenario {scenario}, Method {method}, Run {run}:")
        print(f"  Average error over first 10s: {np.mean(avg_error):.4f} m")
        print(f"  Number of collision events (<{COLLISION_RADIUS}m): {total_collisions}")

        # Create results directory if it doesn't exist
        results_dir = './results'
        os.makedirs(results_dir, exist_ok=True)

        # Plot average error
        plt.figure(figsize=(8, 4))
        plt.plot(time, avg_error, label='Average Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Average Distance to Goal (m)')
        plt.title(f'Average Goal Error - Scenario {scenario}, {method.capitalize()}, Run {run}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        error_plot_path = os.path.join(results_dir, f'average_error_scenario_{scenario}_{method}_run_{run}.png')
        plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to save memory

        # Plot cumulative collisions
        plt.figure(figsize=(8, 4))
        cumulative_collisions = np.cumsum(collisions_per_timestep)
        plt.plot(time, cumulative_collisions, color='red', label='Cumulative Collisions')
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Collisions')
        plt.title(f'Cumulative Collisions (<{COLLISION_RADIUS:.2f}m) - Scenario {scenario}, {method.capitalize()}, Run {run}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        collisions_plot_path = os.path.join(results_dir, f'cumulative_collisions_scenario_{scenario}_{method}_run_{run}.png')
        plt.savefig(collisions_plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to save memory
        
        return True
        
    except Exception as e:
        print(f"Error processing scenario {scenario}, method {method}, run {run}: {e}")
        return False

def main():
    """Process all scenarios, methods, and runs."""
    total_combinations = len(SCENARIOS) * len(METHODS) * len(RUNS)
    processed = 0
    successful = 0
    
    print(f"Processing {total_combinations} combinations...")
    print(f"Scenarios: {list(SCENARIOS)}")
    print(f"Methods: {METHODS}")
    print(f"Runs: {list(RUNS)}")
    print("="*50)
    
    for scenario in SCENARIOS:
        for method in METHODS:
            for run in RUNS:
                processed += 1
                print(f"\nProcessing {processed}/{total_combinations}: Scenario {scenario}, Method {method}, Run {run}")
                if process_scenario(scenario, method, run):
                    successful += 1
    
    print("="*50)
    print(f"Processing complete!")
    print(f"Total combinations: {total_combinations}")
    print(f"Successfully processed: {successful}")
    print(f"Failed/Skipped: {processed - successful}")
    print(f"Results saved in: ./results/")

if __name__ == "__main__":
    main()
