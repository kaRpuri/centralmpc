import os
import numpy as np
import matplotlib.pyplot as plt

# File paths (edit if needed)
TRAJ_FILE = 'trajectories.txt'
GOALS_FILE = 'goals.txt'

# Parameters
COLLISION_RADIUS = 0.2
TIME_LIMIT = 10.0  # seconds


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
    collisions_per_timestep = []
    for t in range(max_steps):
        count = 0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(trajs[i, :, t] - trajs[j, :, t])
                if dist < collision_radius:
                    count += 1
        collisions_per_timestep.append(count)
    return collisions_per_timestep

def main():
    # You may need to set dt to match your simulation
    dt = 0.01  # seconds
    trajs, N = load_trajectories(TRAJ_FILE)
    goals = load_goals(GOALS_FILE, N)
    avg_error, errors = compute_errors(trajs, goals, dt, TIME_LIMIT)
    collisions_per_timestep = count_collisions(trajs, dt, TIME_LIMIT, COLLISION_RADIUS)
    time = np.arange(len(avg_error)) * dt

    total_collisions = sum(collisions_per_timestep)
    print(f"Average error over first 10s: {np.mean(avg_error):.4f} m")
    print(f"Number of collisions (<{COLLISION_RADIUS}m): {total_collisions}")

    # Create results directory (results, results1, results2, ...)
    base_dir = 'results'
    results_dir = base_dir
    idx = 1
    while os.path.exists(results_dir):
        results_dir = f'{base_dir}{idx}'
        idx += 1
    os.makedirs(results_dir)

    # Plot average error
    plt.figure(figsize=(8, 4))
    plt.plot(time, avg_error, label='Average Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Distance to Goal (m)')
    plt.title('Average Goal Error (First 10s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    error_plot_path = os.path.join(results_dir, 'average_error.png')
    plt.savefig(error_plot_path)
    print(f"Saved average error plot to {error_plot_path}")
    plt.show()

    # Plot cumulative collisions
    plt.figure(figsize=(8, 4))
    cumulative_collisions = np.cumsum(collisions_per_timestep)
    plt.plot(time, cumulative_collisions, color='red', label='Cumulative Collisions')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Collisions')
    plt.title('Cumulative Collisions (<{:.2f}m)'.format(COLLISION_RADIUS))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    collisions_plot_path = os.path.join(results_dir, 'cumulative_collisions.png')
    plt.savefig(collisions_plot_path)
    print(f"Saved cumulative collisions plot to {collisions_plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
