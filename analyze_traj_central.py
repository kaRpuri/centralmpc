import os
import numpy as np
import matplotlib.pyplot as plt

# Import common functions from base analysis file
from analyze_trajectories_base import load_trajectories, load_goals, compute_errors, count_collisions, compute_min_distances

# Parameters
COLLISION_RADIUS = 0.2
TIME_LIMIT = 10.0  # seconds
SCENARIOS = range(1, 10)  # 1 through 9


def process_scenario(scenario):
    """Process a single scenario from centralized MPC results."""
    print(f"\nProcessing Scenario {scenario}...")
    
    # Path to centralized MPC results
    traj_file = f'experiments/scenario_{scenario}_central/trajectories.txt'
    goals_file = f'experiments/scenario_{scenario}_central/goals.txt'
    
    # Check if files exist
    if not os.path.exists(traj_file) or not os.path.exists(goals_file):
        print(f"  No data found for scenario {scenario}")
        print(f"    Expected: {traj_file}")
        print(f"    Expected: {goals_file}")
        return False
    
    try:
        dt = 0.01  # seconds
        trajs, N = load_trajectories(traj_file)
        goals = load_goals(goals_file, N)
        avg_error, errors = compute_errors(trajs, goals, dt, TIME_LIMIT)
        collisions_per_timestep, total_collisions = count_collisions(trajs, dt, TIME_LIMIT, COLLISION_RADIUS)
        min_distances = compute_min_distances(trajs, dt, TIME_LIMIT)
        
        # Determine motion type from scenario number
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
        print(f"  Agents: {N}")
        print(f"  Avg error = {np.mean(avg_error):.4f} m")
        print(f"  Total collisions = {total_collisions}")
        print(f"  Min distance = {np.min(min_distances):.3f} m")
        
        # Create results directory
        results_dir = './results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Time array
        time = np.arange(len(avg_error)) * dt
        
        # Plot average error
        plt.figure(figsize=(10, 6))
        plt.plot(time, avg_error, label=f'Centralized MPC - {motion_type}', 
                color='blue', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Average Distance to Goal (m)')
        plt.title(f'Average Goal Error - Scenario {scenario} ({motion_type} Goals) - Centralized MPC')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        error_plot_path = os.path.join(results_dir, f'central_average_error_scenario_{scenario}.png')
        plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot cumulative collisions
        plt.figure(figsize=(10, 6))
        cumulative_collisions = np.cumsum(collisions_per_timestep)
        plt.plot(time, cumulative_collisions, color='red', linewidth=2,
                label=f'Centralized MPC - {motion_type}')
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Collisions')
        plt.title(f'Cumulative Collisions (<{COLLISION_RADIUS:.2f}m) - Scenario {scenario} ({motion_type} Goals) - Centralized MPC')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        collisions_plot_path = os.path.join(results_dir, f'central_cumulative_collisions_scenario_{scenario}.png')
        plt.savefig(collisions_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot minimum distances between any two bots
        plt.figure(figsize=(10, 6))
        plt.plot(time, min_distances, label=f'Centralized MPC - {motion_type}', 
                color='green', linewidth=2)
        
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
        
    except Exception as e:
        print(f"  Error processing scenario {scenario}: {e}")
        return False


def create_comparison_summary():
    """Create a summary comparison across all scenarios."""
    print(f"\nCreating comparison summary...")
    
    scenario_metrics = {}
    
    for scenario in SCENARIOS:
        traj_file = f'experiments/scenario_{scenario}_central/trajectories.txt'
        goals_file = f'experiments/scenario_{scenario}_central/goals.txt'
        
        if os.path.exists(traj_file) and os.path.exists(goals_file):
            try:
                dt = 0.01
                trajs, N = load_trajectories(traj_file)
                goals = load_goals(goals_file, N)
                avg_error, _ = compute_errors(trajs, goals, dt, TIME_LIMIT)
                _, total_collisions = count_collisions(trajs, dt, TIME_LIMIT, COLLISION_RADIUS)
                min_distances = compute_min_distances(trajs, dt, TIME_LIMIT)
                
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
                
                scenario_metrics[scenario] = {
                    'motion_type': motion_type,
                    'mean_error': np.mean(avg_error),
                    'total_collisions': total_collisions,
                    'min_distance': np.min(min_distances),
                    'final_error': avg_error[-1] if len(avg_error) > 0 else np.inf
                }
                
            except Exception as e:
                print(f"    Error processing scenario {scenario} for summary: {e}")
    
    if not scenario_metrics:
        print("  No data available for summary")
        return
    
    # Create summary plots
    results_dir = './results'
    
    # Summary bar plot - Mean Error by Scenario
    plt.figure(figsize=(12, 6))
    scenarios_list = list(scenario_metrics.keys())
    mean_errors = [scenario_metrics[s]['mean_error'] for s in scenarios_list]
    motion_types = [scenario_metrics[s]['motion_type'] for s in scenarios_list]
    
    colors = {'Static': 'blue', 'Translating': 'green', 'Circular': 'orange', 'Circular+Translating': 'red'}
    bar_colors = [colors.get(mt, 'gray') for mt in motion_types]
    
    plt.bar(scenarios_list, mean_errors, color=bar_colors, alpha=0.7)
    plt.xlabel('Scenario')
    plt.ylabel('Mean Error (m)')
    plt.title('Centralized MPC - Mean Goal Tracking Error by Scenario')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add motion type labels
    for i, (scenario, motion_type) in enumerate(zip(scenarios_list, motion_types)):
        plt.text(scenario, mean_errors[i] + 0.01, motion_type, 
                ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.tight_layout()
    summary_error_path = os.path.join(results_dir, 'central_summary_mean_errors.png')
    plt.savefig(summary_error_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary bar plot - Total Collisions by Scenario
    plt.figure(figsize=(12, 6))
    total_collisions_list = [scenario_metrics[s]['total_collisions'] for s in scenarios_list]
    
    plt.bar(scenarios_list, total_collisions_list, color=bar_colors, alpha=0.7)
    plt.xlabel('Scenario')
    plt.ylabel('Total Collisions')
    plt.title('Centralized MPC - Total Collisions by Scenario')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add motion type labels
    for i, (scenario, motion_type) in enumerate(zip(scenarios_list, motion_types)):
        plt.text(scenario, total_collisions_list[i] + 0.5, motion_type, 
                ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.tight_layout()
    summary_collisions_path = os.path.join(results_dir, 'central_summary_total_collisions.png')
    plt.savefig(summary_collisions_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Summary plots saved to {results_dir}/")


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
    print("\nGenerated files:")
    print("  Individual scenario plots:")
    for scenario in SCENARIOS:
        if os.path.exists(f'experiments/scenario_{scenario}_central/trajectories.txt'):
            print(f"    - central_*_scenario_{scenario}.png")
    print("  Summary plots:")
    print("    - central_summary_mean_errors.png")
    print("    - central_summary_total_collisions.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
