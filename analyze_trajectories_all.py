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
SCENARIOS = [1, 3, 7, 9] 
RUNS = range(1, 4)  # 1 through 3


def load_centralized_data(scenario):
    """Load centralized MPC data for a scenario (averaged over multiple runs)."""
    run_data = []
    
    for run in RUNS:
        traj_file = f'experiments/scenario_{scenario}_central/run_{run}/trajectories.txt'
        goals_file = f'experiments/scenario_{scenario}_central/run_{run}/goals.txt'
        
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
                print(f"Error loading centralized data for scenario {scenario}, run {run}: {e}")
    
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


def load_distributed_data(scenario, reallocation_method, collision_method):
    """Load distributed DMPC data for a scenario and method combination."""
    run_data = []
    
    for run in RUNS:
        traj_file = f'../online_dmpc/cpp/results/experiments/scenario_{scenario}/{reallocation_method}/{collision_method}/run_{run}/trajectories.txt'
        goals_file = f'../online_dmpc/cpp/results/experiments/scenario_{scenario}/{reallocation_method}/{collision_method}/run_{run}/goals.txt'
        
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
                print(f"Error loading {reallocation_method}/{collision_method} run {run} for scenario {scenario}: {e}")
    
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
        'Static/BVC': {'color': 'green', 'linestyle': '--', 'linewidth': 2},
        'Reactive/On-demand': {'color': 'red', 'linestyle': '-', 'linewidth': 2},
        'Reactive/BVC': {'color': 'red', 'linestyle': '--', 'linewidth': 2},
        'Predictive/On-demand': {'color': 'green', 'linestyle': '-', 'linewidth': 2},
        'Predictive/BVC': {'color': 'green', 'linestyle': '--', 'linewidth': 2}
    }
    return styles.get(method_name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1})


def process_scenario(scenario):
    """Process and plot all enabled methods for a single scenario."""
    print(f"\nProcessing Scenario {scenario}...")
    
    # Determine motion type
    if scenario <= 3:
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
    
    # Collect all enabled methods data
    method_data = {}
    
    # Load centralized MPC data
    if INCLUDE_CENTRALIZED_MPC:
        central_data = load_centralized_data(scenario)
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
            dist_data = load_distributed_data(scenario, realloc_method, collision_method)
            if dist_data:
                method_data[label] = dist_data
                print(f"    ✓ {label}: Avg error = {dist_data['mean_error']:.4f} m, Collisions = {dist_data['total_collisions_mean']:.1f}")
            else:
                print(f"    ✗ {label}: No data available")
    
    if not method_data:
        print(f"  No data available for scenario {scenario}")
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
    plt.title(f'Goal Tracking Performance - Scenario {scenario} ({motion_type} Goals)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    error_plot_path = os.path.join(results_dir, f'all_methods_error_scenario_{scenario}.png')
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
    plt.title(f'Collision Safety - Scenario {scenario} ({motion_type} Goals)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    collisions_plot_path = os.path.join(results_dir, f'all_methods_collisions_scenario_{scenario}.png')
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
    plt.title(f'Inter-Agent Safety - Scenario {scenario} ({motion_type} Goals)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    min_distance_plot_path = os.path.join(results_dir, f'all_methods_min_distances_scenario_{scenario}.png')
    plt.savefig(min_distance_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Plots saved for scenario {scenario}")
    return True


def create_summary_comparison():
    """Create summary comparison across all scenarios and methods."""
    print(f"\nCreating summary comparison...")
    
    scenario_summary = {}
    
    for scenario in SCENARIOS:
        scenario_metrics = {}
        
        # Motion type
        if scenario <= 3:
            motion_type = "Static"
        elif scenario == 7:
            motion_type = "Translating"
        elif scenario == 8:
            motion_type = "Circular"
        elif scenario == 9:
            motion_type = "Circular+Translating"
        else:
            motion_type = "Unknown"
        
        # Centralized MPC
        if INCLUDE_CENTRALIZED_MPC:
            central_data = load_centralized_data(scenario)
            if central_data:
                scenario_metrics['Centralized MPC'] = {
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
                dist_data = load_distributed_data(scenario, realloc_method, collision_method)
                if dist_data:
                    scenario_metrics[label] = {
                        'mean_error': dist_data['mean_error'],
                        'total_collisions': dist_data['total_collisions_mean']
                    }
        
        if scenario_metrics:
            scenario_summary[scenario] = {'motion_type': motion_type, 'methods': scenario_metrics}
    
    if not scenario_summary:
        print("  No data available for summary")
        return
    
    # Create summary plots
    results_dir = './results'
    
    # Summary plot: Mean Error by Scenario and Method
    plt.figure(figsize=(16, 10))
    
    scenarios_list = list(scenario_summary.keys())
    method_names = set()
    for scenario_data in scenario_summary.values():
        method_names.update(scenario_data['methods'].keys())
    method_names = sorted(list(method_names))
    
    x_positions = np.arange(len(scenarios_list))
    bar_width = 0.8 / len(method_names) if method_names else 0.8
    
    for i, method in enumerate(method_names):
        errors = []
        for scenario in scenarios_list:
            if method in scenario_summary[scenario]['methods']:
                errors.append(scenario_summary[scenario]['methods'][method]['mean_error'])
            else:
                errors.append(0)
        
        style = get_method_style(method)
        plt.bar(x_positions + i * bar_width, errors, bar_width, 
               label=method, color=style['color'], alpha=0.7)
    
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Mean Goal Tracking Error (m)', fontsize=12)
    plt.title('Mean Goal Tracking Error - All Methods Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x_positions + bar_width * (len(method_names)-1) / 2, scenarios_list)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    summary_error_path = os.path.join(results_dir, 'all_methods_summary_errors.png')
    plt.savefig(summary_error_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary plot: Total Collisions by Scenario and Method
    plt.figure(figsize=(16, 10))
    
    for i, method in enumerate(method_names):
        collisions = []
        for scenario in scenarios_list:
            if method in scenario_summary[scenario]['methods']:
                collisions.append(scenario_summary[scenario]['methods'][method]['total_collisions'])
            else:
                collisions.append(0)
        
        style = get_method_style(method)
        plt.bar(x_positions + i * bar_width, collisions, bar_width, 
               label=method, color=style['color'], alpha=0.7)
    
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Total Collisions', fontsize=12)
    plt.title('Total Collisions - All Methods Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x_positions + bar_width * (len(method_names)-1) / 2, scenarios_list)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    summary_collisions_path = os.path.join(results_dir, 'all_methods_summary_collisions.png')
    plt.savefig(summary_collisions_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Summary plots saved to {results_dir}/")


def create_numerical_comparison():
    """Create comprehensive numerical comparison and export to CSV/table format."""
    print(f"\nCreating numerical comparison analysis...")
    
    # Collect all data for static scenarios (1-6) and dynamic scenarios (7-9)
    static_scenarios = [1, 2, 3, 4, 5, 6]
    dynamic_scenarios = [7, 8, 9]
    
    def analyze_scenario_group(scenarios, group_name):
        """Analyze a group of scenarios and return comparison data."""
        comparison_data = []
        
        for scenario in scenarios:
            scenario_data = {'scenario': scenario}
            
            # Load centralized data
            if INCLUDE_CENTRALIZED_MPC:
                central_data = load_centralized_data(scenario)
                if central_data:
                    scenario_data['central_error'] = central_data['mean_error']
                    scenario_data['central_collisions'] = central_data['total_collisions_mean']
                    scenario_data['central_final_error'] = central_data['avg_error_mean'][-1] if len(central_data['avg_error_mean']) > 0 else np.nan
                    scenario_data['central_min_distance'] = np.mean(central_data['min_distances_mean'])
            
            # Load distributed data
            if INCLUDE_STATIC_ON_DEMAND:
                static_ondemand_data = load_distributed_data(scenario, 'static', 'on-demand')
                if static_ondemand_data:
                    scenario_data['static_ondemand_error'] = static_ondemand_data['mean_error']
                    scenario_data['static_ondemand_collisions'] = static_ondemand_data['total_collisions_mean']
                    scenario_data['static_ondemand_final_error'] = static_ondemand_data['avg_error_mean'][-1] if len(static_ondemand_data['avg_error_mean']) > 0 else np.nan
                    scenario_data['static_ondemand_min_distance'] = np.mean(static_ondemand_data['min_distances_mean'])
            
            if INCLUDE_STATIC_BVC:
                static_bvc_data = load_distributed_data(scenario, 'static', 'BVC')
                if static_bvc_data:
                    scenario_data['static_bvc_error'] = static_bvc_data['mean_error']
                    scenario_data['static_bvc_collisions'] = static_bvc_data['total_collisions_mean']
                    scenario_data['static_bvc_final_error'] = static_bvc_data['avg_error_mean'][-1] if len(static_bvc_data['avg_error_mean']) > 0 else np.nan
                    scenario_data['static_bvc_min_distance'] = np.mean(static_bvc_data['min_distances_mean'])
            
            comparison_data.append(scenario_data)
        
        return comparison_data
    
    def print_scenario_analysis(comparison_data, group_name):
        """Print analysis for a group of scenarios."""
        if not comparison_data:
            print(f"No data available for {group_name}")
            return None
        
        # Calculate percentage improvements and statistics
        print("\n" + "="*100)
        print(f"NUMERICAL COMPARISON: {group_name.upper()}")
        print("="*100)
        
        # Create detailed per-scenario table
        print(f"{'Scenario':<10} {'Central Error':<15} {'OnDemand Error':<15} {'BVC Error':<15} {'Central Coll':<15} {'OnDemand Coll':<15} {'BVC Coll':<15}")
        print("-" * 100)
        
        per_scenario_improvements = []
        
        for data in comparison_data:
            scenario = data['scenario']
            
            # Error comparisons
            central_err = data.get('central_error', np.nan)
            ondemand_err = data.get('static_ondemand_error', np.nan)
            bvc_err = data.get('static_bvc_error', np.nan)
            
            # Collision comparisons  
            central_coll = data.get('central_collisions', np.nan)
            ondemand_coll = data.get('static_ondemand_collisions', np.nan)
            bvc_coll = data.get('static_bvc_collisions', np.nan)
            
            print(f"{scenario:<10} {central_err:<15.4f} {ondemand_err:<15.4f} {bvc_err:<15.4f} {central_coll:<15.1f} {ondemand_coll:<15.1f} {bvc_coll:<15.1f}")
            
            # Calculate per-scenario improvements
            scenario_improvements = {'scenario': scenario}
            
            if not np.isnan(central_err) and not np.isnan(ondemand_err):
                central_vs_ondemand_error = ((central_err - ondemand_err) / central_err) * 100
                scenario_improvements['central_vs_ondemand_error'] = central_vs_ondemand_error
            
            if not np.isnan(central_err) and not np.isnan(bvc_err):
                central_vs_bvc_error = ((central_err - bvc_err) / central_err) * 100
                scenario_improvements['central_vs_bvc_error'] = central_vs_bvc_error
            
            if not np.isnan(ondemand_err) and not np.isnan(bvc_err):
                ondemand_vs_bvc_error = ((ondemand_err - bvc_err) / ondemand_err) * 100
                scenario_improvements['ondemand_vs_bvc_error'] = ondemand_vs_bvc_error
            
            if not np.isnan(central_coll) and not np.isnan(ondemand_coll):
                central_vs_ondemand_coll = ((central_coll - ondemand_coll) / max(central_coll, 0.1)) * 100
                scenario_improvements['central_vs_ondemand_collisions'] = central_vs_ondemand_coll
            
            if not np.isnan(central_coll) and not np.isnan(bvc_coll):
                central_vs_bvc_coll = ((central_coll - bvc_coll) / max(central_coll, 0.1)) * 100
                scenario_improvements['central_vs_bvc_collisions'] = central_vs_bvc_coll
            
            if not np.isnan(ondemand_coll) and not np.isnan(bvc_coll):
                ondemand_vs_bvc_coll = ((ondemand_coll - bvc_coll) / max(ondemand_coll, 0.1)) * 100
                scenario_improvements['ondemand_vs_bvc_collisions'] = ondemand_vs_bvc_coll
            
            per_scenario_improvements.append(scenario_improvements)
        
        # Print detailed per-scenario improvements
        print("\n" + "="*120)
        print("PER-SCENARIO IMPROVEMENTS (Positive = Method performs better than baseline)")
        print("="*120)
        print(f"{'Scenario':<10} {'C vs OD Err':<12} {'C vs BVC Err':<13} {'OD vs BVC Err':<14} {'C vs OD Coll':<13} {'C vs BVC Coll':<14} {'Winner':<10}")
        print("-" * 120)
        
        for imp in per_scenario_improvements:
            scenario = imp['scenario']
            c_od_err = imp.get('central_vs_ondemand_error', np.nan)
            c_bvc_err = imp.get('central_vs_bvc_error', np.nan)
            od_bvc_err = imp.get('ondemand_vs_bvc_error', np.nan)
            c_od_coll = imp.get('central_vs_ondemand_collisions', np.nan)
            c_bvc_coll = imp.get('central_vs_bvc_collisions', np.nan)
            
            # Determine winner based on balanced score (error + safety)
            central_score = 0
            ondemand_score = 0
            bvc_score = 0
            
            if not np.isnan(c_od_err):
                ondemand_score += -c_od_err
                central_score += c_od_err
            
            if not np.isnan(c_bvc_err):
                bvc_score += -c_bvc_err
                central_score += c_bvc_err
            
            if not np.isnan(c_od_coll):
                central_score += 0.3 * c_od_coll
                ondemand_score += 0.3 * (-c_od_coll)
            
            if not np.isnan(c_bvc_coll):
                central_score += 0.3 * c_bvc_coll
                bvc_score += 0.3 * (-c_bvc_coll)
            
            if not np.isnan(od_bvc_err):
                bvc_score += -od_bvc_err
                ondemand_score += od_bvc_err
            
            # Determine winner
            scores = {'Central': central_score, 'OnDemand': ondemand_score, 'BVC': bvc_score}
            winner = max(scores, key=scores.get)
            
            print(f"{scenario:<10} {c_od_err:<12.1f} {c_bvc_err:<13.1f} {od_bvc_err:<14.1f} {c_od_coll:<13.1f} {c_bvc_coll:<14.1f} {winner:<10}")
        
        # Calculate overall statistics
        total_improvements = {'central_vs_ondemand_error': [], 'central_vs_bvc_error': [], 'ondemand_vs_bvc_error': [],
                             'central_vs_ondemand_collisions': [], 'central_vs_bvc_collisions': [], 'ondemand_vs_bvc_collisions': []}
        
        for imp in per_scenario_improvements:
            for key in total_improvements:
                if key in imp and not np.isnan(imp[key]):
                    total_improvements[key].append(imp[key])
        
        # Print summary statistics
        print("\n" + "="*100)
        print("PERCENTAGE IMPROVEMENTS SUMMARY (Positive = Method performs better than baseline)")
        print("="*100)
        
        if total_improvements['central_vs_ondemand_error']:
            avg_improvement = np.mean(total_improvements['central_vs_ondemand_error'])
            std_improvement = np.std(total_improvements['central_vs_ondemand_error'])
            print(f"GOAL TRACKING ERROR:")
            print(f"  Centralized vs Static/On-demand:  {avg_improvement:+7.2f}% ± {std_improvement:.2f}% {'(Central better)' if avg_improvement > 0 else '(On-demand better)'}")
        
        if total_improvements['central_vs_bvc_error']:
            avg_improvement = np.mean(total_improvements['central_vs_bvc_error'])
            std_improvement = np.std(total_improvements['central_vs_bvc_error'])
            print(f"  Centralized vs Static/BVC:        {avg_improvement:+7.2f}% ± {std_improvement:.2f}% {'(Central better)' if avg_improvement > 0 else '(BVC better)'}")
        
        if total_improvements['ondemand_vs_bvc_error']:
            avg_improvement = np.mean(total_improvements['ondemand_vs_bvc_error'])
            std_improvement = np.std(total_improvements['ondemand_vs_bvc_error'])
            print(f"  Static/On-demand vs Static/BVC:   {avg_improvement:+7.2f}% ± {std_improvement:.2f}% {'(On-demand better)' if avg_improvement > 0 else '(BVC better)'}")
        
        print(f"\nCOLLISION SAFETY:")
        if total_improvements['central_vs_ondemand_collisions']:
            avg_improvement = np.mean(total_improvements['central_vs_ondemand_collisions'])
            std_improvement = np.std(total_improvements['central_vs_ondemand_collisions'])
            print(f"  Centralized vs Static/On-demand:  {avg_improvement:+7.2f}% ± {std_improvement:.2f}% {'(Central safer)' if avg_improvement > 0 else '(On-demand safer)'}")
        
        if total_improvements['central_vs_bvc_collisions']:
            avg_improvement = np.mean(total_improvements['central_vs_bvc_collisions'])
            std_improvement = np.std(total_improvements['central_vs_bvc_collisions'])
            print(f"  Centralized vs Static/BVC:        {avg_improvement:+7.2f}% ± {std_improvement:.2f}% {'(Central safer)' if avg_improvement > 0 else '(BVC safer)'}")
        
        if total_improvements['ondemand_vs_bvc_collisions']:
            avg_improvement = np.mean(total_improvements['ondemand_vs_bvc_collisions'])
            std_improvement = np.std(total_improvements['ondemand_vs_bvc_collisions'])
            print(f"  Static/On-demand vs Static/BVC:   {avg_improvement:+7.2f}% ± {std_improvement:.2f}% {'(On-demand safer)' if avg_improvement > 0 else '(BVC safer)'}")
        
        return comparison_data, per_scenario_improvements
    
    # Analyze static scenarios
    static_data = analyze_scenario_group(static_scenarios, "Static Scenarios")
    static_results = print_scenario_analysis(static_data, "Static Scenarios (1-6)")
    
    # Analyze dynamic scenarios  
    dynamic_data = analyze_scenario_group(dynamic_scenarios, "Dynamic Scenarios")
    dynamic_results = print_scenario_analysis(dynamic_data, "Dynamic Scenarios (7-9)")
    
    # Export to CSV for further analysis
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    import csv
    
    # Export static scenarios data
    if static_results and static_results[0]:
        static_csv_path = os.path.join(results_dir, 'numerical_comparison_static_scenarios.csv')
        with open(static_csv_path, 'w', newline='') as csvfile:
            fieldnames = static_results[0][0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in static_results[0]:
                writer.writerow(row)
        print(f"\nStatic scenarios data exported to: {static_csv_path}")
    
    # Export dynamic scenarios data
    if dynamic_results and dynamic_results[0]:
        dynamic_csv_path = os.path.join(results_dir, 'numerical_comparison_dynamic_scenarios.csv')
        with open(dynamic_csv_path, 'w', newline='') as csvfile:
            fieldnames = dynamic_results[0][0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in dynamic_results[0]:
                writer.writerow(row)
        print(f"Dynamic scenarios data exported to: {dynamic_csv_path}")
    
    # Create combined summary report
    report_path = os.path.join(results_dir, 'numerical_comparison_summary.txt')
    with open(report_path, 'w') as f:
        f.write("NUMERICAL COMPARISON SUMMARY - STATIC AND DYNAMIC SCENARIOS\n")
        f.write("="*70 + "\n\n")
        
        f.write("STATIC SCENARIOS (1-6):\n")
        if static_results and static_results[0]:
            f.write(f"{'Scenario':<10} {'Central Err':<12} {'OnDemand Err':<13} {'BVC Err':<10} {'Central Coll':<13} {'OnDemand Coll':<13} {'BVC Coll':<10}\n")
            f.write("-" * 80 + "\n")
            for data in static_results[0]:
                scenario = data['scenario']
                central_err = data.get('central_error', np.nan)
                ondemand_err = data.get('static_ondemand_error', np.nan)
                bvc_err = data.get('static_bvc_error', np.nan)
                central_coll = data.get('central_collisions', np.nan)
                ondemand_coll = data.get('static_ondemand_collisions', np.nan)
                bvc_coll = data.get('static_bvc_collisions', np.nan)
                f.write(f"{scenario:<10} {central_err:<12.4f} {ondemand_err:<13.4f} {bvc_err:<10.4f} {central_coll:<13.1f} {ondemand_coll:<13.1f} {bvc_coll:<10.1f}\n")
        
        f.write(f"\nDYNAMIC SCENARIOS (7-9):\n")
        if dynamic_results and dynamic_results[0]:
            f.write(f"{'Scenario':<10} {'Central Err':<12} {'OnDemand Err':<13} {'BVC Err':<10} {'Central Coll':<13} {'OnDemand Coll':<13} {'BVC Coll':<10}\n")
            f.write("-" * 80 + "\n")
            for data in dynamic_results[0]:
                scenario = data['scenario']
                central_err = data.get('central_error', np.nan)
                ondemand_err = data.get('static_ondemand_error', np.nan)
                bvc_err = data.get('static_bvc_error', np.nan)
                central_coll = data.get('central_collisions', np.nan)
                ondemand_coll = data.get('static_ondemand_collisions', np.nan)
                bvc_coll = data.get('static_bvc_collisions', np.nan)
                f.write(f"{scenario:<10} {central_err:<12.4f} {ondemand_err:<13.4f} {bvc_err:<10.4f} {central_coll:<13.1f} {ondemand_coll:<13.1f} {bvc_coll:<10.1f}\n")
    
    print(f"Combined summary report saved to: {report_path}")
    print("="*100)


def main():
    """Main analysis function."""
    print("\n" + "="*90)
    print("COMPREHENSIVE TRAJECTORY ANALYSIS - CENTRALIZED vs DISTRIBUTED")
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
    
    print(f"\nAnalyzing {len(SCENARIOS)} scenarios:")
    print("  Scenarios 1-3: Static goals (4-6 excluded as redundant)")
    print("  Scenario 7:    Translating goals") 
    print("  Scenario 8:    Circular goals")
    print("  Scenario 9:    Circular + Translating goals")
    print("="*90)
    
    processed = 0
    successful = 0
    
    for scenario in SCENARIOS:
        processed += 1
        print(f"\nProcessing scenario {processed}/{len(SCENARIOS)}: Scenario {scenario}")
        if process_scenario(scenario):
            successful += 1
    
    # Create summary comparison
    create_summary_comparison()
    
    # Create numerical comparison for static scenarios
    create_numerical_comparison()
    
    print("\n" + "="*90)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*90)
    print(f"Total scenarios: {len(SCENARIOS)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed/Skipped: {processed - successful}")
    print(f"Results saved in: ./results/")
    print("\nGenerated files:")
    print("  Per-scenario comparisons:")
    print("    - all_methods_error_scenario_X.png")
    print("    - all_methods_collisions_scenario_X.png") 
    print("    - all_methods_min_distances_scenario_X.png")
    print("  Cross-scenario summaries:")
    print("    - all_methods_summary_errors.png")
    print("    - all_methods_summary_collisions.png")
    print("  Numerical comparisons:")
    print("    - numerical_comparison_static_scenarios.csv")
    print("    - per_scenario_improvements.csv")
    print("    - numerical_comparison_summary.txt")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
