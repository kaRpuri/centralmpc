"""
Centralized MPC - With Multiple Goal Motion Patterns (FIXED)
"""

import numpy as np
import cvxpy as cp
from typing import Tuple
import time

# ============================================================
# CONFIGURATION: Choose goal motion type here
# ============================================================
# Options: "static", "circular", "translating", "circular_translating"
GOAL_MOTION_TYPE = "circular"  # <-- CHANGE THIS TO SWITCH PATTERNS
# ============================================================

class CentralizedMPC:
    """Centralized MPC controller"""
    
    def __init__(self, config: dict):
        self.N = config['num_agents']
        self.dim = 3
        self.horizon = config['horizon']
        self.dt = config['dt']
        
        self.pos_min = np.array(config['pos_min'])
        self.pos_max = np.array(config['pos_max'])
        self.vel_max = config['vel_max']
        self.acc_max = config['acc_max']
        
        self.Q_pos = config['Q_pos']
        self.Q_vel = config['Q_vel']
        self.R = config['R']
        self.Q_terminal = config.get('Q_terminal', 10.0)
        
        # Dynamics
        self.A = np.array([[1, 0, 0, self.dt, 0, 0],
                          [0, 1, 0, 0, self.dt, 0],
                          [0, 0, 1, 0, 0, self.dt],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        
        self.B = np.array([[0.5*self.dt**2, 0, 0],
                          [0, 0.5*self.dt**2, 0],
                          [0, 0, 0.5*self.dt**2],
                          [self.dt, 0, 0],
                          [0, self.dt, 0],
                          [0, 0, self.dt]])
        
        self._build_qp()
        
    def _build_qp(self):
        """Build QP"""
        N_agents = self.N
        N_horizon = self.horizon
        n_states = 6
        n_controls = 3
        
        # Decision variables
        self.X = cp.Variable((n_states * N_agents, N_horizon + 1))
        self.U = cp.Variable((n_controls * N_agents, N_horizon))
        
        # Parameters
        self.X0 = cp.Parameter((n_states * N_agents,))
        self.X_ref = cp.Parameter((n_states * N_agents, N_horizon + 1))
        
        # System matrices
        A_big = np.kron(np.eye(N_agents), self.A)
        B_big = np.kron(np.eye(N_agents), self.B)
        
        # Cost function
        cost = 0
        
        # Stage costs
        for k in range(N_horizon):
            for i in range(N_agents):
                idx = i * n_states
                x_err = self.X[idx:idx+3, k] - self.X_ref[idx:idx+3, k]
                v_err = self.X[idx+3:idx+6, k] - self.X_ref[idx+3:idx+6, k]
                cost += self.Q_pos * cp.sum_squares(x_err)
                cost += self.Q_vel * cp.sum_squares(v_err)
        
        # Terminal cost
        for i in range(N_agents):
            idx = i * n_states
            x_err_terminal = self.X[idx:idx+3, N_horizon] - self.X_ref[idx:idx+3, N_horizon]
            v_err_terminal = self.X[idx+3:idx+6, N_horizon] - self.X_ref[idx+3:idx+6, N_horizon]
            cost += self.Q_terminal * self.Q_pos * cp.sum_squares(x_err_terminal)
            cost += self.Q_terminal * self.Q_vel * cp.sum_squares(v_err_terminal)
        
        # Control effort
        for k in range(N_horizon):
            cost += self.R * cp.sum_squares(self.U[:, k])
        
        # Constraints
        constraints = []
        
        # Initial condition
        constraints.append(self.X[:, 0] == self.X0)
        
        # Dynamics
        for k in range(N_horizon):
            constraints.append(self.X[:, k+1] == A_big @ self.X[:, k] + B_big @ self.U[:, k])
        
        # Physical limits
        for i in range(N_agents):
            for k in range(N_horizon + 1):
                idx = i * n_states
                constraints.append(self.X[idx:idx+3, k] >= self.pos_min)
                constraints.append(self.X[idx:idx+3, k] <= self.pos_max)
                
                vel_limit = self.vel_max * 0.95
                constraints.append(self.X[idx+3:idx+6, k] >= -vel_limit)
                constraints.append(self.X[idx+3:idx+6, k] <= vel_limit)
            
            for k in range(N_horizon):
                idx_u = i * n_controls
                constraints.append(self.U[idx_u:idx_u+3, k] >= -self.acc_max)
                constraints.append(self.U[idx_u:idx_u+3, k] <= self.acc_max)
        
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        
    def solve(self, current_states: np.ndarray, goal_positions: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Solve centralized MPC"""
        x0 = current_states.flatten()
        
        x_ref = np.zeros((6 * self.N, self.horizon + 1))
        for i in range(self.N):
            for k in range(self.horizon + 1):
                x_ref[i*6:i*6+3, k] = goal_positions[i, :, k]
                x_ref[i*6+3:i*6+6, k] = 0.0
        
        if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
            return np.zeros((self.N, 3, self.horizon)), False
        
        if np.any(np.isnan(x_ref)) or np.any(np.isinf(x_ref)):
            return np.zeros((self.N, 3, self.horizon)), False
        
        self.X0.value = x0
        self.X_ref.value = x_ref
        
        try:
            start = time.time()
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1e-3,
                eps_rel=1e-3,
                max_iter=10000,
                adaptive_rho=True,
                polish=True
            )
            solve_time = (time.time() - start) * 1000
            
            if self.problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                controls = self.U.value.reshape((self.N, 3, self.horizon))
                
                if controls is None or np.any(np.isnan(controls)) or np.any(np.isinf(controls)):
                    return np.zeros((self.N, 3, self.horizon)), False
                
                if solve_time < 200:
                    print(f"✓ MPC solved in {solve_time:.1f}ms")
                return controls, True
            else:
                print(f"✗ QP status: {self.problem.status}")
                return np.zeros((self.N, 3, self.horizon)), False
        except Exception as e:
            print(f"✗ Solver error: {e}")
            return np.zeros((self.N, 3, self.horizon)), False


class GoalManager:
    """Manages time-varying goal trajectories with multiple motion patterns"""
    
    def __init__(self, config: dict):
        self.motion_type = config.get('motion_type', 'static')
        self.circular_radius = config.get('circular_radius', 2.0)
        self.circular_omega = config.get('circular_omega', 0.5)
        self.translation_velocity = config.get('translation_velocity', 0.5)
        self.workspace_min = np.array(config['pos_min'])
        self.workspace_max = np.array(config['pos_max'])
        
    def compute_goal_trajectory(self, agent_id: int, initial_goal: np.ndarray, 
                                current_time: float, horizon: int, dt: float) -> np.ndarray:
        """Compute goal trajectory over planning horizon
        
        Motion types:
        - static: Goals don't move
        - circular: Goals orbit in a circle (bounded)
        - translating: Goals move back-and-forth (bounded)
        - circular_translating: Goals orbit while bouncing (bounded)
        """
        goal_traj = np.zeros((3, horizon + 1))
        
        if self.motion_type == 'static':
            for k in range(horizon + 1):
                goal_traj[:, k] = initial_goal
                
        elif self.motion_type == 'circular':
            # Goals orbit in a circle around origin (stays within workspace)
            original_angle = np.arctan2(initial_goal[1], initial_goal[0])
            for k in range(horizon + 1):
                t = current_time + k * dt
                angle = original_angle + t * self.circular_omega
                goal_traj[0, k] = self.circular_radius * np.cos(angle)
                goal_traj[1, k] = self.circular_radius * np.sin(angle)
                goal_traj[2, k] = initial_goal[2]
                
        elif self.motion_type == 'translating':
            # Goals move back-and-forth like a sine wave (BOUNDED)
            for k in range(horizon + 1):
                t = current_time + k * dt
                # Use sine wave to keep within bounds
                x_range = self.workspace_max[0] - self.workspace_min[0] - 1.0  # margin
                x_center = (self.workspace_max[0] + self.workspace_min[0]) / 2
                x_offset = (x_range / 2) * np.sin(t * self.translation_velocity * 0.3)
                
                goal_traj[0, k] = initial_goal[0] + x_offset
                goal_traj[1, k] = initial_goal[1]
                goal_traj[2, k] = initial_goal[2]
                
                # Clamp to workspace
                goal_traj[0, k] = np.clip(goal_traj[0, k], 
                                          self.workspace_min[0] + 0.5, 
                                          self.workspace_max[0] - 0.5)
                
        elif self.motion_type == 'circular_translating':
            # Goals orbit while center moves in a figure-8 pattern (BOUNDED)
            original_angle = np.arctan2(initial_goal[1], initial_goal[0])
            for k in range(horizon + 1):
                t = current_time + k * dt
                
                # Figure-8 pattern for center (stays bounded)
                x_range = self.workspace_max[0] - self.workspace_min[0] - 2.0
                y_range = self.workspace_max[1] - self.workspace_min[1] - 2.0
                center_x = (x_range / 2) * np.sin(t * self.translation_velocity * 0.2)
                center_y = (y_range / 2) * np.sin(2 * t * self.translation_velocity * 0.2)
                
                # Orbit around moving center
                angle = original_angle + t * self.circular_omega
                goal_traj[0, k] = center_x + self.circular_radius * 0.5 * np.cos(angle)
                goal_traj[1, k] = center_y + self.circular_radius * 0.5 * np.sin(angle)
                goal_traj[2, k] = initial_goal[2]
                
                # Clamp to workspace
                goal_traj[0, k] = np.clip(goal_traj[0, k], 
                                          self.workspace_min[0] + 0.5, 
                                          self.workspace_max[0] - 0.5)
                goal_traj[1, k] = np.clip(goal_traj[1, k], 
                                          self.workspace_min[1] + 0.5, 
                                          self.workspace_max[1] - 0.5)
                
        return goal_traj


class Simulator:
    """Multi-agent simulation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.N = config['num_agents']
        self.dt = config['dt']
        self.dt_plan = config['dt_plan']
        self.sim_duration = config['sim_duration']
        
        self.mpc = CentralizedMPC(config)
        self.goal_manager = GoalManager(config)
        
        self.states = np.array(config['initial_positions'])
        if self.states.shape[1] == 3:
            self.states = np.hstack([self.states, np.zeros((self.N, 3))])
        
        self.goals = np.array(config['goal_positions'])
        
        self.state_history = []
        self.goal_history = []
        self.time_history = []
        
        self.noise_std = config.get('noise_std', 0.001)
        self.consecutive_failures = 0
        
    def _emergency_brake(self) -> np.ndarray:
        """Generate emergency braking controls"""
        brake_controls = np.zeros((self.N, 3, self.mpc.horizon))
        for i in range(self.N):
            current_vel = self.states[i, 3:6]
            vel_norm = np.linalg.norm(current_vel)
            if vel_norm > 0.01:
                brake_direction = -current_vel / vel_norm
                brake_mag = min(self.mpc.acc_max, vel_norm / self.dt)
                for k in range(self.mpc.horizon):
                    brake_controls[i, :, k] = brake_direction * brake_mag
        return brake_controls
        
    def simulate(self):
        """Run simulation"""
        K = int(self.sim_duration / self.dt)
        replan_interval = int(self.dt_plan / self.dt)
        
        controls = None
        control_idx = 0
        current_time = 0.0

        mpc_solve_count = 0
        sim_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Centralized MPC - Goal Motion: {self.config['motion_type'].upper()}")
        print(f"  Agents: {self.N}")
        print(f"  Duration: {self.sim_duration}s")
        if self.config['motion_type'] == 'circular':
            print(f"  Circular: radius={self.config['circular_radius']}m, ω={self.config['circular_omega']}rad/s")
        elif self.config['motion_type'] == 'translating':
            print(f"  Translation: sinusoidal (bounded)")
        elif self.config['motion_type'] == 'circular_translating':
            print(f"  Circular+Translate: figure-8 pattern (bounded)")
        print(f"{'='*60}\n")
        
        for k in range(K):
            current_time = k * self.dt
            
            if k % replan_interval == 0:
                goal_trajectories = np.zeros((self.N, 3, self.mpc.horizon + 1))
                for i in range(self.N):
                    goal_trajectories[i] = self.goal_manager.compute_goal_trajectory(
                        i, self.goals[i], current_time, self.mpc.horizon, self.dt
                    )
                
                controls, success = self.mpc.solve(self.states, goal_trajectories)

                mpc_solve_count += 1
                
                if not success:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= 3:
                        controls = self._emergency_brake()
                    else:
                        controls = np.zeros((self.N, 3, self.mpc.horizon))
                else:
                    self.consecutive_failures = 0
                
                control_idx = 0
                current_goals = goal_trajectories[:, :, 0]
            
            if controls is not None and control_idx < controls.shape[2]:
                u = controls[:, :, control_idx]
            else:
                u = np.zeros((self.N, 3))
            
            for i in range(self.N):
                noise = np.random.randn(6) * self.noise_std
                self.states[i] = self.mpc.A @ self.states[i] + self.mpc.B @ u[i] + noise
                
                vel_norm = np.linalg.norm(self.states[i, 3:6])
                if vel_norm > self.mpc.vel_max:
                    self.states[i, 3:6] *= (self.mpc.vel_max / vel_norm)
            
            self.state_history.append(self.states.copy())
            self.goal_history.append(current_goals.copy())
            self.time_history.append(current_time)
            
            control_idx += 1
            
            if k % 500 == 0:
                print(f"  t={current_time:.1f}s: ", end="")
                for i in range(self.N):
                    current_goal_traj = self.goal_manager.compute_goal_trajectory(
                        i, self.goals[i], current_time, 1, self.dt
                    )
                    current_goal = current_goal_traj[:, 0]
                    dist = np.linalg.norm(self.states[i, :3] - current_goal)
                    print(f"A{i}={dist:.2f}m ", end="")
                print()
        
        print("\n✓ Simulation complete!\n")

        sim_end_time = time.time()
        sim_elapsed = sim_end_time - sim_start_time
        avg_control_freq = mpc_solve_count / sim_elapsed if sim_elapsed > 0 else 0.0
        print(f"Average control frequency: {avg_control_freq:.2f} Hz (MPC solves: {mpc_solve_count}, elapsed wall time: {sim_elapsed:.2f} s)")
        self._check_results()
        
    def _check_results(self):
        """Check results"""
        final_states = self.state_history[-1]
        
        print("="*60)
        print("RESULTS")
        print("="*60)
        
        print("\n📍 Goal Tracking:")
        final_time = self.sim_duration
        goal_tolerance = self.config.get('goal_tolerance', 0.3)
        
        for i in range(self.N):
            final_goal_traj = self.goal_manager.compute_goal_trajectory(
                i, self.goals[i], final_time, 1, self.dt
            )
            final_goal = final_goal_traj[:, 0]
            
            dist = np.linalg.norm(final_states[i, :3] - final_goal)
            vel = np.linalg.norm(final_states[i, 3:6])
            status = "✓" if dist < goal_tolerance else "✗"
            print(f"  Agent {i}: dist={dist:.4f}m, vel={vel:.3f}m/s {status}")
        
        print("\n🚨 Collision Check:")
        collision_radius = 0.5
        collision_count = 0
        min_distance = float('inf')
        
        for k, states in enumerate(self.state_history):
            for i in range(self.N):
                for j in range(i+1, self.N):
                    dist = np.linalg.norm(states[i, :3] - states[j, :3])
                    min_distance = min(min_distance, dist)
                    if dist < collision_radius:
                        collision_count += 1
        
        if collision_count == 0:
            print(f"  ✓ No collisions! (min distance: {min_distance:.3f}m)")
        else:
            print(f"  Total collisions: {collision_count}, min dist: {min_distance:.3f}m")
        
        print("="*60 + "\n")
    
    def save_trajectories(self, output_path: str = 'trajectories.txt'):
        """Save trajectories"""
        with open(output_path, 'w') as f:
            f.write(f"{self.N} {self.N} ")
            f.write(f"{self.config['pos_min'][0]} {self.config['pos_min'][1]} {self.config['pos_min'][2]} ")
            f.write(f"{self.config['pos_max'][0]} {self.config['pos_max'][1]} {self.config['pos_max'][2]}\n")
            
            po = np.array([self.state_history[0][i, :3] for i in range(self.N)]).T
            for d in range(3):
                f.write(" ".join(map(str, po[d, :])) + "\n")
            
            for d in range(3):
                f.write(" ".join(map(str, self.goals[:, d])) + "\n")
            
            for i in range(self.N):
                traj = np.array([s[i, :3] for s in self.state_history])
                for d in range(3):
                    f.write(" ".join(map(str, traj[:, d])) + "\n")
        
        print(f"✓ Trajectories saved to {output_path}")
    
    def save_goals(self, output_path: str = 'goals.txt'):
        """Save goal trajectories (for visualization)"""
        with open(output_path, 'w') as f:
            for i in range(self.N):
                goal_traj = np.array([g[i] for g in self.goal_history])
                for d in range(3):
                    f.write(" ".join(map(str, goal_traj[:, d])) + "\n")
        
        print(f"✓ Goals saved to {output_path}")


def main():
    """Main entry point"""
    
    config = {
        "num_agents": 4,
        "dim": 3,
        "horizon": 30,
        "dt": 0.01,
        "dt_plan": 0.2,
        "sim_duration": 20.0,
        
        "pos_min": [-5.0, -5.0, 0.0],
        "pos_max": [5.0, 5.0, 3.0],
        "vel_max": 1.5,
        "acc_max": 3.0,
        
        "Q_pos": 50.0,
        "Q_vel": 1.0,
        "R": 0.1,
        "Q_terminal": 20.0,
        
        "goal_tolerance": 0.4,  # Relaxed for moving goals
        "noise_std": 0.001,
        
        # Goal motion parameters
        "motion_type": GOAL_MOTION_TYPE,
        "circular_radius": 2.0,
        "circular_omega": 0.3,
        "translation_velocity": 0.5,
        
        "initial_positions": [
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [-2.0, 0.0, 1.0],
            [0.0, -2.0, 1.0]
        ],
        
        "goal_positions": [
            [-2.0, 0.0, 1.0],
            [0.0, -2.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0]
        ]
    }
    
    sim = Simulator(config)
    sim.simulate()
    sim.save_trajectories('trajectories.txt')
    sim.save_goals('goals.txt')
    
    print("\n✓ Done!\n")


if __name__ == '__main__':
    main()
