import numpy as np
import cvxpy as cp
from scipy.special import comb


np.random.seed(42)


N = 4
T = 30.0
dt = 0.1
K = int(T / dt)
min_distance = 0.6


n_control_points = 6
bezier_degree = n_control_points - 1


# FURTHER REDUCED NOISE
process_noise_std = 0.004
wind = np.array([0.002, -0.001, 0.0])


po = np.array([
    [0.0, 0.0, 1.0],
    [2.0, 0.0, 1.0],
    [0.0, 2.0, 1.0],
    [2.0, 2.0, 1.0]
])


pf = np.array([
    [2.0, 2.0, 1.0],
    [0.0, 2.0, 1.0],
    [2.0, 0.0, 1.0],
    [0.0, 0.0, 1.0]
])


print("="*70)
print("BEZIER MPC - FIXED COLLISION AVOIDANCE VERSION")
print("="*70)
print(f"Drones: {N}, Control points: {n_control_points}")
print(f"Process noise: {process_noise_std}m")
print(f"Wind: {wind}")
print(f"Improvements: Smooth APF, velocity damping, adaptive gains\n")


# ========== BEZIER FUNCTIONS ==========
def bernstein_poly(i, n, t):
    return comb(n, i, exact=True) * (t**i) * ((1-t)**(n-i))


def bezier_curve(control_points, t):
    n = len(control_points) - 1
    curve = np.zeros(3)
    for i in range(n + 1):
        curve += control_points[i] * bernstein_poly(i, n, t)
    return curve


def bezier_derivative(control_points, t):
    n = len(control_points) - 1
    deriv = np.zeros(3)
    for i in range(n):
        diff = control_points[i+1] - control_points[i]
        deriv += n * diff * bernstein_poly(i, n-1, t)
    return deriv


# ========== OPTIMIZATION ==========
print("Building optimization...")


P = cp.Variable((N, n_control_points, 3))
cost = 0.0


# Smoothness
for i in range(N):
    for k in range(n_control_points - 2):
        accel = P[i, k+2] - 2*P[i, k+1] + P[i, k]
        cost += 0.5 * cp.sum_squares(accel)


# Path deviation
for i in range(N):
    for j in range(1, n_control_points - 1):
        t_j = j / (n_control_points - 1)
        straight = po[i] + t_j * (pf[i] - po[i])
        cost += 1.0 * cp.sum_squares(P[i, j] - straight)


constraints = []


# Boundary
for i in range(N):
    constraints.append(P[i, 0] == po[i])
    constraints.append(P[i, -1] == pf[i])


# Velocity
v_max = 0.35
for i in range(N):
    for j in range(n_control_points - 1):
        delta = P[i, j+1] - P[i, j]
        constraints.append(cp.norm(delta) <= v_max / (n_control_points - 1) * T)


# ========== SOLVE ==========
print(f"Variables: {N * n_control_points * 3}")
print(f"Constraints: {len(constraints)}\n")


problem = cp.Problem(cp.Minimize(cost), constraints)


import time
start = time.time()


problem.solve(solver=cp.ECOS, verbose=False)
elapsed = time.time() - start


print(f"Status: {problem.status}")
print(f"Time: {elapsed:.2f}s\n")


if problem.status in ['optimal', 'optimal_inaccurate']:
    P_opt = P.value

    print("Executing trajectories with improved collision avoidance...\n")

    positions = po.copy()
    velocities = np.zeros((N, 3))
    trajectory = [positions.copy()]

    # Assign priorities to break symmetry during conflicts
    priorities = np.arange(N)

    # Collision avoidance parameters
    collision_buffer = 1.5
    base_repulsion_gain = 3.5  # Reduced from 5.0

    for step in range(K):
        t = (step + 1) / K

        for i in range(N):
            pos_ref = bezier_curve(P_opt[i], t)
            vel_ref = bezier_derivative(P_opt[i], t) / T

            pos_error = pos_ref - positions[i]
            vel_error = vel_ref - velocities[i]

            u_tracking = 5.0 * pos_error + 2.0 * vel_error

            # IMPROVED COLLISION AVOIDANCE
            u_avoid = np.zeros(3)
            collision_active = False
            min_dist_to_others = float('inf')

            for j in range(N):
                if i != j:
                    rel = positions[i] - positions[j]
                    dist = np.linalg.norm(rel)
                    min_dist_to_others = min(min_dist_to_others, dist)

                    collision_radius = min_distance * collision_buffer

                    if dist < collision_radius and dist > 1e-6:
                        collision_active = True
                        violation = collision_radius - dist

                        # Smooth exponential repulsion instead of linear
                        decay_factor = np.exp(-2.0 * dist / min_distance)
                        smooth_gain = base_repulsion_gain * decay_factor

                        # Priority-based yielding to break symmetry
                        if priorities[i] < priorities[j]:
                            priority_factor = 1.4  # Lower priority yields more
                        else:
                            priority_factor = 0.7  # Higher priority yields less

                        # Apply smooth force with adaptive gain
                        adaptive_gain = smooth_gain * priority_factor * (violation / min_distance)
                        u_avoid += adaptive_gain * rel / dist

                    elif dist < 1e-6:
                        # Very close - small random push instead of large one
                        u_avoid += np.random.randn(3) * 0.3

            # Combine tracking and avoidance
            u = u_tracking + u_avoid

            # Limit control input
            u_norm = np.linalg.norm(u)
            if u_norm > 1.2:
                u = u / u_norm * 1.2

            # Apply velocity damping during collisions
            if collision_active:
                # Strong damping to prevent oscillations
                velocities[i] *= 0.65

            velocities[i] += u * dt

            # Environmental disturbances
            positions[i] += np.random.normal(0, process_noise_std, 3)
            velocities[i] += wind * dt

            # Reduced gust frequency and strength
            if np.random.random() < 0.02:
                velocities[i] += np.random.normal(0, 0.01, 3)

            positions[i] += velocities[i] * dt

            # Goal-based damping
            d = np.linalg.norm(positions[i] - pf[i])
            if d < 0.03:
                velocities[i] *= 0.6
            elif d < 0.2:
                velocities[i] *= 0.85
            elif d < 0.5:
                velocities[i] *= 0.95

        trajectory.append(positions.copy())

        if step % 60 == 0:
            errs = [np.linalg.norm(positions[i] - pf[i]) for i in range(N)]
            vels = [np.linalg.norm(velocities[i]) for i in range(N)]
            print(f"  t={step*dt:5.1f}s: avg_err={np.mean(errs):.3f}m, "
                  f"avg_vel={np.mean(vels):.3f}m/s")

    trajectory = np.array(trajectory)

    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    violations = 0
    min_sep = float('inf')
    close_calls = 0

    for step in range(len(trajectory)):
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(trajectory[step, i] - trajectory[step, j])
                min_sep = min(min_sep, dist)
                if dist < min_distance:
                    violations += 1
                elif dist < min_distance * 1.2:
                    close_calls += 1

    print(f"Min separation: {min_sep:.3f}m (required: {min_distance}m)")
    print(f"Violations: {violations}")
    print(f"Close calls: {close_calls}\n")

    print("Final State:")
    for i in range(N):
        err = np.linalg.norm(trajectory[-1, i] - pf[i])
        vel = np.linalg.norm(velocities[i])
        settled = err < 0.06 and vel < 0.04
        status = "✓" if settled else "⚠"
        print(f"  Drone {i}: error={err:.4f}m, vel={vel:.4f}m/s {status}")

    if violations == 0:
        print("\n✓✓✓ SUCCESS: Smooth collision-free trajectories! ✓✓✓")
    else:
        print(f"\n⚠ {violations} violations detected")

    # Save
    with open('trajectories.txt', 'w') as f:
        f.write(f"{N} {N} {po[:,0].min()} {po[:,1].min()} {po[:,2].min()} "
            f"{po[:,0].max()} {po[:,1].max()} {po[:,2].max()}\n")
        for d in range(3):
            f.write(" ".join([f"{po[i,d]:.4f}" for i in range(N)]) + "\n")
        for d in range(3):
            f.write(" ".join([f"{pf[i,d]:.4f}" for i in range(N)]) + "\n")
        for i in range(N):
            for d in range(3):
                f.write(" ".join([f"{trajectory[k, i, d]:.4f}" for k in range(K)]) + "\n")

    print(f"\n✓ Saved to trajectories.txt")
    print("="*70)

else:
    print(f"Failed: {problem.status}")