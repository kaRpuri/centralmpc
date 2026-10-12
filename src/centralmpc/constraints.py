"""Constraint builders for the centralized QP.

The box constraints are what the controller has always enforced. The pairwise
separation constraint is optional and off unless a scenario turns it on, so
the default problem is unchanged.
"""

from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np

from .dynamics import CONTROL_DIM, SPATIAL_DIM, STATE_DIM

# The original controller enforced velocity limits at 95 percent of vel_max so
# that the simulator's own velocity clamp was never the binding limit.
VELOCITY_MARGIN = 0.95


def agent_pairs(num_agents: int) -> list[tuple[int, int]]:
    """All unordered agent pairs, in ascending order.

    Args:
        num_agents: Number of agents.

    Returns:
        Pairs ``(i, j)`` with ``i < j``.
    """
    return list(itertools.combinations(range(num_agents), 2))


def initial_state_constraint(X: cp.Variable, X0: cp.Parameter) -> list[cp.Constraint]:
    """Pin the first state of the horizon to the measured state."""
    return [X[:, 0] == X0]


def dynamics_constraints(
    X: cp.Variable, U: cp.Variable, A_big: np.ndarray, B_big: np.ndarray, horizon: int
) -> list[cp.Constraint]:
    """Propagate the stacked double integrator across the horizon."""
    return [X[:, k + 1] == A_big @ X[:, k] + B_big @ U[:, k] for k in range(horizon)]


def box_constraints(
    X: cp.Variable,
    U: cp.Variable,
    num_agents: int,
    horizon: int,
    pos_min: np.ndarray,
    pos_max: np.ndarray,
    vel_max: float,
    acc_max: float,
) -> list[cp.Constraint]:
    """Workspace, velocity and acceleration limits for every agent and step.

    Args:
        X: Stacked state variable, shape (6N, horizon + 1).
        U: Stacked control variable, shape (3N, horizon).
        num_agents: Number of agents.
        horizon: Number of control intervals.
        pos_min: Lower workspace corner, shape (3,).
        pos_max: Upper workspace corner, shape (3,).
        vel_max: Velocity limit per axis, in m/s.
        acc_max: Acceleration limit per axis, in m/s^2.

    Returns:
        The constraint list, in the order the original controller built it.
    """
    constraints: list[cp.Constraint] = []
    vel_limit = vel_max * VELOCITY_MARGIN

    for i in range(num_agents):
        idx = i * STATE_DIM
        for k in range(horizon + 1):
            constraints.append(X[idx : idx + 3, k] >= pos_min)
            constraints.append(X[idx : idx + 3, k] <= pos_max)
            constraints.append(X[idx + 3 : idx + 6, k] >= -vel_limit)
            constraints.append(X[idx + 3 : idx + 6, k] <= vel_limit)

        for k in range(horizon):
            idx_u = i * CONTROL_DIM
            constraints.append(U[idx_u : idx_u + 3, k] >= -acc_max)
            constraints.append(U[idx_u : idx_u + 3, k] <= acc_max)

    return constraints


def separation_constraints(
    X: cp.Variable,
    normals: dict[tuple[int, int], cp.Parameter],
    horizon: int,
    min_separation: float,
) -> list[cp.Constraint]:
    """Linearized pairwise separation, one half space per pair and step.

    For agents ``i`` and ``j`` at step ``k``, with a unit normal ``n`` taken
    from a reference trajectory, the constraint is

        n . (p_i(k) - p_j(k)) >= min_separation

    which is linear in the decision variables and so keeps the problem a QP.
    Step 0 is skipped because it is already fixed by the initial state.

    Args:
        X: Stacked state variable, shape (6N, horizon + 1).
        normals: Unit normal parameter per pair, each shape (3, horizon).
        horizon: Number of control intervals.
        min_separation: Required centre to centre distance, in metres.

    Returns:
        One constraint per pair per step in 1..horizon.
    """
    constraints: list[cp.Constraint] = []
    for (i, j), normal in normals.items():
        pi = X[i * STATE_DIM : i * STATE_DIM + SPATIAL_DIM, :]
        pj = X[j * STATE_DIM : j * STATE_DIM + SPATIAL_DIM, :]
        for k in range(horizon):
            delta = pi[:, k + 1] - pj[:, k + 1]
            constraints.append(normal[:, k] @ delta >= min_separation)
    return constraints


def unit_normals(
    reference: np.ndarray, pairs: list[tuple[int, int]], horizon: int
) -> dict[tuple[int, int], np.ndarray]:
    """Unit separation normals for each pair, taken from a reference trajectory.

    Args:
        reference: Predicted positions, shape (N, 3, horizon + 1).
        pairs: Agent pairs to evaluate.
        horizon: Number of control intervals.

    Returns:
        A unit normal array of shape (3, horizon) per pair. Pairs that
        coincide in the reference fall back to the x axis so the half space
        stays well defined.
    """
    fallback = np.array([1.0, 0.0, 0.0])
    out: dict[tuple[int, int], np.ndarray] = {}
    for i, j in pairs:
        delta = reference[i, :, 1 : horizon + 1] - reference[j, :, 1 : horizon + 1]
        norm = np.linalg.norm(delta, axis=0)
        normals = np.empty((3, horizon))
        degenerate = norm < 1e-9
        normals[:, ~degenerate] = delta[:, ~degenerate] / norm[~degenerate]
        normals[:, degenerate] = fallback[:, None]
        out[i, j] = normals
    return out
