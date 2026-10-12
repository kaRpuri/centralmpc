"""Constraint builders, including the separation normals."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from centralmpc.constraints import (
    VELOCITY_MARGIN,
    agent_pairs,
    box_constraints,
    separation_constraints,
    unit_normals,
)
from centralmpc.dynamics import CONTROL_DIM, STATE_DIM

HORIZON = 5


def test_agent_pairs_are_unique_and_ordered() -> None:
    pairs = agent_pairs(4)
    assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert len(agent_pairs(16)) == 16 * 15 // 2
    assert agent_pairs(1) == []


def test_box_constraint_count() -> None:
    """Four state bounds per step plus two control bounds per interval."""
    num_agents = 3
    X = cp.Variable((STATE_DIM * num_agents, HORIZON + 1))
    U = cp.Variable((CONTROL_DIM * num_agents, HORIZON))

    constraints = box_constraints(
        X,
        U,
        num_agents,
        HORIZON,
        np.zeros(3),
        np.ones(3),
        vel_max=1.0,
        acc_max=1.0,
    )
    expected = num_agents * (4 * (HORIZON + 1) + 2 * HORIZON)
    assert len(constraints) == expected


def test_velocity_margin_is_applied() -> None:
    """The velocity bound is vel_max times the margin, not vel_max."""
    X = cp.Variable((STATE_DIM, HORIZON + 1))
    U = cp.Variable((CONTROL_DIM, HORIZON))
    vel_max = 2.0

    problem = cp.Problem(
        cp.Maximize(X[3, 0]),
        [
            *box_constraints(
                X, U, 1, HORIZON, -np.ones(3) * 10, np.ones(3) * 10, vel_max, acc_max=1.0
            )
        ],
    )
    problem.solve(solver=cp.OSQP)
    assert np.isclose(X[3, 0].value, vel_max * VELOCITY_MARGIN, atol=1e-5)


def test_unit_normals_point_from_j_to_i_and_are_unit_length() -> None:
    reference = np.zeros((2, 3, HORIZON + 1))
    reference[0, 0, :] = 1.0
    reference[1, 0, :] = -1.0

    normals = unit_normals(reference, agent_pairs(2), HORIZON)
    normal = normals[0, 1]

    assert normal.shape == (3, HORIZON)
    assert np.allclose(np.linalg.norm(normal, axis=0), 1.0)
    assert np.allclose(normal, np.array([[1.0], [0.0], [0.0]]))


def test_unit_normals_fall_back_when_agents_coincide() -> None:
    """Coincident reference positions still give a usable half space."""
    reference = np.zeros((2, 3, HORIZON + 1))
    normals = unit_normals(reference, agent_pairs(2), HORIZON)

    assert np.allclose(normals[0, 1], np.array([[1.0], [0.0], [0.0]]))
    assert np.all(np.isfinite(normals[0, 1]))


def test_separation_constraint_counts() -> None:
    num_agents = 3
    pairs = agent_pairs(num_agents)
    X = cp.Variable((STATE_DIM * num_agents, HORIZON + 1))
    normals = {pair: cp.Parameter((3, HORIZON)) for pair in pairs}

    hard = separation_constraints(X, normals, None, HORIZON, 0.5)
    assert len(hard) == len(pairs) * HORIZON

    slack = cp.Variable((len(pairs), HORIZON))
    soft = separation_constraints(X, normals, slack, HORIZON, 0.5)
    # One extra constraint pins the slack non-negative.
    assert len(soft) == len(pairs) * HORIZON + 1


def test_separation_constraint_pushes_agents_apart() -> None:
    """With the half space active, the optimizer cannot stack two agents."""
    num_agents, min_separation = 2, 0.5
    pairs = agent_pairs(num_agents)
    X = cp.Variable((STATE_DIM * num_agents, HORIZON + 1))
    normals = {pair: cp.Parameter((3, HORIZON)) for pair in pairs}
    normals[0, 1].value = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, HORIZON))

    # Both agents want to sit at the origin.
    cost = cp.sum_squares(X[:3, :]) + cp.sum_squares(X[STATE_DIM : STATE_DIM + 3, :])
    problem = cp.Problem(
        cp.Minimize(cost), separation_constraints(X, normals, None, HORIZON, min_separation)
    )
    problem.solve(solver=cp.OSQP)

    assert problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    gap = X[0, 1:].value - X[STATE_DIM, 1:].value
    assert np.all(gap >= min_separation - 1e-4)
