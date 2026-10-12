"""The joint QP: feasibility, bound compliance and the avoidance switch."""

from __future__ import annotations

import dataclasses

import cvxpy as cp
import numpy as np

from centralmpc.constraints import VELOCITY_MARGIN, agent_pairs
from centralmpc.controller import CentralMPC, ControllerConfig


def goal_block(goals: np.ndarray, horizon: int) -> np.ndarray:
    """Hold each goal constant over the horizon, shape (N, 3, horizon + 1)."""
    return np.repeat(goals[:, :, None], horizon + 1, axis=2)


def rest_states(positions: np.ndarray) -> np.ndarray:
    """Zero velocity states at the given positions, shape (N, 6)."""
    return np.hstack([positions, np.zeros_like(positions)])


def test_two_agents_return_a_feasible_solution(controller_config: ControllerConfig) -> None:
    mpc = CentralMPC(controller_config)
    states = rest_states(np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))
    goals = goal_block(np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]), mpc.horizon)

    result = mpc.solve(states, goals)

    assert result.success
    assert result.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert result.controls.shape == (2, 3, controller_config.horizon)
    assert np.all(np.isfinite(result.controls))
    assert result.solve_time_ms > 0


def test_solution_respects_the_acceleration_bound(controller_config: ControllerConfig) -> None:
    mpc = CentralMPC(controller_config)
    # Goals far outside the workspace, so the controller saturates.
    states = rest_states(np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0]]))
    goals = goal_block(np.array([[50.0, 50.0, 50.0], [-50.0, -50.0, 0.0]]), mpc.horizon)

    result = mpc.solve(states, goals)

    assert result.success
    assert np.max(np.abs(result.controls)) <= controller_config.acc_max + 1e-4


def test_solution_respects_position_and_velocity_bounds(
    controller_config: ControllerConfig,
) -> None:
    mpc = CentralMPC(controller_config)
    states = rest_states(np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
    goals = goal_block(np.array([[50.0, 50.0, 50.0], [-50.0, -50.0, -50.0]]), mpc.horizon)

    assert mpc.solve(states, goals).success
    plan = mpc.X.value.reshape((2, 6, controller_config.horizon + 1))

    positions, velocities = plan[:, :3, :], plan[:, 3:, :]
    assert np.all(positions >= controller_config.pos_min[None, :, None] - 1e-4)
    assert np.all(positions <= controller_config.pos_max[None, :, None] + 1e-4)
    assert np.max(np.abs(velocities)) <= controller_config.vel_max * VELOCITY_MARGIN + 1e-4


def test_agent_already_at_its_goal_barely_moves(controller_config: ControllerConfig) -> None:
    mpc = CentralMPC(controller_config)
    positions = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    result = mpc.solve(rest_states(positions), goal_block(positions, mpc.horizon))

    assert result.success
    assert np.max(np.abs(result.controls)) < 1e-3


def test_non_finite_state_is_rejected_without_solving(
    controller_config: ControllerConfig,
) -> None:
    mpc = CentralMPC(controller_config)
    states = rest_states(np.array([[np.nan, 0.0, 1.0], [1.0, 0.0, 1.0]]))
    goals = goal_block(np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]), mpc.horizon)

    result = mpc.solve(states, goals)

    assert not result.success
    assert result.status == "non_finite_state"
    assert np.all(result.controls == 0)


def test_avoidance_off_adds_no_separation_constraints(
    controller_config: ControllerConfig,
) -> None:
    """The default problem is the original one, with nothing extra."""
    baseline = CentralMPC(controller_config)
    with_avoidance = CentralMPC(
        dataclasses.replace(controller_config, avoidance_enabled=True, min_separation=0.5)
    )

    pairs = len(agent_pairs(controller_config.num_agents))
    added = len(with_avoidance.problem.constraints) - len(baseline.problem.constraints)
    # One half space per pair per step, plus the slack sign constraint.
    assert added == pairs * controller_config.horizon + 1


def test_avoidance_separates_a_head_on_pair() -> None:
    """Two agents told to swap places plan around each other, not through."""
    min_separation = 0.6
    config = ControllerConfig(
        num_agents=2,
        horizon=40,
        dt=0.05,
        pos_min=np.array([-5.0, -5.0, 0.0]),
        pos_max=np.array([5.0, 5.0, 3.0]),
        vel_max=1.5,
        acc_max=3.0,
        Q_pos=50.0,
        Q_vel=1.0,
        R=0.1,
        Q_terminal=20.0,
        avoidance_enabled=True,
        min_separation=min_separation,
    )
    mpc = CentralMPC(config)
    states = rest_states(np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))
    goals = goal_block(np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]), mpc.horizon)

    result = mpc.solve(states, goals)
    assert result.success

    plan = mpc.X.value.reshape((2, 6, config.horizon + 1))[:, :3, :]
    separation = np.linalg.norm(plan[0] - plan[1], axis=0)
    # The plan keeps the pair apart, where the unconstrained plan would not.
    assert separation.min() > 0.5 * min_separation


def test_warm_start_does_not_change_the_answer(controller_config: ControllerConfig) -> None:
    """Re-solving the same problem returns the same controls."""
    mpc = CentralMPC(controller_config)
    states = rest_states(np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))
    goals = goal_block(np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]), mpc.horizon)

    first = mpc.solve(states, goals)
    second = mpc.solve(states, goals)

    assert first.success and second.success
    assert np.allclose(first.controls, second.controls, atol=1e-6)
