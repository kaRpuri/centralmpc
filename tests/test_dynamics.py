"""The discretized double integrator."""

from __future__ import annotations

import numpy as np
import pytest

from centralmpc.dynamics import CONTROL_DIM, STATE_DIM, DoubleIntegrator

DT = 0.01


@pytest.fixture
def model() -> DoubleIntegrator:
    return DoubleIntegrator(dt=DT)


def test_matrix_shapes(model: DoubleIntegrator) -> None:
    assert model.A.shape == (STATE_DIM, STATE_DIM)
    assert model.B.shape == (STATE_DIM, CONTROL_DIM)


def test_step_matches_analytic_double_integrator(model: DoubleIntegrator) -> None:
    """p' = p + v dt + a dt^2 / 2 and v' = v + a dt, on every axis."""
    state = np.array([1.0, -2.0, 0.5, 0.3, 0.1, -0.2])
    control = np.array([0.4, -0.7, 1.1])

    position, velocity = state[:3], state[3:]
    expected = np.concatenate(
        [position + velocity * DT + 0.5 * control * DT**2, velocity + control * DT]
    )

    assert np.allclose(model.step(state, control), expected, rtol=0, atol=1e-15)


def test_zero_control_coasts_at_constant_velocity(model: DoubleIntegrator) -> None:
    state = np.array([0.0, 0.0, 0.0, 1.0, -0.5, 0.25])
    steps = 100
    for _ in range(steps):
        state = model.step(state, np.zeros(CONTROL_DIM))

    assert np.allclose(state[:3], np.array([1.0, -0.5, 0.25]) * steps * DT)
    assert np.allclose(state[3:], [1.0, -0.5, 0.25])


def test_constant_acceleration_from_rest(model: DoubleIntegrator) -> None:
    """After t seconds from rest, p = a t^2 / 2 and v = a t.

    The zero order hold discretization is exact for a constant
    acceleration, so this holds with no discretization error: the per step
    half dt^2 terms sum with the velocity terms to exactly a t^2 / 2.
    """
    acceleration = np.array([2.0, 0.0, -1.0])
    state = np.zeros(STATE_DIM)
    steps = 200
    for _ in range(steps):
        state = model.step(state, acceleration)

    elapsed = steps * DT
    assert np.allclose(state[3:], acceleration * elapsed)
    assert np.allclose(state[:3], 0.5 * acceleration * elapsed**2)


def test_stacked_is_block_diagonal(model: DoubleIntegrator) -> None:
    num_agents = 3
    A_big, B_big = model.stacked(num_agents)

    assert A_big.shape == (STATE_DIM * num_agents, STATE_DIM * num_agents)
    assert B_big.shape == (STATE_DIM * num_agents, CONTROL_DIM * num_agents)
    for i in range(num_agents):
        rows = slice(i * STATE_DIM, (i + 1) * STATE_DIM)
        assert np.array_equal(A_big[rows, rows], model.A)
        assert np.array_equal(B_big[rows, i * CONTROL_DIM : (i + 1) * CONTROL_DIM], model.B)


def test_stacked_agents_do_not_interact(model: DoubleIntegrator) -> None:
    """Driving one agent leaves the others untouched."""
    A_big, B_big = model.stacked(2)
    state = np.zeros(2 * STATE_DIM)
    control = np.zeros(2 * CONTROL_DIM)
    control[:CONTROL_DIM] = [1.0, 1.0, 1.0]

    nxt = A_big @ state + B_big @ control
    assert np.any(nxt[:STATE_DIM] != 0)
    assert np.all(nxt[STATE_DIM:] == 0)
