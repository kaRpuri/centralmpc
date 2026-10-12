"""Each goal motion pattern, checked against its closed form."""

from __future__ import annotations

import numpy as np
import pytest

from centralmpc.goals import MOTION_TYPES, GoalMotion

WORKSPACE_MIN = np.array([-5.0, -5.0, 0.0])
WORKSPACE_MAX = np.array([5.0, 5.0, 3.0])
RADIUS = 2.0
OMEGA = 0.5
TRANSLATION = 0.4
HORIZON = 10
DT = 0.1


def motion(motion_type: str) -> GoalMotion:
    return GoalMotion(
        motion_type=motion_type,
        workspace_min=WORKSPACE_MIN,
        workspace_max=WORKSPACE_MAX,
        circular_radius=RADIUS,
        circular_omega=OMEGA,
        translation_velocity=TRANSLATION,
    )


def test_rejects_unknown_motion_type() -> None:
    with pytest.raises(ValueError, match="unknown motion_type"):
        motion("spiral")


@pytest.mark.parametrize("motion_type", MOTION_TYPES)
def test_shape_and_finiteness(motion_type: str) -> None:
    traj = motion(motion_type).trajectory(np.array([2.0, 0.0, 1.0]), 3.0, HORIZON, DT)
    assert traj.shape == (3, HORIZON + 1)
    assert np.all(np.isfinite(traj))


@pytest.mark.parametrize("motion_type", MOTION_TYPES)
def test_stays_inside_the_workspace(motion_type: str) -> None:
    """Every pattern is bounded, over a long span of absolute time."""
    goal_motion = motion(motion_type)
    for t in np.linspace(0.0, 120.0, 60):
        traj = goal_motion.trajectory(np.array([2.0, 1.0, 1.0]), float(t), HORIZON, DT)
        assert np.all(traj[:2] >= WORKSPACE_MIN[:2, None]), (motion_type, t)
        assert np.all(traj[:2] <= WORKSPACE_MAX[:2, None]), (motion_type, t)


def test_static_holds_the_goal() -> None:
    goal = np.array([1.5, -2.5, 2.0])
    traj = motion("static").trajectory(goal, 17.0, HORIZON, DT)
    assert np.array_equal(traj, np.repeat(goal[:, None], HORIZON + 1, axis=1))


def test_circular_orbits_at_the_configured_radius_and_rate() -> None:
    """The goal sits at angle atan2(y, x) + omega t on a circle of radius R."""
    goal = np.array([2.0, 0.0, 1.0])
    start_angle = 0.0

    traj = motion("circular").trajectory(goal, 0.0, HORIZON, DT)
    for k in range(HORIZON + 1):
        angle = start_angle + (k * DT) * OMEGA
        assert np.allclose(traj[:, k], [RADIUS * np.cos(angle), RADIUS * np.sin(angle), goal[2]])

    # Half a revolution puts it diametrically opposite.
    half = motion("circular").trajectory(goal, float(np.pi / OMEGA), 0, DT)
    assert np.allclose(half[:, 0], [-RADIUS, 0.0, goal[2]], atol=1e-12)
    # The radius is held exactly, and z never moves.
    assert np.allclose(np.linalg.norm(traj[:2], axis=0), RADIUS)
    assert np.all(traj[2] == goal[2])


def test_translating_is_a_sine_about_the_nominal_goal() -> None:
    """x = gx + (span / 2) sin(0.3 v t), clipped to the workspace."""
    goal = np.array([0.0, 1.0, 2.0])
    span = (WORKSPACE_MAX[0] - WORKSPACE_MIN[0]) - 1.0
    traj = motion("translating").trajectory(goal, 0.0, HORIZON, DT)

    for k in range(HORIZON + 1):
        t = k * DT
        expected_x = goal[0] + (span / 2) * np.sin(t * TRANSLATION * 0.3)
        expected_x = np.clip(expected_x, WORKSPACE_MIN[0] + 0.5, WORKSPACE_MAX[0] - 0.5)
        assert np.allclose(traj[:, k], [expected_x, goal[1], goal[2]])

    # At t = 0 the offset vanishes, so the goal is its nominal value.
    assert np.allclose(traj[:, 0], goal)
    # y and z are untouched by this pattern.
    assert np.all(traj[1] == goal[1])
    assert np.all(traj[2] == goal[2])


def test_circular_translating_orbits_a_moving_centre() -> None:
    """A half radius orbit about a centre tracing a figure eight."""
    goal = np.array([2.0, 0.0, 1.5])
    start_angle = np.arctan2(goal[1], goal[0])
    x_span = (WORKSPACE_MAX[0] - WORKSPACE_MIN[0]) - 2.0
    y_span = (WORKSPACE_MAX[1] - WORKSPACE_MIN[1]) - 2.0

    traj = motion("circular_translating").trajectory(goal, 0.0, HORIZON, DT)
    for k in range(HORIZON + 1):
        t = k * DT
        centre_x = (x_span / 2) * np.sin(t * TRANSLATION * 0.2)
        centre_y = (y_span / 2) * np.sin(2 * t * TRANSLATION * 0.2)
        angle = start_angle + t * OMEGA
        expected = [
            np.clip(
                centre_x + RADIUS * 0.5 * np.cos(angle),
                WORKSPACE_MIN[0] + 0.5,
                WORKSPACE_MAX[0] - 0.5,
            ),
            np.clip(
                centre_y + RADIUS * 0.5 * np.sin(angle),
                WORKSPACE_MIN[1] + 0.5,
                WORKSPACE_MAX[1] - 0.5,
            ),
            goal[2],
        ]
        assert np.allclose(traj[:, k], expected)

    # At t = 0 the centre is at the origin, so the goal sits on the orbit.
    assert np.allclose(traj[:2, 0], [RADIUS * 0.5 * np.cos(start_angle), 0.0], atol=1e-12)


def test_absolute_time_sets_the_phase() -> None:
    """Evaluating later in the run continues the same trajectory."""
    goal = np.array([2.0, 0.0, 1.0])
    early = motion("circular").trajectory(goal, 0.0, HORIZON, DT)
    late = motion("circular").trajectory(goal, HORIZON * DT, HORIZON, DT)
    assert np.allclose(early[:, HORIZON], late[:, 0])
