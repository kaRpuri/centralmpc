"""Shared fixtures: small configs that keep the test suite fast."""

from __future__ import annotations

import numpy as np
import pytest

from centralmpc.controller import ControllerConfig
from centralmpc.scenario import ScenarioConfig

WORKSPACE_MIN = (-5.0, -5.0, 0.0)
WORKSPACE_MAX = (5.0, 5.0, 3.0)


@pytest.fixture
def controller_config() -> ControllerConfig:
    """Two agents, short horizon, avoidance off."""
    return ControllerConfig(
        num_agents=2,
        horizon=8,
        dt=0.05,
        pos_min=np.array(WORKSPACE_MIN),
        pos_max=np.array(WORKSPACE_MAX),
        vel_max=1.5,
        acc_max=3.0,
        Q_pos=50.0,
        Q_vel=1.0,
        R=0.1,
        Q_terminal=20.0,
    )


@pytest.fixture
def head_on_config() -> ScenarioConfig:
    """Two agents swapping positions straight through each other.

    Noise is off and the duration is short so the test stays fast and
    deterministic. The horizon is long enough in seconds for the separation
    constraint to act: 40 steps at 0.05 s is a 2 s lookahead.
    """
    return ScenarioConfig(
        name="head_on",
        num_agents=2,
        horizon=40,
        dt=0.05,
        dt_plan=0.2,
        sim_duration=6.0,
        pos_min=WORKSPACE_MIN,
        pos_max=WORKSPACE_MAX,
        vel_max=1.5,
        acc_max=3.0,
        Q_pos=50.0,
        Q_vel=1.0,
        R=0.1,
        Q_terminal=20.0,
        goal_tolerance=0.3,
        std_position=0.0,
        std_velocity=0.0,
        motion_type="static",
        initial_positions=((-2.0, 0.0, 1.0), (2.0, 0.0, 1.0)),
        goal_positions=((2.0, 0.0, 1.0), (-2.0, 0.0, 1.0)),
        collision_radius=0.2,
        min_separation=0.5,
        seed_base=7,
    )
