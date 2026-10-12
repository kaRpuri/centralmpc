"""Scenario config loading and the closed loop simulation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from centralmpc.scenario import (
    RUN_VARIATIONS,
    ScenarioConfig,
    SimulationResult,
    save_goals,
    save_trajectories,
    simulate,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_every_shipped_config_loads() -> None:
    configs = sorted(CONFIG_DIR.glob("*.json"))
    assert configs, f"no configs found in {CONFIG_DIR}"
    for path in configs:
        config = ScenarioConfig.from_json(path)
        assert config.num_agents == len(config.initial_positions)
        assert config.num_agents == len(config.goal_positions)
        assert config.horizon > 0
        assert config.dt > 0


def test_default_config_has_avoidance_off() -> None:
    """The shipped default reproduces the original formulation."""
    assert ScenarioConfig.from_json(CONFIG_DIR / "default.json").avoidance_enabled is False


def test_avoidance_demo_differs_from_default_only_by_the_flag() -> None:
    default = json.loads((CONFIG_DIR / "default.json").read_text())
    demo = json.loads((CONFIG_DIR / "avoidance_demo.json").read_text())
    differing = {k for k in default if default[k] != demo[k]}
    assert differing == {"name", "avoidance_enabled"}


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    data = json.loads((CONFIG_DIR / "default.json").read_text())
    data["mystery_weight"] = 1.0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="unknown config keys"):
        ScenarioConfig.from_json(path)


def test_position_count_must_match_num_agents(head_on_config: ScenarioConfig) -> None:
    with pytest.raises(ValueError, match="goal_positions has 1 entries"):
        dataclasses.replace(head_on_config, goal_positions=((0.0, 0.0, 1.0),))


def test_for_run_reproduces_the_recorded_variation(head_on_config: ScenarioConfig) -> None:
    """Runs 1 and 3 scale noise and tolerance by 0.9 and 1.1 of run 2."""
    base = dataclasses.replace(head_on_config, std_position=0.002, std_velocity=0.01)

    for run, factor in enumerate(RUN_VARIATIONS, start=1):
        varied = base.for_run(run)
        assert varied.std_position == pytest.approx(base.std_position * factor)
        assert varied.std_velocity == pytest.approx(base.std_velocity * factor)
        assert varied.goal_tolerance == pytest.approx(base.goal_tolerance * factor)

    assert base.for_run(2).std_position == base.std_position


def test_derived_objects_carry_the_config(head_on_config: ScenarioConfig) -> None:
    controller = head_on_config.controller_config()
    assert controller.num_agents == head_on_config.num_agents
    assert controller.horizon == head_on_config.horizon
    assert np.array_equal(controller.pos_min, np.array(head_on_config.pos_min))

    motion = head_on_config.goal_motion()
    assert motion.motion_type == head_on_config.motion_type


def test_simulate_shapes_and_timeline(head_on_config: ScenarioConfig) -> None:
    config = dataclasses.replace(head_on_config, sim_duration=1.0)
    result = simulate(config, seed=config.seed_base)

    steps = int(config.sim_duration / config.dt)
    assert result.states.shape == (steps, config.num_agents, 6)
    assert result.goals.shape == (steps, config.num_agents, 3)
    assert result.times.shape == (steps,)
    assert np.allclose(np.diff(result.times), config.dt)
    assert np.all(np.isfinite(result.states))
    assert result.total_solves == steps // int(config.dt_plan / config.dt)


def test_simulate_is_reproducible_for_a_seed(head_on_config: ScenarioConfig) -> None:
    config = dataclasses.replace(head_on_config, sim_duration=1.0, std_position=0.01)
    first = simulate(config, seed=11)
    second = simulate(config, seed=11)
    third = simulate(config, seed=12)

    assert np.array_equal(first.states, second.states)
    assert not np.array_equal(first.states, third.states)


def test_simulate_respects_the_speed_clamp(head_on_config: ScenarioConfig) -> None:
    config = dataclasses.replace(head_on_config, sim_duration=2.0)
    result = simulate(config, seed=config.seed_base)

    speeds = np.linalg.norm(result.states[:, :, 3:], axis=2)
    assert np.max(speeds) <= config.vel_max + 1e-9


def test_success_rate_is_a_fraction(head_on_config: ScenarioConfig) -> None:
    result = simulate(dataclasses.replace(head_on_config, sim_duration=1.0), seed=1)
    assert 0.0 <= result.success_rate <= 1.0
    assert result.failed_solves <= result.total_solves


def test_success_rate_of_an_empty_run_is_zero(head_on_config: ScenarioConfig) -> None:
    empty = SimulationResult(
        config=head_on_config,
        states=np.zeros((0, 2, 6)),
        goals=np.zeros((0, 2, 3)),
        times=np.zeros(0),
    )
    assert empty.success_rate == 0.0


def test_trajectory_files_round_trip(head_on_config: ScenarioConfig, tmp_path: Path) -> None:
    """What the writers emit is what the analysis loaders read back."""
    from centralmpc.analysis.metrics import load_goals, load_trajectories

    config = dataclasses.replace(head_on_config, sim_duration=1.0)
    result = simulate(config, seed=config.seed_base)

    save_trajectories(result, tmp_path / "trajectories.txt")
    save_goals(result, tmp_path / "goals.txt")

    positions, num_agents = load_trajectories(tmp_path / "trajectories.txt")
    goals = load_goals(tmp_path / "goals.txt", num_agents)

    assert num_agents == config.num_agents
    assert np.allclose(positions, result.positions())
    assert np.allclose(goals, result.goal_positions())
