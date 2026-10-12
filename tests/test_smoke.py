"""End to end: simulate, score, write figures and drive the CLI."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from centralmpc.analysis import plots
from centralmpc.analysis.metrics import run_metrics
from centralmpc.cli import main
from centralmpc.scenario import ScenarioConfig, simulate


def score(config: ScenarioConfig):
    """Run a scenario and score it over its full duration."""
    result = simulate(config, seed=config.seed_base)
    metrics = run_metrics(
        result.positions(),
        result.goal_positions(),
        config.dt,
        config.sim_duration,
        config.collision_radius,
    )
    return result, metrics


def test_head_on_pair_collides_without_avoidance(head_on_config: ScenarioConfig) -> None:
    """The baseline formulation has no inter-agent term, so the pair passes through."""
    _, metrics = score(dataclasses.replace(head_on_config, avoidance_enabled=False))

    assert metrics.min_separation < head_on_config.collision_radius
    assert metrics.total_collisions == 1


def test_minimum_separation_holds_with_avoidance(head_on_config: ScenarioConfig) -> None:
    """With the constraint on, the same pair never enters the collision radius."""
    config = dataclasses.replace(head_on_config, avoidance_enabled=True, min_separation=0.5)
    result, metrics = score(config)

    assert metrics.total_collisions == 0
    assert metrics.min_separation > config.collision_radius
    # The solver still returns a plan on the large majority of replans.
    assert result.success_rate > 0.8


def test_avoidance_improves_separation_on_the_same_scenario(
    head_on_config: ScenarioConfig,
) -> None:
    """Same scenario and seed, one flag changed."""
    _, baseline = score(dataclasses.replace(head_on_config, avoidance_enabled=False))
    _, avoiding = score(dataclasses.replace(head_on_config, avoidance_enabled=True))

    assert avoiding.min_separation > baseline.min_separation
    assert avoiding.total_collisions <= baseline.total_collisions


def test_agents_reach_their_goals(head_on_config: ScenarioConfig) -> None:
    config = dataclasses.replace(head_on_config, avoidance_enabled=True)
    result, _ = score(config)

    final = result.states[-1, :, :3]
    goals = np.array(config.goal_positions)
    distances = np.linalg.norm(final - goals, axis=1)
    assert np.all(distances < 1.0), distances


def test_figures_are_written(head_on_config: ScenarioConfig, tmp_path: Path) -> None:
    config = dataclasses.replace(head_on_config, sim_duration=1.0)
    _, metrics = score(config)

    for path in (
        plots.plot_tracking_error(metrics, tmp_path / "error.png"),
        plots.plot_min_separation(metrics, tmp_path / "separation.png", config.collision_radius),
        plots.plot_cumulative_collisions(metrics, tmp_path / "collisions.png"),
        plots.plot_scaling([4, 8], [10.0, 40.0], [1.0, 0.9], tmp_path / "scaling.png"),
    ):
        assert path.exists() and path.stat().st_size > 0


def test_animation_is_written(head_on_config: ScenarioConfig, tmp_path: Path) -> None:
    config = dataclasses.replace(head_on_config, sim_duration=1.0)
    result = simulate(config, seed=config.seed_base)
    path = plots.animate(result, tmp_path / "demo.gif", fps=5, max_frames=6)

    assert path.exists()
    assert path.stat().st_size > 0


def test_cli_run_writes_output_and_a_summary(tmp_path: Path) -> None:
    """The run command end to end on the shipped default scenario."""
    exit_code = main(
        [
            "--log-level",
            "WARNING",
            "run",
            "--config",
            "default",
            "--duration",
            "1.0",
            "--output",
            str(tmp_path),
        ]
    )
    assert exit_code == 0

    out_dir = tmp_path / "default" / "run_2"
    assert (out_dir / "trajectories.txt").exists()
    assert (out_dir / "goals.txt").exists()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["scenario"] == "default"
    assert summary["agents"] == 4
    assert summary["avoidance_enabled"] is False
    assert summary["solves"] > 0
    assert summary["solve_time_ms_median"] > 0


def test_cli_analyze_reads_back_a_run(tmp_path: Path) -> None:
    main(
        [
            "--log-level",
            "WARNING",
            "run",
            "--config",
            "default",
            "--duration",
            "1.0",
            "--output",
            str(tmp_path),
        ]
    )
    exit_code = main(
        [
            "--log-level",
            "WARNING",
            "analyze",
            str(tmp_path / "default" / "run_2"),
            "--time-limit",
            "1.0",
            "--no-plots",
        ]
    )
    assert exit_code == 0
