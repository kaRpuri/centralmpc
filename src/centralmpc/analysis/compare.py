"""Comparison against the distributed DMPC baseline.

The five analyze_* scripts in the repository root each re-implemented the
same load, score and aggregate pipeline for a different slice of the data.
This module keeps one copy of it, parameterized by where the runs live.

The distributed runs are produced by a separate project and are not part of
this repository, so the comparison only runs when that output tree is
available locally. Pass its path with ``--distributed-root``.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import RunMetrics, aggregate, load_goals, load_trajectories, run_metrics  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DT = 0.01
DEFAULT_TIME_LIMIT = 10.0
DEFAULT_COLLISION_RADIUS = 0.2
DEFAULT_RUNS = (1, 2, 3)

# Goal motion of each recorded scenario, as the scenario runner assigned it.
SCENARIO_MOTION = {
    1: "static",
    2: "static",
    3: "static",
    4: "static",
    5: "static",
    6: "static",
    7: "translating",
    8: "circular",
    9: "circular_translating",
}

# Distributed baseline variants: goal reallocation strategy by collision method.
DISTRIBUTED_METHODS = (
    ("static", "on-demand"),
    ("static", "BVC"),
    ("reactive", "on-demand"),
    ("reactive", "BVC"),
    ("predictive", "on-demand"),
    ("predictive", "BVC"),
)


@dataclass(frozen=True)
class ScoringOptions:
    """How runs are scored when comparing methods.

    Attributes:
        dt: Timestep of the recorded runs, in seconds.
        time_limit: Scoring window from the start of each run, in seconds.
        collision_radius: Separation below which a pair counts as colliding.
        runs: Run numbers to average over.
    """

    dt: float = DEFAULT_DT
    time_limit: float = DEFAULT_TIME_LIMIT
    collision_radius: float = DEFAULT_COLLISION_RADIUS
    runs: tuple[int, ...] = DEFAULT_RUNS


def score_run_dir(directory: str | Path, options: ScoringOptions) -> RunMetrics | None:
    """Score one run directory holding trajectories.txt and goals.txt.

    Args:
        directory: Directory to read.
        options: Scoring options.

    Returns:
        The metrics, or None if the files are missing or malformed.
    """
    directory = Path(directory)
    traj, goals = directory / "trajectories.txt", directory / "goals.txt"
    if not traj.exists() or not goals.exists():
        return None
    try:
        positions, num_agents = load_trajectories(traj)
        goal_positions = load_goals(goals, num_agents)
    except (ValueError, IndexError) as exc:
        logger.warning("skipping %s: %s", directory, exc)
        return None
    return run_metrics(
        positions, goal_positions, options.dt, options.time_limit, options.collision_radius
    )


def score_method(
    run_dirs: list[Path], options: ScoringOptions, label: str
) -> dict[str, object] | None:
    """Score and average every available run of one method.

    Args:
        run_dirs: Candidate run directories.
        options: Scoring options.
        label: Name used in log messages.

    Returns:
        The aggregate, or None if no run could be scored.
    """
    scored = [m for d in run_dirs if (m := score_run_dir(d, options)) is not None]
    if not scored:
        logger.info("no data for %s", label)
        return None
    result = aggregate(scored)
    logger.info(
        "%s: mean error %.4f m, collisions %.1f (n=%d)",
        label,
        result["mean_error"],
        result["total_collisions_mean"],
        result["num_runs"],
    )
    return result


def central_run_dirs(root: Path, scenario: int, options: ScoringOptions) -> list[Path]:
    """Run directories of a recorded centralized scenario."""
    return [root / f"scenario_{scenario}_central" / f"run_{r}" for r in options.runs]


def distributed_run_dirs(
    root: Path, scenario: int, reallocation: str, collision: str, options: ScoringOptions
) -> list[Path]:
    """Run directories of one distributed baseline variant."""
    return [
        root / f"scenario_{scenario}" / reallocation / collision / f"run_{r}" for r in options.runs
    ]


def compare_scenario(
    scenario: int,
    central_root: Path,
    distributed_root: Path | None,
    options: ScoringOptions,
) -> dict[str, dict[str, object]]:
    """Score every method available for one scenario.

    Args:
        scenario: Scenario number.
        central_root: Directory holding ``scenario_<n>_central/run_<r>``.
        distributed_root: Directory holding the baseline tree, or None to
            score the centralized runs alone.
        options: Scoring options.

    Returns:
        Aggregates keyed by method label. Empty if nothing could be scored.
    """
    methods: dict[str, dict[str, object]] = {}

    central = score_method(
        central_run_dirs(central_root, scenario, options), options, "Centralized MPC"
    )
    if central is not None:
        methods["Centralized MPC"] = central

    if distributed_root is not None:
        for reallocation, collision in DISTRIBUTED_METHODS:
            label = f"{reallocation.capitalize()}/{collision}"
            scored = score_method(
                distributed_run_dirs(distributed_root, scenario, reallocation, collision, options),
                options,
                label,
            )
            if scored is not None:
                methods[label] = scored

    return methods


def plot_method_errors(
    methods: dict[str, dict[str, object]], scenario: int, path: str | Path
) -> Path:
    """Mean tracking error over time for every scored method."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data in methods.items():
        mean = np.asarray(data["error_mean"])
        time = np.arange(len(mean)) * float(data["dt"])
        ax.plot(time, mean, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean distance to goal (m)")
    ax.set_title(f"Scenario {scenario} ({SCENARIO_MOTION.get(scenario, 'unknown')})")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def write_comparison_csv(
    results: dict[int, dict[str, dict[str, object]]], path: str | Path
) -> Path:
    """Write mean error and collision counts per scenario and method.

    Args:
        results: Aggregates keyed by scenario then method label.
        path: Destination CSV.

    Returns:
        The path written.
    """
    labels = sorted({label for methods in results.values() for label in methods})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scenario", "motion"]
            + [f"{label} error (m)" for label in labels]
            + [f"{label} collisions" for label in labels]
        )
        for scenario in sorted(results):
            methods = results[scenario]
            writer.writerow(
                [scenario, SCENARIO_MOTION.get(scenario, "unknown")]
                + [
                    f"{methods[label]['mean_error']:.4f}" if label in methods else ""
                    for label in labels
                ]
                + [
                    f"{methods[label]['total_collisions_mean']:.1f}" if label in methods else ""
                    for label in labels
                ]
            )
    logger.info("wrote %s", path)
    return path
