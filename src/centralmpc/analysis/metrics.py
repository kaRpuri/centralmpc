"""Trajectory metrics shared by every analysis entry point.

These were duplicated across the five analyze_* scripts in the repository
root. The definitions here are the ones those scripts used, so numbers
computed now are comparable with the numbers already in ``results/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPATIAL_DIM = 3


def load_trajectories(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a trajectories.txt file.

    Args:
        path: File written by :func:`centralmpc.scenario.save_trajectories`.

    Returns:
        ``(positions, num_agents)`` with positions shaped (N, 3, T).
    """
    with Path(path).open() as f:
        header = f.readline().split()
        num_agents = int(header[0])
        # Three lines of initial positions, then three of goal positions.
        for _ in range(2 * SPATIAL_DIM):
            f.readline()
        trajectories = [
            np.array([list(map(float, f.readline().split())) for _ in range(SPATIAL_DIM)])
            for _ in range(num_agents)
        ]
    return np.array(trajectories), num_agents


def load_goals(path: str | Path, num_agents: int) -> np.ndarray:
    """Read a goals.txt file.

    Args:
        path: File written by :func:`centralmpc.scenario.save_goals`.
        num_agents: Number of agents the file should describe.

    Returns:
        Goal positions, shape (N, 3, T).

    Raises:
        ValueError: If the file is empty or its line count does not match.
    """
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError(f"{path} is empty; run the simulation to generate it")
    if len(lines) % (num_agents * SPATIAL_DIM) != 0:
        raise ValueError(f"{path} has {len(lines)} lines, not a multiple of {num_agents * 3}")

    steps = len(lines[0].split())
    goals = np.zeros((num_agents, SPATIAL_DIM, steps))
    for i in range(num_agents):
        for d in range(SPATIAL_DIM):
            values = list(map(float, lines[i * SPATIAL_DIM + d].split()))
            if len(values) != steps:
                raise ValueError(f"{path} line {i * 3 + d + 1} has {len(values)} of {steps} values")
            goals[i, d, :] = values
    return goals


def _step_limit(num_steps: int, dt: float, time_limit: float) -> int:
    """Number of steps to score, capped by the available history."""
    return min(int(time_limit / dt), num_steps)


def tracking_errors(
    positions: np.ndarray, goals: np.ndarray, dt: float, time_limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each agent to its goal over time.

    Args:
        positions: Agent positions, shape (N, 3, T).
        goals: Goal positions, shape (N, 3, T).
        dt: Timestep, in seconds.
        time_limit: Scoring window from the start of the run, in seconds.

    Returns:
        ``(mean_error, per_agent_error)`` with shapes (K,) and (N, K).
    """
    num_agents, _, num_steps = positions.shape
    steps = _step_limit(num_steps, dt, time_limit)
    errors = np.linalg.norm(positions[:, :, :steps] - goals[:, :, :steps], axis=1)
    return errors.mean(axis=0), errors


def count_collisions(
    positions: np.ndarray, dt: float, time_limit: float, collision_radius: float
) -> tuple[list[int], int]:
    """Count collision events, where a pair entering the radius counts once.

    A pair stays "in collision" until it separates again, so a single close
    pass is one event rather than one per timestep.

    Args:
        positions: Agent positions, shape (N, 3, T).
        dt: Timestep, in seconds.
        time_limit: Scoring window, in seconds.
        collision_radius: Separation below which a pair is in collision.

    Returns:
        ``(new_events_per_step, total_pairs_that_ever_collided)``.
    """
    num_agents, _, num_steps = positions.shape
    steps = _step_limit(num_steps, dt, time_limit)

    in_collision = np.zeros((num_agents, num_agents), dtype=bool)
    collided_pairs: set[tuple[int, int]] = set()
    per_step: list[int] = []

    for t in range(steps):
        new_events = 0
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                distance = np.linalg.norm(positions[i, :, t] - positions[j, :, t])
                if distance < collision_radius:
                    if not in_collision[i, j]:
                        collided_pairs.add((i, j))
                        in_collision[i, j] = True
                        new_events += 1
                else:
                    in_collision[i, j] = False
        per_step.append(new_events)

    return per_step, len(collided_pairs)


def min_distances(positions: np.ndarray, dt: float, time_limit: float) -> list[float]:
    """Closest approach between any two agents at each timestep.

    Args:
        positions: Agent positions, shape (N, 3, T).
        dt: Timestep, in seconds.
        time_limit: Scoring window, in seconds.

    Returns:
        One minimum pairwise distance per scored step.
    """
    num_agents, _, num_steps = positions.shape
    steps = _step_limit(num_steps, dt, time_limit)
    out: list[float] = []
    for t in range(steps):
        pairs = [
            np.linalg.norm(positions[i, :, t] - positions[j, :, t])
            for i in range(num_agents)
            for j in range(i + 1, num_agents)
        ]
        out.append(min(pairs) if pairs else float("inf"))
    return out


@dataclass
class RunMetrics:
    """Scored metrics for one run.

    Attributes:
        num_agents: Number of agents.
        dt: Timestep, in seconds.
        mean_error_over_time: Mean goal distance across agents, per step.
        mean_error: Mean goal distance over the whole scoring window, in metres.
        collisions_per_step: New collision events per step.
        total_collisions: Pairs that collided at least once.
        min_distances: Closest approach per step, in metres.
        min_separation: Smallest separation seen in the window, in metres.
    """

    num_agents: int
    dt: float
    mean_error_over_time: np.ndarray
    mean_error: float
    collisions_per_step: list[int]
    total_collisions: int
    min_distances: list[float]
    min_separation: float


def run_metrics(
    positions: np.ndarray,
    goals: np.ndarray,
    dt: float,
    time_limit: float,
    collision_radius: float,
) -> RunMetrics:
    """Score one run.

    Args:
        positions: Agent positions, shape (N, 3, T).
        goals: Goal positions, shape (N, 3, T).
        dt: Timestep, in seconds.
        time_limit: Scoring window, in seconds.
        collision_radius: Collision threshold, in metres.

    Returns:
        The scored metrics.
    """
    mean_error_over_time, _ = tracking_errors(positions, goals, dt, time_limit)
    per_step, total = count_collisions(positions, dt, time_limit, collision_radius)
    separations = min_distances(positions, dt, time_limit)
    return RunMetrics(
        num_agents=positions.shape[0],
        dt=dt,
        mean_error_over_time=mean_error_over_time,
        mean_error=float(np.mean(mean_error_over_time)),
        collisions_per_step=per_step,
        total_collisions=total,
        min_distances=separations,
        min_separation=float(min(separations)) if separations else float("inf"),
    )


def aggregate(runs: list[RunMetrics]) -> dict[str, object]:
    """Average metrics across repeated runs, truncating to the shortest run.

    Args:
        runs: Metrics from runs of the same configuration.

    Returns:
        Mean and standard deviation arrays plus scalar means.

    Raises:
        ValueError: If no runs are given.
    """
    if not runs:
        raise ValueError("no runs to aggregate")

    length = min(len(r.mean_error_over_time) for r in runs)
    errors = np.array([r.mean_error_over_time[:length] for r in runs])
    collisions = np.array([r.collisions_per_step[:length] for r in runs])
    separations = np.array([r.min_distances[:length] for r in runs])

    return {
        "num_runs": len(runs),
        "dt": runs[0].dt,
        "num_agents": runs[0].num_agents,
        "error_mean": errors.mean(axis=0),
        "error_std": errors.std(axis=0),
        "collisions_mean": collisions.mean(axis=0),
        "collisions_std": collisions.std(axis=0),
        "min_distance_mean": separations.mean(axis=0),
        "min_distance_std": separations.std(axis=0),
        "mean_error": float(errors.mean()),
        "total_collisions_mean": float(np.mean([r.total_collisions for r in runs])),
        "min_separation": float(np.min([r.min_separation for r in runs])),
    }
