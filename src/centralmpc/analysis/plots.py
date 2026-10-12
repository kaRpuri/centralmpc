"""Figures and the 3D animation, all driven by real run output."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from ..scenario import SimulationResult  # noqa: E402
from .metrics import RunMetrics  # noqa: E402

logger = logging.getLogger(__name__)

FIGURE_SIZE = (8, 4)
AGENT_COLORS = plt.get_cmap("tab10")


def _finish(fig: plt.Figure, path: str | Path) -> Path:
    """Tidy up, save and close a figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def plot_tracking_error(metrics: RunMetrics, path: str | Path) -> Path:
    """Mean distance to goal over time.

    Args:
        metrics: Scored run.
        path: Destination image.

    Returns:
        The path written.
    """
    time = np.arange(len(metrics.mean_error_over_time)) * metrics.dt
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(time, metrics.mean_error_over_time, label="mean over agents")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance to goal (m)")
    ax.set_title("Goal tracking error")
    ax.grid(True)
    ax.legend()
    return _finish(fig, path)


def plot_min_separation(
    metrics: RunMetrics, path: str | Path, collision_radius: float | None = None
) -> Path:
    """Closest approach between any two agents over time.

    Args:
        metrics: Scored run.
        path: Destination image.
        collision_radius: Draw this threshold as a reference line.

    Returns:
        The path written.
    """
    time = np.arange(len(metrics.min_distances)) * metrics.dt
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(time, metrics.min_distances, label="closest pair")
    if collision_radius is not None:
        ax.axhline(
            collision_radius,
            color="red",
            linestyle="--",
            label=f"collision radius {collision_radius:g} m",
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("separation (m)")
    ax.set_title("Minimum inter-agent separation")
    ax.grid(True)
    ax.legend()
    return _finish(fig, path)


def plot_cumulative_collisions(metrics: RunMetrics, path: str | Path) -> Path:
    """Running total of collision events.

    Args:
        metrics: Scored run.
        path: Destination image.

    Returns:
        The path written.
    """
    time = np.arange(len(metrics.collisions_per_step)) * metrics.dt
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(time, np.cumsum(metrics.collisions_per_step), color="red")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative collision events")
    ax.set_title("Collision events over time")
    ax.grid(True)
    return _finish(fig, path)


def plot_scaling(
    agent_counts: list[int],
    median_solve_ms: list[float],
    success_rate: list[float],
    path: str | Path,
) -> Path:
    """Solve time and solve success against the number of agents.

    Args:
        agent_counts: Agent counts tested, ascending.
        median_solve_ms: Median QP solve time per replan, in milliseconds.
        success_rate: Fraction of replans that returned a solution.
        path: Destination image.

    Returns:
        The path written.
    """
    fig, (ax_time, ax_rate) = plt.subplots(1, 2, figsize=(11, 4))

    ax_time.plot(agent_counts, median_solve_ms, marker="o")
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xticks(agent_counts)
    ax_time.set_xticklabels([str(n) for n in agent_counts])
    ax_time.set_xlabel("agents")
    ax_time.set_ylabel("median QP solve time (ms)")
    ax_time.set_title("Solve time against problem size")
    ax_time.grid(True, which="both", alpha=0.4)

    ax_rate.plot(agent_counts, [100 * r for r in success_rate], marker="o", color="green")
    ax_rate.set_xscale("log")
    ax_rate.set_xticks(agent_counts)
    ax_rate.set_xticklabels([str(n) for n in agent_counts])
    ax_rate.set_ylim(-5, 105)
    ax_rate.set_xlabel("agents")
    ax_rate.set_ylabel("replans solved (%)")
    ax_rate.set_title("Solve success against problem size")
    ax_rate.grid(True, alpha=0.4)

    return _finish(fig, path)


def animate(
    result: SimulationResult,
    path: str | Path,
    fps: int = 20,
    max_frames: int = 200,
    trail: int = 150,
    margin: float = 0.6,
) -> Path:
    """Render a 3D animation of a run to an animated GIF.

    Frames are sampled evenly from the history so the file stays small
    regardless of how long the run was.

    Args:
        result: The run to animate.
        path: Destination GIF.
        fps: Frames per second in the output.
        max_frames: Upper bound on frames written.
        trail: Number of past steps drawn behind each agent.
        margin: Padding around the occupied volume, in metres. The axes are
            fitted to where the agents and goals actually go rather than to
            the whole workspace, which is usually much larger.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = result.config
    positions = result.states[:, :, :3]
    num_steps, num_agents = positions.shape[0], positions.shape[1]
    frames = np.linspace(0, num_steps - 1, min(max_frames, num_steps)).astype(int)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    agent_dots = [
        ax.plot([], [], [], "o", color=AGENT_COLORS(i % 10), markersize=7)[0]
        for i in range(num_agents)
    ]
    agent_trails = [
        ax.plot([], [], [], "-", color=AGENT_COLORS(i % 10), linewidth=1, alpha=0.6)[0]
        for i in range(num_agents)
    ]
    goal_dots = ax.plot([], [], [], "x", color="black", markersize=6, linestyle="none")[0]

    occupied = np.concatenate([positions.reshape(-1, 3), result.goals.reshape(-1, 3)])
    lower = np.maximum(occupied.min(axis=0) - margin, np.array(config.pos_min))
    upper = np.minimum(occupied.max(axis=0) + margin, np.array(config.pos_max))
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    title = ax.set_title("")

    def update(frame_index: int) -> list:
        t = frames[frame_index]
        start = max(0, t - trail)
        for i in range(num_agents):
            agent_dots[i].set_data([positions[t, i, 0]], [positions[t, i, 1]])
            agent_dots[i].set_3d_properties([positions[t, i, 2]])
            agent_trails[i].set_data(positions[start : t + 1, i, 0], positions[start : t + 1, i, 1])
            agent_trails[i].set_3d_properties(positions[start : t + 1, i, 2])
        goals = result.goals[t]
        goal_dots.set_data(goals[:, 0], goals[:, 1])
        goal_dots.set_3d_properties(goals[:, 2])
        separation = min(
            float(np.linalg.norm(positions[t, i] - positions[t, j]))
            for i in range(num_agents)
            for j in range(i + 1, num_agents)
        )
        title.set_text(
            f"{config.num_agents} agents, {config.motion_type}, "
            f"avoidance {'on' if config.avoidance_enabled else 'off'}\n"
            f"t = {result.times[t]:.1f} s, closest pair {separation:.2f} m"
        )
        return [*agent_dots, *agent_trails, goal_dots, title]

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    logger.info("wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path
