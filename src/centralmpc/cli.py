"""Command line entry point: ``centralmpc <command>``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .analysis import metrics as metrics_mod
from .analysis import plots
from .scenario import (
    ScenarioConfig,
    SimulationResult,
    save_goals,
    save_trajectories,
    simulate,
)

logger = logging.getLogger("centralmpc")

CONFIG_DIR = Path("configs")
OUTPUT_DIR = Path("runs")
ASSET_DIR = Path("assets")
SCALE_CONFIGS = ("scale_4_agents", "scale_8_agents", "scale_16_agents", "scale_32_agents")


def _config_path(name: str) -> Path:
    """Resolve a config name or path to a file.

    Args:
        name: Either a path to a JSON file or a name in ``configs/``.

    Returns:
        The resolved path.

    Raises:
        FileNotFoundError: If no such config exists.
    """
    candidates = [Path(name), CONFIG_DIR / name, CONFIG_DIR / f"{name}.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no config for {name!r}; tried {[str(c) for c in candidates]}")


def load_config(name: str, overrides: dict[str, object] | None = None) -> ScenarioConfig:
    """Load a scenario config, optionally overriding fields from the command line.

    Args:
        name: Config name or path.
        overrides: Field values to replace. None values are ignored.

    Returns:
        The config.
    """
    config = ScenarioConfig.from_json(_config_path(name))
    changes = {k: v for k, v in (overrides or {}).items() if v is not None}
    if changes:
        import dataclasses

        config = dataclasses.replace(config, **changes)
    return config


def _score(result: SimulationResult) -> metrics_mod.RunMetrics:
    """Score a finished run over its full duration."""
    return metrics_mod.run_metrics(
        result.positions(),
        result.goal_positions(),
        result.config.dt,
        result.config.sim_duration,
        result.config.collision_radius,
    )


def _report(result: SimulationResult, scored: metrics_mod.RunMetrics) -> dict[str, object]:
    """Build the summary dictionary printed and saved after a run."""
    solve_times = np.array(result.solve_times_ms)
    return {
        "scenario": result.config.name,
        "agents": result.config.num_agents,
        "horizon": result.config.horizon,
        "duration_s": result.config.sim_duration,
        "avoidance_enabled": result.config.avoidance_enabled,
        "min_separation_requested_m": (
            result.config.min_separation if result.config.avoidance_enabled else None
        ),
        "min_separation_observed_m": round(scored.min_separation, 4),
        "collision_radius_m": result.config.collision_radius,
        "collision_events": scored.total_collisions,
        "mean_goal_error_m": round(scored.mean_error, 4),
        "solves": result.total_solves,
        "failed_solves": result.failed_solves,
        "solve_success_rate": round(result.success_rate, 4),
        "solve_time_ms_median": round(float(np.median(solve_times)), 2),
        "solve_time_ms_p95": round(float(np.percentile(solve_times, 95)), 2),
    }


def command_run(args: argparse.Namespace) -> None:
    """Run one scenario and write its trajectories, goals and summary."""
    config = load_config(
        args.config,
        {
            "sim_duration": args.duration,
            "avoidance_enabled": args.avoidance,
            "min_separation": args.min_separation,
            "horizon": args.horizon,
        },
    )
    result = simulate(config, seed=config.seed_base + args.run)
    scored = _score(result)

    out_dir = Path(args.output) / config.name / f"run_{args.run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_trajectories(result, out_dir / "trajectories.txt")
    save_goals(result, out_dir / "goals.txt")

    report = _report(result, scored)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if args.plots:
        plots.plot_tracking_error(scored, out_dir / "tracking_error.png")
        plots.plot_min_separation(scored, out_dir / "min_separation.png", config.collision_radius)
        plots.plot_cumulative_collisions(scored, out_dir / "collisions.png")


def command_scale(args: argparse.Namespace) -> None:
    """Run the scale configs and plot solve time and success against agent count."""
    counts: list[int] = []
    medians: list[float] = []
    rates: list[float] = []
    rows: list[dict[str, object]] = []

    for name in args.configs:
        config = load_config(name, {"sim_duration": args.duration})
        result = simulate(config, seed=config.seed_base + args.run)
        scored = _score(result)
        report = _report(result, scored)
        rows.append(report)
        counts.append(config.num_agents)
        medians.append(float(report["solve_time_ms_median"]))
        rates.append(float(report["solve_success_rate"]))
        logger.info(
            "%d agents: median solve %.1f ms, %.0f%% of replans solved",
            config.num_agents,
            medians[-1],
            100 * rates[-1],
        )

    order = np.argsort(counts)
    counts = [counts[i] for i in order]
    medians = [medians[i] for i in order]
    rates = [rates[i] for i in order]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plots.plot_scaling(counts, medians, rates, out)
    summary = Path(args.output).with_suffix(".json")
    summary.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


def command_analyze(args: argparse.Namespace) -> None:
    """Score a saved run directory and write its figures."""
    directory = Path(args.directory)
    positions, num_agents = metrics_mod.load_trajectories(directory / "trajectories.txt")
    goals = metrics_mod.load_goals(directory / "goals.txt", num_agents)
    scored = metrics_mod.run_metrics(
        positions, goals, args.dt, args.time_limit, args.collision_radius
    )

    report = {
        "directory": str(directory),
        "agents": num_agents,
        "time_limit_s": args.time_limit,
        "collision_radius_m": args.collision_radius,
        "mean_goal_error_m": round(scored.mean_error, 4),
        "collision_events": scored.total_collisions,
        "min_separation_m": round(scored.min_separation, 4),
    }
    print(json.dumps(report, indent=2))

    if args.plots:
        plots.plot_tracking_error(scored, directory / "tracking_error.png")
        plots.plot_min_separation(scored, directory / "min_separation.png", args.collision_radius)
        plots.plot_cumulative_collisions(scored, directory / "collisions.png")


def command_animate(args: argparse.Namespace) -> None:
    """Run a scenario and render it to an animated GIF."""
    config = load_config(
        args.config,
        {
            "sim_duration": args.duration,
            "avoidance_enabled": args.avoidance,
            "min_separation": args.min_separation,
            "horizon": args.horizon,
        },
    )
    result = simulate(config, seed=config.seed_base + args.run)
    scored = _score(result)
    path = plots.animate(result, args.output, fps=args.fps, max_frames=args.max_frames)
    print(
        json.dumps(
            {
                "output": str(path),
                "size_mb": round(path.stat().st_size / 1e6, 2),
                "min_separation_m": round(scored.min_separation, 4),
                "collision_events": scored.total_collisions,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser for every command."""
    parser = argparse.ArgumentParser(
        prog="centralmpc",
        description="Centralized MPC for multi-robot trajectory planning in 3D.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_scenario_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", default="default", help="config name or path")
        sub.add_argument("--run", type=int, default=2, help="run number, seeds the noise")
        sub.add_argument("--duration", type=float, help="override sim_duration, in seconds")
        sub.add_argument("--horizon", type=int, help="override the prediction horizon")
        sub.add_argument(
            "--avoidance",
            action=argparse.BooleanOptionalAction,
            help="enable the pairwise separation constraint",
        )
        sub.add_argument(
            "--min-separation",
            type=float,
            dest="min_separation",
            help="separation the constraint asks for, in metres",
        )

    run = subparsers.add_parser("run", help="simulate one scenario")
    add_scenario_options(run)
    run.add_argument("--output", default=str(OUTPUT_DIR), help="output directory")
    run.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=False, help="also write figures"
    )
    run.set_defaults(func=command_run)

    scale = subparsers.add_parser("scale", help="measure solve time against agent count")
    scale.add_argument("--configs", nargs="+", default=list(SCALE_CONFIGS), help="configs to run")
    scale.add_argument("--run", type=int, default=2, help="run number, seeds the noise")
    scale.add_argument("--duration", type=float, help="override sim_duration, in seconds")
    scale.add_argument("--output", default=str(ASSET_DIR / "scaling.png"), help="output image path")
    scale.set_defaults(func=command_scale)

    analyze = subparsers.add_parser("analyze", help="score a saved run directory")
    analyze.add_argument("directory", help="directory holding trajectories.txt and goals.txt")
    analyze.add_argument("--dt", type=float, default=0.01, help="timestep of the run")
    analyze.add_argument(
        "--time-limit", type=float, default=10.0, dest="time_limit", help="scoring window"
    )
    analyze.add_argument(
        "--collision-radius",
        type=float,
        default=0.2,
        dest="collision_radius",
        help="collision threshold, in metres",
    )
    analyze.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=True, help="write figures"
    )
    analyze.set_defaults(func=command_analyze)

    animate = subparsers.add_parser("animate", help="render a run to an animated GIF")
    add_scenario_options(animate)
    animate.add_argument("--output", default=str(ASSET_DIR / "demo.gif"), help="output GIF path")
    animate.add_argument("--fps", type=int, default=20, help="frames per second")
    animate.add_argument("--max-frames", type=int, default=200, dest="max_frames", help="frame cap")
    animate.set_defaults(func=command_animate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a command.

    Args:
        argv: Argument list, or None to read from the process arguments.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
    return 0
