#!/usr/bin/env python3
"""Compare the centralized runs against the distributed DMPC baseline.

The distributed runs come from a separate project and are not part of this
repository, so point --distributed-root at a local copy of its results tree.
Without it, only the centralized runs are scored.

Usage:
    python scripts/compare_distributed.py \
        --central-root runs --distributed-root /path/to/dmpc/results/experiments
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from centralmpc.analysis.compare import (
    ScoringOptions,
    compare_scenario,
    plot_method_errors,
    write_comparison_csv,
)

logger = logging.getLogger("compare")


def main() -> int:
    """Score every scenario and write the comparison table and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central-root", default="runs", help="centralized run tree")
    parser.add_argument("--distributed-root", help="distributed baseline run tree")
    parser.add_argument(
        "--scenarios", type=int, nargs="+", default=list(range(1, 10)), help="scenarios to score"
    )
    parser.add_argument("--output", default="results", help="output directory")
    parser.add_argument("--dt", type=float, default=0.01, help="timestep of the runs")
    parser.add_argument("--time-limit", type=float, default=10.0, help="scoring window")
    parser.add_argument("--collision-radius", type=float, default=0.2, help="collision threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    options = ScoringOptions(
        dt=args.dt, time_limit=args.time_limit, collision_radius=args.collision_radius
    )
    out = Path(args.output)
    distributed = Path(args.distributed_root) if args.distributed_root else None

    results = {}
    for scenario in args.scenarios:
        methods = compare_scenario(scenario, Path(args.central_root), distributed, options)
        if not methods:
            logger.info("scenario %d: no data", scenario)
            continue
        results[scenario] = methods
        plot_method_errors(methods, scenario, out / f"errors_scenario_{scenario}.png")

    if not results:
        logger.error("no runs scored; check --central-root")
        return 1

    write_comparison_csv(results, out / "method_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
