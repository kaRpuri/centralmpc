"""Scenario configuration and the closed loop simulation that runs one."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .controller import CentralMPC, ControllerConfig
from .dynamics import CONTROL_DIM, SPATIAL_DIM, DoubleIntegrator
from .goals import GoalMotion

logger = logging.getLogger(__name__)

# Recorded runs 1 and 3 of every scenario scaled the measurement noise and the
# goal tolerance by these factors relative to run 2, which is the base config.
RUN_VARIATIONS = (0.9, 1.0, 1.1)

# Number of consecutive failed solves before the simulator brakes instead of
# coasting on a zero command.
FAILURES_BEFORE_BRAKE = 3


@dataclass(frozen=True)
class ScenarioConfig:
    """One scenario, loaded from a JSON file in ``configs/``.

    Attributes:
        name: Scenario name, used for output directory names.
        num_agents: Number of agents.
        horizon: Controller prediction horizon, in control intervals.
        dt: Control and simulation timestep, in seconds.
        dt_plan: Replanning interval, in seconds.
        sim_duration: Simulated duration, in seconds.
        pos_min: Lower workspace corner, shape (3,).
        pos_max: Upper workspace corner, shape (3,).
        vel_max: Velocity limit per axis, in m/s.
        acc_max: Acceleration limit per axis, in m/s^2.
        Q_pos: Position tracking weight.
        Q_vel: Velocity tracking weight.
        R: Control effort weight.
        Q_terminal: Terminal multiplier on the tracking weights.
        goal_tolerance: Distance at which an agent counts as having arrived.
        std_position: Standard deviation of the position measurement noise.
        std_velocity: Standard deviation of the velocity measurement noise.
        motion_type: Goal motion pattern, see :data:`centralmpc.goals.MOTION_TYPES`.
        circular_radius: Orbit radius for the circular patterns, in metres.
        circular_omega: Orbit rate, in rad/s.
        translation_velocity: Scale on the translating patterns, in m/s.
        collision_radius: Separation below which a pair counts as a collision
            when the run is scored, in metres.
        avoidance_enabled: Whether the controller adds the separation constraint.
        min_separation: Separation the constraint asks for, in metres.
        separation_penalty: Quadratic weight on the separation slack.
        seed_base: Base seed. Run ``r`` uses ``seed_base + r``.
        initial_positions: Starting positions, shape (N, 3).
        goal_positions: Nominal goals, shape (N, 3).
    """

    name: str
    num_agents: int
    horizon: int
    dt: float
    dt_plan: float
    sim_duration: float
    pos_min: tuple[float, ...]
    pos_max: tuple[float, ...]
    vel_max: float
    acc_max: float
    Q_pos: float
    Q_vel: float
    R: float
    Q_terminal: float
    goal_tolerance: float
    std_position: float
    std_velocity: float
    motion_type: str
    initial_positions: tuple[tuple[float, ...], ...]
    goal_positions: tuple[tuple[float, ...], ...]
    circular_radius: float = 2.0
    circular_omega: float = 0.3
    translation_velocity: float = 0.2
    collision_radius: float = 0.2
    avoidance_enabled: bool = False
    min_separation: float = 0.2
    separation_penalty: float = 1.0e4
    seed_base: int = 42

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_positions", self.initial_positions),
            ("goal_positions", self.goal_positions),
        ):
            if len(value) != self.num_agents:
                raise ValueError(
                    f"{name} has {len(value)} entries but num_agents is {self.num_agents}"
                )

    @classmethod
    def from_json(cls, path: str | Path) -> ScenarioConfig:
        """Load a scenario from a JSON file.

        Args:
            path: Path to the config file.

        Returns:
            The parsed config.

        Raises:
            ValueError: If the file holds keys this version does not know.
        """
        data = json.loads(Path(path).read_text())
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        for key in ("pos_min", "pos_max"):
            data[key] = tuple(data[key])
        for key in ("initial_positions", "goal_positions"):
            data[key] = tuple(tuple(row) for row in data[key])
        return cls(**data)

    def for_run(self, run: int) -> ScenarioConfig:
        """Apply the per run variation used by the recorded experiments.

        Args:
            run: Run number, 1 based.

        Returns:
            A copy with the noise and tolerance scaled, matching the recorded
            run configs exactly.
        """
        factor = RUN_VARIATIONS[(run - 1) % len(RUN_VARIATIONS)]
        return replace_fields(
            self,
            std_position=self.std_position * factor,
            std_velocity=self.std_velocity * factor,
            goal_tolerance=self.goal_tolerance * factor,
        )

    def controller_config(self) -> ControllerConfig:
        """The subset of the scenario the controller needs."""
        return ControllerConfig(
            num_agents=self.num_agents,
            horizon=self.horizon,
            dt=self.dt,
            pos_min=np.array(self.pos_min),
            pos_max=np.array(self.pos_max),
            vel_max=self.vel_max,
            acc_max=self.acc_max,
            Q_pos=self.Q_pos,
            Q_vel=self.Q_vel,
            R=self.R,
            Q_terminal=self.Q_terminal,
            avoidance_enabled=self.avoidance_enabled,
            min_separation=self.min_separation,
            separation_penalty=self.separation_penalty,
        )

    def goal_motion(self) -> GoalMotion:
        """The goal motion pattern for this scenario."""
        return GoalMotion(
            motion_type=self.motion_type,
            workspace_min=np.array(self.pos_min),
            workspace_max=np.array(self.pos_max),
            circular_radius=self.circular_radius,
            circular_omega=self.circular_omega,
            translation_velocity=self.translation_velocity,
        )


def replace_fields(config: ScenarioConfig, **changes: float) -> ScenarioConfig:
    """Return a copy of ``config`` with the named fields replaced."""
    import dataclasses

    return dataclasses.replace(config, **changes)


@dataclass
class SimulationResult:
    """Everything one closed loop run produced.

    Attributes:
        config: The config that was run.
        states: State history, shape (T, N, 6).
        goals: Goal history at each step, shape (T, N, 3).
        times: Simulated time of each step, shape (T,).
        solve_times_ms: Solver wall clock time for every replan.
        failed_solves: Number of replans where the QP did not return a point.
        total_solves: Number of replans.
    """

    config: ScenarioConfig
    states: np.ndarray
    goals: np.ndarray
    times: np.ndarray
    solve_times_ms: list[float] = field(default_factory=list)
    failed_solves: int = 0
    total_solves: int = 0

    @property
    def success_rate(self) -> float:
        """Fraction of replans that produced a solution."""
        if self.total_solves == 0:
            return 0.0
        return (self.total_solves - self.failed_solves) / self.total_solves

    def positions(self) -> np.ndarray:
        """Position history as ``(N, 3, T)``, the layout the analysis uses."""
        return np.transpose(self.states[:, :, :SPATIAL_DIM], (1, 2, 0))

    def goal_positions(self) -> np.ndarray:
        """Goal history as ``(N, 3, T)``."""
        return np.transpose(self.goals, (1, 2, 0))


def _emergency_brake(states: np.ndarray, acc_max: float, dt: float, horizon: int) -> np.ndarray:
    """Decelerate along the current velocity, saturated at ``acc_max``."""
    num_agents = states.shape[0]
    brake = np.zeros((num_agents, CONTROL_DIM, horizon))
    for i in range(num_agents):
        velocity = states[i, 3:6]
        speed = np.linalg.norm(velocity)
        if speed > 0.01:
            direction = -velocity / speed
            magnitude = min(acc_max, speed / dt)
            for k in range(horizon):
                brake[i, :, k] = direction * magnitude
    return brake


def simulate(config: ScenarioConfig, seed: int | None = None) -> SimulationResult:
    """Run one closed loop simulation of a scenario.

    The agent states are advanced with the same double integrator the
    controller plans with, then position and velocity measurement noise is
    added and the speed is clamped at ``vel_max``.

    Args:
        config: The scenario to run.
        seed: Seed for the global numpy random state. Pass None to leave the
            random state alone, which is what the recorded experiments did
            after seeding once per run.

    Returns:
        The full history and solver statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    num_agents = config.num_agents
    dynamics = DoubleIntegrator(dt=config.dt)
    mpc = CentralMPC(config.controller_config())
    motion = config.goal_motion()

    states = np.array(config.initial_positions, dtype=float)
    states = np.hstack([states, np.zeros((num_agents, SPATIAL_DIM))])
    goals = np.array(config.goal_positions, dtype=float)

    for i in range(num_agents):
        states[i, :3] += np.random.normal(0, config.std_position, 3)
        states[i, 3:] += np.random.normal(0, config.std_velocity, 3)

    steps = int(config.sim_duration / config.dt)
    replan_interval = int(config.dt_plan / config.dt)

    state_history: list[np.ndarray] = []
    goal_history: list[np.ndarray] = []
    time_history: list[float] = []
    solve_times: list[float] = []
    failed_solves = 0
    total_solves = 0
    consecutive_failures = 0

    controls: np.ndarray | None = None
    control_idx = 0
    current_goals = goals.copy()

    logger.info(
        "simulating %s: %d agents, %.1f s, motion %s, avoidance %s",
        config.name,
        num_agents,
        config.sim_duration,
        config.motion_type,
        "on" if config.avoidance_enabled else "off",
    )

    for k in range(steps):
        current_time = k * config.dt

        if k % replan_interval == 0:
            goal_trajectories = np.zeros((num_agents, SPATIAL_DIM, config.horizon + 1))
            for i in range(num_agents):
                goal_trajectories[i] = motion.trajectory(
                    goals[i], current_time, config.horizon, config.dt
                )

            result = mpc.solve(states, goal_trajectories)
            total_solves += 1
            solve_times.append(result.solve_time_ms)

            if not result.success:
                failed_solves += 1
                consecutive_failures += 1
                if consecutive_failures >= FAILURES_BEFORE_BRAKE:
                    controls = _emergency_brake(states, config.acc_max, config.dt, config.horizon)
                else:
                    controls = np.zeros((num_agents, CONTROL_DIM, config.horizon))
            else:
                consecutive_failures = 0
                controls = result.controls

            control_idx = 0
            current_goals = goal_trajectories[:, :, 0]

        if controls is not None and control_idx < controls.shape[2]:
            u = controls[:, :, control_idx]
        else:
            u = np.zeros((num_agents, CONTROL_DIM))

        for i in range(num_agents):
            states[i] = dynamics.step(states[i], u[i])
            states[i, :3] += np.random.normal(0, config.std_position, 3)
            states[i, 3:] += np.random.normal(0, config.std_velocity, 3)
            speed = np.linalg.norm(states[i, 3:6])
            if speed > config.vel_max:
                states[i, 3:6] *= config.vel_max / speed

        state_history.append(states.copy())
        goal_history.append(current_goals.copy())
        time_history.append(current_time)
        control_idx += 1

    return SimulationResult(
        config=config,
        states=np.array(state_history),
        goals=np.array(goal_history),
        times=np.array(time_history),
        solve_times_ms=solve_times,
        failed_solves=failed_solves,
        total_solves=total_solves,
    )


def save_trajectories(result: SimulationResult, path: str | Path) -> None:
    """Write the position history in the text format the analysis reads.

    The layout is the one the distributed DMPC tooling uses: a header line,
    three lines of initial positions, three lines of goal positions, then
    three lines per agent holding the x, y and z history.

    Args:
        result: The run to write.
        path: Destination file.
    """
    config = result.config
    num_agents = config.num_agents
    positions = result.positions()
    goals = np.array(config.goal_positions)

    with Path(path).open("w") as f:
        f.write(f"{num_agents} {num_agents} ")
        f.write(f"{config.pos_min[0]} {config.pos_min[1]} {config.pos_min[2]} ")
        f.write(f"{config.pos_max[0]} {config.pos_max[1]} {config.pos_max[2]}\n")

        initial = result.states[0, :, :SPATIAL_DIM].T
        for d in range(SPATIAL_DIM):
            f.write(" ".join(map(str, initial[d, :])) + "\n")

        for d in range(SPATIAL_DIM):
            f.write(" ".join(map(str, goals[:, d])) + "\n")

        for i in range(num_agents):
            for d in range(SPATIAL_DIM):
                f.write(" ".join(map(str, positions[i, d, :])) + "\n")

    logger.info("wrote trajectories to %s", path)


def save_goals(result: SimulationResult, path: str | Path) -> None:
    """Write the goal history, three lines per agent.

    Args:
        result: The run to write.
        path: Destination file.
    """
    goals = result.goal_positions()
    with Path(path).open("w") as f:
        for i in range(result.config.num_agents):
            for d in range(SPATIAL_DIM):
                f.write(" ".join(map(str, goals[i, d, :])) + "\n")
    logger.info("wrote goals to %s", path)
