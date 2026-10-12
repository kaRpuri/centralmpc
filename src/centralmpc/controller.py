"""The centralized MPC controller: one QP over every agent at once."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from .constraints import (
    agent_pairs,
    box_constraints,
    dynamics_constraints,
    initial_state_constraint,
    separation_constraints,
    unit_normals,
)
from .dynamics import CONTROL_DIM, SPATIAL_DIM, STATE_DIM, DoubleIntegrator

logger = logging.getLogger(__name__)

SOLVER_OPTIONS = {
    "solver": cp.OSQP,
    "warm_start": True,
    "verbose": False,
    "eps_abs": 1e-3,
    "eps_rel": 1e-3,
    "max_iter": 10000,
    "adaptive_rho": True,
    "polish": True,
}


@dataclass(frozen=True)
class ControllerConfig:
    """Everything the QP needs, with no values hardcoded in the controller.

    Attributes:
        num_agents: Number of agents in the joint problem.
        horizon: Number of control intervals in the prediction horizon.
        dt: Horizon timestep, in seconds.
        pos_min: Lower workspace corner, shape (3,).
        pos_max: Upper workspace corner, shape (3,).
        vel_max: Velocity limit per axis, in m/s.
        acc_max: Acceleration limit per axis, in m/s^2.
        Q_pos: Position tracking weight.
        Q_vel: Velocity tracking weight.
        R: Control effort weight.
        Q_terminal: Multiplier applied to the tracking weights at the last step.
        avoidance_enabled: Add the pairwise separation constraint. Off by
            default, which reproduces the original formulation exactly.
        min_separation: Required centre to centre distance when avoidance is
            enabled, in metres.
    """

    num_agents: int
    horizon: int
    dt: float
    pos_min: np.ndarray
    pos_max: np.ndarray
    vel_max: float
    acc_max: float
    Q_pos: float
    Q_vel: float
    R: float
    Q_terminal: float = 10.0
    avoidance_enabled: bool = False
    min_separation: float = 0.2


@dataclass
class SolveResult:
    """Outcome of one QP solve.

    Attributes:
        controls: Acceleration plan, shape (N, 3, horizon). Zeros on failure.
        success: Whether the solver returned an optimal or inaccurate optimal
            point and a finite control plan.
        solve_time_ms: Wall clock time inside the solver, in milliseconds.
        status: The cvxpy problem status string.
    """

    controls: np.ndarray
    success: bool
    solve_time_ms: float = 0.0
    status: str = ""


@dataclass
class CentralMPC:
    """Joint MPC over all agents, solved as a single quadratic program.

    The problem is built once at construction. Each call to :meth:`solve`
    updates the measured state and the reference, then re-solves with a warm
    start from the previous solution.
    """

    config: ControllerConfig
    dynamics: DoubleIntegrator = field(init=False)
    problem: cp.Problem = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config
        self.dynamics = DoubleIntegrator(dt=cfg.dt)
        self._pairs = agent_pairs(cfg.num_agents) if cfg.avoidance_enabled else []
        self._normals: dict[tuple[int, int], cp.Parameter] = {
            pair: cp.Parameter((SPATIAL_DIM, cfg.horizon)) for pair in self._pairs
        }
        self._reference: np.ndarray | None = None
        self._build_qp()

    # Convenience passthroughs used by the simulator.
    @property
    def horizon(self) -> int:
        """Number of control intervals in the prediction horizon."""
        return self.config.horizon

    @property
    def vel_max(self) -> float:
        """Velocity limit per axis, in m/s."""
        return self.config.vel_max

    @property
    def acc_max(self) -> float:
        """Acceleration limit per axis, in m/s^2."""
        return self.config.acc_max

    def _build_qp(self) -> None:
        """Assemble the cost and constraints once, with the state as a parameter."""
        cfg = self.config
        n_agents, n_horizon = cfg.num_agents, cfg.horizon

        self.X = cp.Variable((STATE_DIM * n_agents, n_horizon + 1))
        self.U = cp.Variable((CONTROL_DIM * n_agents, n_horizon))
        self.X0 = cp.Parameter((STATE_DIM * n_agents,))
        self.X_ref = cp.Parameter((STATE_DIM * n_agents, n_horizon + 1))

        A_big, B_big = self.dynamics.stacked(n_agents)

        cost = 0
        for k in range(n_horizon):
            for i in range(n_agents):
                idx = i * STATE_DIM
                x_err = self.X[idx : idx + 3, k] - self.X_ref[idx : idx + 3, k]
                v_err = self.X[idx + 3 : idx + 6, k] - self.X_ref[idx + 3 : idx + 6, k]
                cost += cfg.Q_pos * cp.sum_squares(x_err)
                cost += cfg.Q_vel * cp.sum_squares(v_err)

        for i in range(n_agents):
            idx = i * STATE_DIM
            x_err_terminal = self.X[idx : idx + 3, n_horizon] - self.X_ref[idx : idx + 3, n_horizon]
            v_err_terminal = (
                self.X[idx + 3 : idx + 6, n_horizon] - self.X_ref[idx + 3 : idx + 6, n_horizon]
            )
            cost += cfg.Q_terminal * cfg.Q_pos * cp.sum_squares(x_err_terminal)
            cost += cfg.Q_terminal * cfg.Q_vel * cp.sum_squares(v_err_terminal)

        for k in range(n_horizon):
            cost += cfg.R * cp.sum_squares(self.U[:, k])

        constraints = initial_state_constraint(self.X, self.X0)
        constraints += dynamics_constraints(self.X, self.U, A_big, B_big, n_horizon)
        constraints += box_constraints(
            self.X,
            self.U,
            n_agents,
            n_horizon,
            cfg.pos_min,
            cfg.pos_max,
            cfg.vel_max,
            cfg.acc_max,
        )
        if cfg.avoidance_enabled:
            constraints += separation_constraints(
                self.X, self._normals, n_horizon, cfg.min_separation
            )

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def _state_reference(self, goal_positions: np.ndarray) -> np.ndarray:
        """Stack goal positions into a full state reference with zero velocity."""
        cfg = self.config
        x_ref = np.zeros((STATE_DIM * cfg.num_agents, cfg.horizon + 1))
        for i in range(cfg.num_agents):
            for k in range(cfg.horizon + 1):
                x_ref[i * 6 : i * 6 + 3, k] = goal_positions[i, :, k]
                x_ref[i * 6 + 3 : i * 6 + 6, k] = 0.0
        return x_ref

    def _update_normals(self, current_states: np.ndarray) -> None:
        """Set the separation normals from the previous plan, or from the state.

        On the first solve there is no previous plan, so the current position
        held constant over the horizon is used as the reference.
        """
        cfg = self.config
        if self._reference is None:
            reference = np.repeat(current_states[:, :SPATIAL_DIM, None], cfg.horizon + 1, axis=2)
        else:
            reference = self._reference
        for pair, value in unit_normals(reference, self._pairs, cfg.horizon).items():
            self._normals[pair].value = value

    def _failure(self, status: str = "", solve_time_ms: float = 0.0) -> SolveResult:
        """A zero control plan marked unsuccessful."""
        cfg = self.config
        return SolveResult(
            controls=np.zeros((cfg.num_agents, CONTROL_DIM, cfg.horizon)),
            success=False,
            solve_time_ms=solve_time_ms,
            status=status,
        )

    def solve(self, current_states: np.ndarray, goal_positions: np.ndarray) -> SolveResult:
        """Solve the joint QP for the current state and reference.

        Args:
            current_states: Measured states, shape (N, 6).
            goal_positions: Goal positions over the horizon, shape
                (N, 3, horizon + 1).

        Returns:
            A :class:`SolveResult`. On any failure the controls are zeros and
            ``success`` is False, which the simulator treats as a missed solve.
        """
        cfg = self.config
        x0 = current_states.flatten()
        x_ref = self._state_reference(goal_positions)

        if not np.all(np.isfinite(x0)):
            logger.warning("skipping solve: measured state is not finite")
            return self._failure("non_finite_state")
        if not np.all(np.isfinite(x_ref)):
            logger.warning("skipping solve: reference is not finite")
            return self._failure("non_finite_reference")

        self.X0.value = x0
        self.X_ref.value = x_ref
        if cfg.avoidance_enabled:
            self._update_normals(current_states)

        try:
            start = time.time()
            self.problem.solve(**SOLVER_OPTIONS)
            solve_time_ms = (time.time() - start) * 1000
        except cp.error.SolverError as exc:
            logger.warning("solver error: %s", exc)
            return self._failure("solver_error")

        status = self.problem.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            logger.warning("QP status %s", status)
            return self._failure(status, solve_time_ms)

        controls = self.U.value.reshape((cfg.num_agents, CONTROL_DIM, cfg.horizon))
        if not np.all(np.isfinite(controls)):
            logger.warning("QP returned a non-finite control plan")
            return self._failure("non_finite_controls", solve_time_ms)

        self._reference = self.X.value.reshape((cfg.num_agents, STATE_DIM, cfg.horizon + 1))[
            :, :SPATIAL_DIM, :
        ]

        logger.debug("solved in %.1f ms (%s)", solve_time_ms, status)
        return SolveResult(
            controls=controls, success=True, solve_time_ms=solve_time_ms, status=status
        )
