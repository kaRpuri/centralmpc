"""Double integrator model for a single agent in 3D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_DIM = 6
CONTROL_DIM = 3
SPATIAL_DIM = 3


@dataclass(frozen=True)
class DoubleIntegrator:
    """Discrete time triple-axis double integrator.

    The state is ``[x, y, z, vx, vy, vz]`` and the control is acceleration
    ``[ax, ay, az]``. Discretization is exact for an acceleration held
    constant over one step, which is what a zero order hold gives.
    """

    dt: float

    @property
    def A(self) -> np.ndarray:
        """State transition matrix, shape (6, 6)."""
        dt = self.dt
        return np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    @property
    def B(self) -> np.ndarray:
        """Control matrix, shape (6, 3)."""
        dt = self.dt
        return np.array(
            [
                [0.5 * dt**2, 0, 0],
                [0, 0.5 * dt**2, 0],
                [0, 0, 0.5 * dt**2],
                [dt, 0, 0],
                [0, dt, 0],
                [0, 0, dt],
            ]
        )

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Advance one state by one timestep under a constant acceleration.

        Args:
            state: Current state, shape (6,).
            control: Acceleration command, shape (3,).

        Returns:
            The next state, shape (6,).
        """
        return self.A @ state + self.B @ control

    def stacked(self, num_agents: int) -> tuple[np.ndarray, np.ndarray]:
        """Block diagonal dynamics for ``num_agents`` independent agents.

        Args:
            num_agents: Number of agents to stack.

        Returns:
            ``(A_big, B_big)`` with shapes (6N, 6N) and (6N, 3N).
        """
        return (
            np.kron(np.eye(num_agents), self.A),
            np.kron(np.eye(num_agents), self.B),
        )
