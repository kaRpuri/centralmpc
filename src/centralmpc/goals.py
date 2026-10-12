"""Goal motion patterns for the reference trajectory handed to the controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOTION_TYPES = ("static", "circular", "translating", "circular_translating")


@dataclass(frozen=True)
class GoalMotion:
    """Time varying goal positions for one agent.

    Each pattern is evaluated in closed form at absolute time ``t``, so the
    reference over a planning horizon is reproducible and does not depend on
    the controller's own history.

    Attributes:
        motion_type: One of ``MOTION_TYPES``.
        workspace_min: Lower workspace corner, shape (3,).
        workspace_max: Upper workspace corner, shape (3,).
        circular_radius: Orbit radius for the circular patterns, in metres.
        circular_omega: Orbit rate, in rad/s.
        translation_velocity: Scale on the translating patterns, in m/s.
    """

    motion_type: str
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    circular_radius: float = 2.0
    circular_omega: float = 0.5
    translation_velocity: float = 0.5

    def __post_init__(self) -> None:
        if self.motion_type not in MOTION_TYPES:
            raise ValueError(f"unknown motion_type {self.motion_type!r}, expected {MOTION_TYPES}")

    def trajectory(
        self, initial_goal: np.ndarray, current_time: float, horizon: int, dt: float
    ) -> np.ndarray:
        """Evaluate the goal position over a planning horizon.

        Args:
            initial_goal: The agent's nominal goal, shape (3,).
            current_time: Absolute time at the start of the horizon, in seconds.
            horizon: Number of control intervals in the horizon.
            dt: Horizon timestep, in seconds.

        Returns:
            Goal positions, shape (3, horizon + 1).
        """
        goal_traj = np.zeros((3, horizon + 1))

        if self.motion_type == "static":
            for k in range(horizon + 1):
                goal_traj[:, k] = initial_goal

        elif self.motion_type == "circular":
            original_angle = np.arctan2(initial_goal[1], initial_goal[0])
            for k in range(horizon + 1):
                t = current_time + k * dt
                angle = original_angle + t * self.circular_omega
                goal_traj[0, k] = self.circular_radius * np.cos(angle)
                goal_traj[1, k] = self.circular_radius * np.sin(angle)
                goal_traj[2, k] = initial_goal[2]

        elif self.motion_type == "translating":
            for k in range(horizon + 1):
                t = current_time + k * dt
                x_range = self.workspace_max[0] - self.workspace_min[0] - 1.0
                x_offset = (x_range / 2) * np.sin(t * self.translation_velocity * 0.3)

                goal_traj[0, k] = initial_goal[0] + x_offset
                goal_traj[1, k] = initial_goal[1]
                goal_traj[2, k] = initial_goal[2]

                goal_traj[0, k] = np.clip(
                    goal_traj[0, k], self.workspace_min[0] + 0.5, self.workspace_max[0] - 0.5
                )

        elif self.motion_type == "circular_translating":
            original_angle = np.arctan2(initial_goal[1], initial_goal[0])
            for k in range(horizon + 1):
                t = current_time + k * dt

                x_range = self.workspace_max[0] - self.workspace_min[0] - 2.0
                y_range = self.workspace_max[1] - self.workspace_min[1] - 2.0
                center_x = (x_range / 2) * np.sin(t * self.translation_velocity * 0.2)
                center_y = (y_range / 2) * np.sin(2 * t * self.translation_velocity * 0.2)

                angle = original_angle + t * self.circular_omega
                goal_traj[0, k] = center_x + self.circular_radius * 0.5 * np.cos(angle)
                goal_traj[1, k] = center_y + self.circular_radius * 0.5 * np.sin(angle)
                goal_traj[2, k] = initial_goal[2]

                goal_traj[0, k] = np.clip(
                    goal_traj[0, k], self.workspace_min[0] + 0.5, self.workspace_max[0] - 0.5
                )
                goal_traj[1, k] = np.clip(
                    goal_traj[1, k], self.workspace_min[1] + 0.5, self.workspace_max[1] - 0.5
                )

        return goal_traj
