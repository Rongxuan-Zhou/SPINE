"""Helper dynamics and contact models used by the Python CITO prototype."""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple


def simple_point_mass_dynamics(dt: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Returns a discrete-time point-mass dynamics function f(x, u) -> x_next.

    State: [x, v]; Control: [force]; mass=1 for simplicity.
    """

    def step(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        x, v = state
        force = control[0]
        v_next = v + dt * force
        x_next = x + dt * v_next
        return np.array([x_next, v_next], dtype=float)

    return step


def planar_contact(boundary: float = 0.0) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
    """Contact distance/normal for a 1D plane at x=boundary with outward normal +x."""

    def contact_fn(state: np.ndarray) -> Tuple[float, np.ndarray]:
        distance = state[0] - boundary
        normal = np.array([1.0, 0.0], dtype=float)
        return float(distance), normal

    return contact_fn


__all__ = ["simple_point_mass_dynamics", "planar_contact"]
