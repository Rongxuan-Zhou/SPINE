"""Configuration for the pure-Python CITO solver."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CITOParameters:
    """Hyperparameters controlling the CITO loop and VSCM penalties."""

    dt: float = 0.05
    max_iters: int = 15
    trust_region: float = 0.2
    track_weight: float = 5.0
    control_weight: float = 1e-3
    dynamics_weight: float = 10.0
    contact_weight: float = 25.0
    smoothing_length: float = 0.02  # meters, for VSCM softplus smoothing
    step_size: float = 0.05  # gradient descent step within each convexified subproblem
