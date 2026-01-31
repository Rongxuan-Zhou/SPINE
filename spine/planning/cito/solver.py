"""Simplified successive convexification with a VSCM-style contact cost."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from .configs import CITOParameters

logger = logging.getLogger(__name__)


Array = np.ndarray


@dataclass
class Trajectory:
    """Container for state and control trajectories."""

    states: Array  # shape (T+1, state_dim)
    controls: Array  # shape (T, control_dim)


@dataclass
class CITOPlanResult:
    """Output bundle with diagnostics."""

    trajectory: Trajectory
    costs: List[float]
    penetrations: List[float]


ContactFn = Callable[[Array], Tuple[float, Array]]
DynamicsFn = Callable[[Array, Array], Array]


def _softplus(x: Array) -> Array:
    # numerically stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


class CITOPlanner:
    """Lightweight CITO optimizer with VSCM contact penalty."""

    def __init__(self, params: CITOParameters) -> None:
        self.params = params

    def optimize(
        self,
        traj: Trajectory,
        dynamics_fn: DynamicsFn,
        contact_fn: ContactFn,
        target_states: Array,
    ) -> CITOPlanResult:
        """Runs a few SCVX-like iterations to reduce penetration and track a target."""
        costs: List[float] = []
        max_pen_list: List[float] = []
        states = traj.states.copy()
        controls = traj.controls.copy()

        for it in range(self.params.max_iters):
            cost, max_pen = self._compute_cost(states, controls, target_states, dynamics_fn, contact_fn)
            costs.append(cost)
            max_pen_list.append(max_pen)
            logger.debug("iter=%d cost=%.4f max_pen=%.4f", it, cost, max_pen)
            grad_states, grad_controls = self._compute_gradients(states, controls, target_states, dynamics_fn, contact_fn)
            # trust region on update magnitude
            state_step = np.clip(-self.params.step_size * grad_states, -self.params.trust_region, self.params.trust_region)
            control_step = np.clip(-self.params.step_size * grad_controls, -self.params.trust_region, self.params.trust_region)
            states = states + state_step
            controls = controls + control_step

        return CITOPlanResult(trajectory=Trajectory(states=states, controls=controls), costs=costs, penetrations=max_pen_list)

    def _compute_cost(
        self,
        states: Array,
        controls: Array,
        target_states: Array,
        dynamics_fn: DynamicsFn,
        contact_fn: ContactFn,
    ) -> Tuple[float, float]:
        track_err = states - target_states
        cost = self.params.track_weight * float(np.sum(track_err**2))
        cost += self.params.control_weight * float(np.sum(controls**2))
        dyn_res = self._dynamics_residual(states, controls, dynamics_fn)
        cost += self.params.dynamics_weight * float(np.sum(dyn_res**2))

        max_pen = 0.0
        contact_cost = 0.0
        for s in states:
            dist, _ = contact_fn(s)
            max_pen = min(max_pen, dist)
            scaled = -dist / self.params.smoothing_length
            contact_cost += float(np.sum(_softplus(scaled)))
        cost += self.params.contact_weight * contact_cost
        return cost, abs(max_pen)

    def _compute_gradients(
        self,
        states: Array,
        controls: Array,
        target_states: Array,
        dynamics_fn: DynamicsFn,
        contact_fn: ContactFn,
    ) -> Tuple[Array, Array]:
        grad_states = np.zeros_like(states)
        grad_controls = np.zeros_like(controls)
        # tracking gradient
        grad_states += 2 * self.params.track_weight * (states - target_states)
        grad_controls += 2 * self.params.control_weight * controls
        # dynamics residual gradient via finite differences
        dyn_res = self._dynamics_residual(states, controls, dynamics_fn)
        eps = 1e-3
        for t in range(controls.shape[0]):
            # grad wrt controls
            for j in range(controls.shape[1]):
                controls_eps = controls.copy()
                controls_eps[t, j] += eps
                res_eps = self._dynamics_residual(states, controls_eps, dynamics_fn)
                grad_controls[t, j] += (
                    2
                    * self.params.dynamics_weight
                    * np.sum((res_eps - dyn_res) * dyn_res)
                    / eps
                )
            # grad wrt states (current and next)
            for j in range(states.shape[1]):
                states_eps = states.copy()
                states_eps[t, j] += eps
                res_eps = self._dynamics_residual(states_eps, controls, dynamics_fn)
                grad_states[t, j] += (
                    2
                    * self.params.dynamics_weight
                    * np.sum((res_eps - dyn_res) * dyn_res)
                    / eps
                )
                states_eps_next = states.copy()
                states_eps_next[t + 1, j] += eps
                res_eps_next = self._dynamics_residual(states_eps_next, controls, dynamics_fn)
                grad_states[t + 1, j] += (
                    2
                    * self.params.dynamics_weight
                    * np.sum((res_eps_next - dyn_res) * dyn_res)
                    / eps
                )
        # contact gradients
        for i, s in enumerate(states):
            dist, normal = contact_fn(s)
            scaled = -dist / self.params.smoothing_length
            coeff = self.params.contact_weight * _sigmoid(scaled) / self.params.smoothing_length
            grad_states[i, : len(normal)] += -coeff * normal
        return grad_states, grad_controls

    @staticmethod
    def _dynamics_residual(states: Array, controls: Array, dynamics_fn: DynamicsFn) -> Array:
        res = []
        for t in range(controls.shape[0]):
            pred_next = dynamics_fn(states[t], controls[t])
            res.append(states[t + 1] - pred_next)
        return np.stack(res, axis=0)
