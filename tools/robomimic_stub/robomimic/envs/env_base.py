"""Minimal robomimic EnvBase wrapper for MimicGen compatibility."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class EnvBase:
    """Lightweight wrapper around a robosuite env providing robomimic-style API."""

    rollout_exceptions = (Exception,)

    def __init__(self, env: Any) -> None:
        self.base_env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    def reset(self) -> Dict[str, Any] | None:
        obs = self.base_env.reset()
        if obs is None and hasattr(self.base_env, "_get_observations"):
            return self.base_env._get_observations()
        return obs

    def get_observation(self) -> Dict[str, Any]:
        if hasattr(self.base_env, "_get_observations"):
            return self.base_env._get_observations()
        raise AttributeError("Underlying env lacks _get_observations")

    def get_state(self) -> Dict[str, Any]:
        state = {"states": self.base_env.sim.get_state().flatten()}
        model_xml = None
        if hasattr(self.base_env, "model"):
            model = self.base_env.model
            if hasattr(model, "to_xml"):
                try:
                    model_xml = model.to_xml()
                except Exception:
                    model_xml = None
        state["model"] = model_xml
        return state

    def reset_to(self, state: Dict[str, Any]) -> None:
        if "states" not in state:
            raise KeyError("reset_to expects key 'states'")
        states = np.asarray(state["states"])
        if hasattr(self.base_env.sim, "set_state_from_flattened"):
            self.base_env.sim.set_state_from_flattened(states)
        else:
            # fallback for mujoco versions without set_state_from_flattened
            self.base_env.sim.set_state(states)
        self.base_env.sim.forward()

    def is_success(self) -> Dict[str, bool]:
        if hasattr(self.base_env, "_check_success"):
            res = self.base_env._check_success()
            if isinstance(res, dict):
                if "task" not in res:
                    res = dict(res)
                    res["task"] = any(res.values())
                return res
            return {"task": bool(res)}
        return {"task": False}
