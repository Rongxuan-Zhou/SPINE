"""Minimal env utils stub used by MimicGen generation."""

from __future__ import annotations

from typing import Any, Dict, List


def set_env_specific_obs_processing(env_meta=None):  # noqa: D401
    """No-op placeholder for robomimic.EnvUtils compatibility."""
    return None


def is_robosuite_env(env_meta: Dict[str, Any]) -> bool:
    return env_meta.get("type", "robosuite") == "robosuite"


def get_env_type(env_meta: Dict[str, Any]) -> str:
    return env_meta.get("type", "robosuite")


def create_env_for_data_processing(
    env_meta: Dict[str, Any],
    env_class=None,
    camera_names: List[str] | None = None,
    camera_height: int = 84,
    camera_width: int = 84,
    reward_shaping: bool = False,
    render: bool | None = None,
    render_offscreen: bool | None = None,
    use_image_obs: bool | None = None,
    use_depth_obs: bool | None = None,
):
    """Create a robosuite env using metadata stored in a MimicGen dataset."""
    from robomimic.envs.env_base import EnvBase
    import robosuite as suite

    env_kwargs = dict(env_meta.get("env_kwargs", {}))
    env_name = env_meta.get("env_name")
    if env_name is None and env_class is None:
        raise ValueError("env_meta missing env_name and env_class not provided.")

    camera_names = camera_names or []
    env_kwargs.setdefault("has_renderer", bool(render))
    env_kwargs.setdefault("has_offscreen_renderer", bool(render_offscreen))
    env_kwargs.setdefault("use_camera_obs", bool(use_image_obs))
    env_kwargs.setdefault("camera_names", camera_names)
    # robosuite expects plural camera_heights / camera_widths
    env_kwargs.setdefault("camera_heights", camera_height)
    env_kwargs.setdefault("camera_widths", camera_width)
    env_kwargs.setdefault("reward_shaping", reward_shaping)

    if env_class is not None:
        env = env_class(**env_kwargs)
    else:
        env = suite.make(env_name=env_name, **env_kwargs)

    if not hasattr(env, "serialize"):
        def _serialize():  # noqa: D401
            meta = dict(env_meta or {})
            meta.setdefault("type", env_meta.get("type", "robosuite") if env_meta else "robosuite")
            meta.setdefault("env_name", env_name)
            meta.setdefault("env_kwargs", env_kwargs)
            return meta
        env.serialize = _serialize  # type: ignore[attr-defined]

    return EnvBase(env)
