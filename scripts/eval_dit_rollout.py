#!/usr/bin/env python
"""Rollout evaluation for trained SpineDiT models in robosuite envs.

This script runs diffusion sampling at each step, executes the first action
in the sampled horizon, and reports success rate.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTRA_PY_PATHS = [
    REPO_ROOT,
    REPO_ROOT / "external" / "mimicgen",
    REPO_ROOT / "tools" / "robomimic_stub",
]
for _p in _EXTRA_PY_PATHS:
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

try:
    import robosuite as suite
    from robosuite.controllers import load_controller_config
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("robosuite is required for rollout evaluation.") from exc

from models.interpretable_dit import SpineDiT
from models.scheduler import DDPMScheduler


class _InitTimeout(RuntimeError):
    pass


@contextlib.contextmanager
def _time_limit(seconds: float, msg: str):
    """Guard potentially hanging env init/reset in headless rendering setups."""
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # pragma: no cover - signal callback
        raise _InitTimeout(msg)

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _get_eef_pose(env) -> Tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    site_id = robot.eef_site_id
    pos = env.sim.data.site_xpos[site_id].copy()
    if hasattr(env.sim.data, "site_xquat"):
        quat = env.sim.data.site_xquat[site_id].copy()
    else:
        mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
        # robosuite uses wxyz
        quat = np.array([mat[0, 0], mat[1, 1], mat[2, 2], 0.0])
    return pos, quat


def _teleport_object_to_gripper(env, body_id: int, z_min: float) -> None:
    eef_pos, eef_quat = _get_eef_pose(env)
    target_pos = eef_pos.copy()
    target_pos[2] = max(target_pos[2], z_min)
    joint_name = env.sim.model.joint_names[env.sim.model.body_jntadr[body_id]]
    qpos = env.sim.data.get_joint_qpos(joint_name)
    qpos[:3] = target_pos
    qpos[3:7] = eef_quat
    env.sim.data.set_joint_qpos(joint_name, qpos)
    env.sim.forward()


def _get_body_force(env, body_name: str = "robot0_link7") -> float:
    body_id = env.sim.model.body_name2id(body_name)
    wrench = env.sim.data.cfrc_ext[body_id]
    return float(np.linalg.norm(wrench[:3]))


def _extract_obs_joint_force(
    obs: dict,
    joint_dim: int,
    fallback_force: float,
    use_rgb: bool = False,
    rgb_key: str = "agentview_image",
    rgb_size: int = 84,
) -> Tuple[np.ndarray, float, Optional[np.ndarray], torch.Tensor | None]:
    q_obs = obs.get("robot0_joint_pos", None)
    if q_obs is None:
        q_curr = np.zeros(joint_dim, dtype=np.float32)
    else:
        q_obs = np.array(q_obs, dtype=np.float32).reshape(-1)
        q_curr = np.zeros(joint_dim, dtype=np.float32)
        q_curr[: min(joint_dim, len(q_obs))] = q_obs[:joint_dim]

    f_obs = obs.get("robot0_ee_force", None)
    if f_obs is None:
        force = float(fallback_force)
        force_vec = None
    else:
        force_vec = np.array(f_obs, dtype=np.float32).reshape(-1)
        force = float(np.linalg.norm(force_vec))
    rgb_tensor = None
    if use_rgb:
        rgb = None
        for key in (rgb_key, "agentview_image", "robot0_eye_in_hand_image"):
            if key in obs:
                rgb = np.array(obs[key], dtype=np.float32)
                break
        if rgb is None:
            raise KeyError(
                f"RGB key '{rgb_key}' (or fallbacks) not found in rollout observation."
            )
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB HWC image, got shape={rgb.shape}")
        if np.max(rgb) > 1.0:
            rgb = rgb / 255.0
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        if rgb_tensor.shape[-1] != rgb_size or rgb_tensor.shape[-2] != rgb_size:
            rgb_tensor = F.interpolate(
                rgb_tensor.unsqueeze(0),
                size=(rgb_size, rgb_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
    return q_curr, force, force_vec, rgb_tensor


def _build_physics_token(
    force_scalar: float,
    force_vec: Optional[np.ndarray],
    token_dim: int,
    contact_force_threshold: float,
    force_mag_clip: float,
) -> np.ndarray:
    fnorm = float(abs(force_scalar))
    contact = 1.0 if fnorm > float(contact_force_threshold) else 0.0
    normal_z = 0.5
    if force_vec is not None and force_vec.shape[0] >= 3:
        denom = float(np.linalg.norm(force_vec[:3]))
        if denom > 1e-6:
            normal_z = float(np.clip(0.5 * (force_vec[2] / denom + 1.0), 0.0, 1.0))
    mag = float(np.log1p(max(fnorm, 0.0)) / max(np.log1p(float(force_mag_clip)), 1e-6))
    mag = float(np.clip(mag, 0.0, 1.0))
    tok = np.array([contact, normal_z, mag], dtype=np.float32)
    if token_dim > 3:
        tok = np.concatenate([tok, np.zeros(token_dim - 3, dtype=np.float32)], axis=0)
    elif token_dim < 3:
        tok = tok[:token_dim]
    return tok


def _get_target_body(env, task: str) -> str:
    if task == "threading" and hasattr(env, "needle"):
        return env.needle.root_body
    if hasattr(env, "nuts") and env.nuts:
        return env.nuts[0].root_body
    return "SquareNut"


def execute_oracle_grasp(env, task: str, mode: str = "waypoints") -> None:
    """Deterministic grasping so insertion starts from a fixed state."""
    target_body = _get_target_body(env, task)
    body_id = env.sim.model.body_name2id(target_body)
    obj_pos = env.sim.data.body_xpos[body_id].copy()

    def step_to(target: np.ndarray, steps: int = 50, gain: float = 3.0):
        for _ in range(steps):
            eef_pos, _ = _get_eef_pose(env)
            delta = target - eef_pos
            action = np.zeros(env.action_dim, dtype=np.float32)
            action[:3] = np.clip(gain * delta, -0.05, 0.05)
            if env.action_dim > 3:
                action[-1] = 1.0
            env.step(action)

    pregrasp = obj_pos + np.array([0.0, 0.0, 0.08])
    grasp = obj_pos + np.array([0.0, 0.0, 0.01])

    step_to(pregrasp, steps=80)
    step_to(grasp, steps=60, gain=2.0)

    for _ in range(30):
        action = np.zeros(env.action_dim, dtype=np.float32)
        if env.action_dim > 0:
            action[-1] = -1.0
        env.step(action)

    z_min = grasp[2] + 0.05
    if mode == "teleport":
        _teleport_object_to_gripper(env, body_id, z_min)
    else:
        _teleport_object_to_gripper(env, body_id, z_min)

    safe = obj_pos + np.array([0.0, 0.0, 0.12])
    step_to(safe, steps=60, gain=2.0)


def _build_env(
    task: str,
    controller: str,
    debug: bool,
    controller_raw_delta: bool,
    controller_delta_limit: float,
    suppress_optional_import_warnings: bool,
    use_rgb: bool,
    camera_name: str,
    camera_height: int,
    camera_width: int,
) -> suite.Env:
    if suppress_optional_import_warnings:
        os.environ.setdefault("MIMICGEN_SUPPRESS_OPTIONAL_IMPORT_WARNINGS", "1")
    with contextlib.redirect_stdout(io.StringIO()):
        from spine.simulation.robosuite_spine_envs import (
            SpineNutAssemblySquare,
            SpineThreading,
        )

    render = bool(debug)
    controller_config = load_controller_config(default_controller=controller)
    # JOINT_POSITION expects normalized delta actions. This option makes arm action
    # ranges identity in radians so q_target - q_curr can be passed directly.
    if controller_raw_delta and controller == "JOINT_POSITION":
        lim = float(controller_delta_limit)
        controller_config["input_max"] = lim
        controller_config["input_min"] = -lim
        controller_config["output_max"] = lim
        controller_config["output_min"] = -lim

    env_kwargs = dict(
        robots="Panda",
        has_renderer=render,
        has_offscreen_renderer=bool(use_rgb),
        use_camera_obs=bool(use_rgb),
        control_freq=50,
        controller_configs=controller_config,
    )
    if use_rgb:
        env_kwargs["camera_names"] = camera_name
        env_kwargs["camera_heights"] = int(camera_height)
        env_kwargs["camera_widths"] = int(camera_width)
    if task == "square":
        return SpineNutAssemblySquare(**env_kwargs)
    if task == "threading":
        return SpineThreading(**env_kwargs)
    raise ValueError(f"Unknown task: {task}")


def _close_env_safely(env) -> None:
    try:
        env.close()
    except Exception:
        pass


def _safe_reset_env(env, timeout_s: float):
    with _time_limit(timeout_s, f"env.reset timeout after {timeout_s:.1f}s"):
        return env.reset()


def _init_env_with_rgb_fallback(args, use_rgb: bool):
    """Initialize env robustly for RGB rollout with backend/timeout fallback."""
    backends = []
    env_backend = os.environ.get("MUJOCO_GL", "").strip()
    if env_backend:
        backends.append(env_backend)
    for b in [x.strip() for x in args.rgb_backends.split(",") if x.strip()]:
        if b not in backends:
            backends.append(b)

    if use_rgb:
        for backend in backends:
            env = None
            try:
                os.environ["MUJOCO_GL"] = backend
                env = _build_env(
                    args.task,
                    args.controller,
                    debug=args.debug,
                    controller_raw_delta=args.controller_raw_delta,
                    controller_delta_limit=args.controller_delta_limit,
                    suppress_optional_import_warnings=args.suppress_optional_import_warnings,
                    use_rgb=True,
                    camera_name=args.camera_name,
                    camera_height=args.camera_height,
                    camera_width=args.camera_width,
                )
                obs0 = _safe_reset_env(env, args.rgb_init_timeout)
                # Validate RGB exists in observation to avoid silent fallback later.
                _, _, _, _ = _extract_obs_joint_force(
                    obs0,
                    joint_dim=args.joint_dim,
                    fallback_force=0.0,
                    use_rgb=True,
                    rgb_key=args.rgb_key,
                    rgb_size=args.camera_height,
                )
                print(f"[info] RGB env initialized with MUJOCO_GL={backend}", flush=True)
                return env, True, obs0, backend
            except Exception as exc:
                print(
                    f"[warn] RGB init failed for MUJOCO_GL={backend}: {exc}",
                    flush=True,
                )
                if env is not None:
                    _close_env_safely(env)

        print(
            "[warn] RGB rendering unavailable; fallback to env without camera and zero RGB input.",
            flush=True,
        )

    env = _build_env(
        args.task,
        args.controller,
        debug=args.debug,
        controller_raw_delta=args.controller_raw_delta,
        controller_delta_limit=args.controller_delta_limit,
        suppress_optional_import_warnings=args.suppress_optional_import_warnings,
        use_rgb=False,
        camera_name=args.camera_name,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )
    obs0 = _safe_reset_env(env, args.rgb_init_timeout)
    return env, False, obs0, None


def _auto_action_mode(
    controller: str, action_low: np.ndarray, action_high: np.ndarray
) -> str:
    if controller == "JOINT_POSITION":
        return "delta"
    span = float(np.mean(action_high - action_low))
    return "delta" if span < 1.0 else "absolute"


def _format_action(
    q_target: np.ndarray,
    q_curr: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    mode: str,
    gripper: float,
) -> np.ndarray:
    # Map model output to arm joints (+ optional gripper action).
    action_dim = int(action_low.shape[0])
    arm_action_dim = action_dim - 1 if action_dim > 7 else action_dim
    arm_q = q_target[:arm_action_dim]
    arm_curr = q_curr[:arm_action_dim]
    if mode == "delta":
        arm_cmd = arm_q - arm_curr
    else:
        arm_cmd = arm_q
    action = np.zeros_like(action_low, dtype=np.float32)
    action[:arm_action_dim] = arm_cmd
    if action_dim > arm_action_dim:
        action[-1] = gripper
    return np.clip(action, action_low, action_high)


def _make_ddim_schedule(total_steps: int, sample_steps: int) -> list[int]:
    if sample_steps >= total_steps:
        return list(range(total_steps - 1, -1, -1))
    raw = np.linspace(total_steps - 1, 0, num=sample_steps)
    sched: list[int] = []
    for value in raw:
        t = int(round(float(value)))
        if not sched or t != sched[-1]:
            sched.append(t)
    if sched[-1] != 0:
        sched.append(0)
    return sched


def _sample_actions(
    model: SpineDiT,
    scheduler: DDPMScheduler,
    obs_joint: torch.Tensor,
    obs_force: torch.Tensor,
    obs_rgb: torch.Tensor | None,
    obs_phys_token: torch.Tensor | None,
    obs_phys_mask: torch.Tensor | None,
    horizon: int,
    action_dim: int,
    device: str,
    sample_steps: int,
) -> torch.Tensor:
    x = torch.randn(1, horizon, action_dim, device=device)
    schedule = _make_ddim_schedule(scheduler.n_steps, sample_steps=sample_steps)
    for idx, t in enumerate(schedule):
        timestep = torch.full((1,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            pred_noise = model(
                x,
                timestep,
                obs_joint,
                obs_force,
                obs_rgb=obs_rgb,
                obs_phys_token=obs_phys_token,
                obs_phys_mask=obs_phys_mask,
            )

        alpha_bar_t = scheduler.alphas_cumprod[t]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        x0 = (x - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t

        if idx == len(schedule) - 1:
            x = x0
        else:
            t_prev = schedule[idx + 1]
            alpha_bar_prev = scheduler.alphas_cumprod[t_prev]
            x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(
                1.0 - alpha_bar_prev
            ) * pred_noise
    return x.squeeze(0)


def _load_replay_traj(
    replay_hdf5: Path,
    demo_key: str,
    joint_dim: int,
) -> np.ndarray:
    with h5py.File(replay_hdf5, "r") as f:
        if demo_key not in f["data"]:
            raise KeyError(f"{demo_key} not found in {replay_hdf5}")
        demo = f["data"][demo_key]
        if "obs" not in demo:
            raise KeyError(f"{demo_key} has no obs group in {replay_hdf5}")
        obs = demo["obs"]
        if "joint_positions" in obs:
            q = np.array(obs["joint_positions"], dtype=np.float32)
        elif "robot0_joint_pos" in obs:
            q = np.array(obs["robot0_joint_pos"], dtype=np.float32)
        else:
            raise KeyError(
                f"{demo_key} missing joint_positions / robot0_joint_pos in {replay_hdf5}"
            )
    if q.shape[1] < joint_dim:
        q_pad = np.zeros((q.shape[0], joint_dim), dtype=np.float32)
        q_pad[:, : q.shape[1]] = q
        q = q_pad
    elif q.shape[1] > joint_dim:
        q = q[:, :joint_dim]
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout eval for SpineDiT.")
    parser.add_argument("--task", choices=["square", "threading"], required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--policy-source", choices=["model", "replay"], default="model")
    parser.add_argument("--replay-hdf5", type=Path, default=None)
    parser.add_argument("--replay-demo", type=str, default="demo_0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--model-horizon", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--resample-every", type=int, default=4)
    parser.add_argument("--joint-dim", type=int, default=9)
    parser.add_argument("--action-dim", type=int, default=9)
    parser.add_argument("--force-dim", type=int, default=1)
    parser.add_argument(
        "--use-rgb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable RGB conditioning during rollout (default: auto-from-checkpoint).",
    )
    parser.add_argument(
        "--use-physics-inpainting",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable physics inpainting branch (default: auto-from-checkpoint).",
    )
    parser.add_argument("--physics-token-dim", type=int, default=3)
    parser.add_argument(
        "--physics-inpaint-mode",
        choices=["teacher", "fullmask"],
        default="fullmask",
    )
    parser.add_argument("--contact-force-threshold", type=float, default=2.0)
    parser.add_argument("--force-mag-clip", type=float, default=50.0)
    parser.add_argument("--rgb-key", type=str, default="agentview_image")
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--camera-width", type=int, default=84)
    parser.add_argument(
        "--rgb-init-timeout",
        type=float,
        default=20.0,
        help="Timeout seconds for RGB env init/reset; <=0 disables timeout.",
    )
    parser.add_argument(
        "--rgb-backends",
        type=str,
        default="egl,osmesa",
        help="Comma-separated MUJOCO_GL backends to try for RGB.",
    )
    parser.add_argument("--controller", type=str, default="JOINT_POSITION")
    parser.add_argument(
        "--controller-raw-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--controller-delta-limit", type=float, default=0.1)
    parser.add_argument(
        "--suppress-optional-import-warnings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--action-mode", choices=["auto", "absolute", "delta"], default="auto"
    )
    parser.add_argument(
        "--execution-mode", choices=["controller", "teleport"], default="controller"
    )
    parser.add_argument("--oracle-grasp", action="store_true")
    parser.add_argument(
        "--oracle-mode", choices=["waypoints", "teleport"], default="waypoints"
    )
    parser.add_argument("--gripper", type=float, default=-1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=Path, default=Path("artifacts/rollout_eval"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.policy_source == "replay" and args.replay_hdf5 is None:
        raise ValueError("--replay-hdf5 is required when --policy-source replay")
    if args.sample_steps < 1:
        raise ValueError("--sample-steps must be >= 1")
    if args.resample_every < 1:
        raise ValueError("--resample-every must be >= 1")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable, fallback to cpu", flush=True)
        device = "cpu"
    else:
        device = args.device

    raw_ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = (
        raw_ckpt["model_state_dict"]
        if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt
        else raw_ckpt
    )
    ckpt_has_rgb = any(k.startswith("rgb_enc.") for k in state_dict.keys())
    ckpt_has_phys = any(
        k.startswith("physics_enc.")
        or k.startswith("physics_head.")
        or k == "physics_mask_token"
        for k in state_dict.keys()
    )
    use_rgb = ckpt_has_rgb if args.use_rgb is None else bool(args.use_rgb)
    use_phys = (
        ckpt_has_phys
        if args.use_physics_inpainting is None
        else bool(args.use_physics_inpainting)
    )

    physics_token_dim = int(args.physics_token_dim)
    if use_phys and "physics_enc.weight" in state_dict:
        ckpt_phys_dim = int(state_dict["physics_enc.weight"].shape[1])
        if ckpt_phys_dim != physics_token_dim:
            print(
                f"[warn] override physics_token_dim {physics_token_dim} -> {ckpt_phys_dim} from checkpoint",
                flush=True,
            )
            physics_token_dim = ckpt_phys_dim

    model = SpineDiT(
        force_dim=args.force_dim,
        horizon=args.model_horizon,
        joint_dim=args.joint_dim,
        action_dim=args.action_dim,
        use_rgb=use_rgb,
        use_physics_inpainting=use_phys,
        physics_token_dim=physics_token_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps, device=device)

    env, env_use_rgb, first_obs, selected_backend = _init_env_with_rgb_fallback(
        args, use_rgb=bool(use_rgb)
    )
    action_low, action_high = env.action_spec
    action_mode = (
        _auto_action_mode(args.controller, action_low, action_high)
        if args.action_mode == "auto"
        else args.action_mode
    )

    if args.out.suffix.lower() == ".json" and not args.out.is_dir():
        out_dir = args.out.parent / args.out.stem
        summary_path = args.out
    else:
        out_dir = args.out
        summary_path = out_dir / f"{args.task}_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        for ep in range(args.episodes):
            if ep == 0:
                obs = first_obs
            else:
                obs = _safe_reset_env(env, args.rgb_init_timeout)
            if args.oracle_grasp:
                execute_oracle_grasp(env, args.task, mode=args.oracle_mode)
                if hasattr(env, "_get_observations"):
                    obs = env._get_observations(force_update=True)

            replay_q = None
            if args.policy_source == "replay":
                replay_q = _load_replay_traj(
                    args.replay_hdf5, args.replay_demo, args.joint_dim
                )

            t_start = time.time()
            plan_actions = None
            plan_index = 0
            joint_ids = env.robots[0].joint_indexes
            for _ in range(args.horizon):
                q_curr, force, force_vec, rgb = _extract_obs_joint_force(
                    obs,
                    joint_dim=args.joint_dim,
                    fallback_force=_get_body_force(env),
                    use_rgb=env_use_rgb,
                    rgb_key=args.rgb_key,
                    rgb_size=args.camera_height,
                )

                if args.policy_source == "replay":
                    idx = min(plan_index + 1, len(replay_q) - 1)
                    q_target = replay_q[idx]
                    plan_index += 1
                else:
                    if plan_actions is None or plan_index >= min(
                        args.resample_every, len(plan_actions)
                    ):
                        obs_joint = torch.tensor(
                            q_curr, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        obs_force = torch.tensor(
                            [force], dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        obs_rgb = None
                        if use_rgb:
                            if rgb is not None:
                                obs_rgb = rgb.to(device).unsqueeze(0)
                            else:
                                # Keep model input shape valid when camera rendering is unavailable.
                                obs_rgb = torch.zeros(
                                    1,
                                    3,
                                    args.camera_height,
                                    args.camera_width,
                                    dtype=torch.float32,
                                    device=device,
                                )
                        obs_phys_token = None
                        obs_phys_mask = None
                        if use_phys:
                            token_np = _build_physics_token(
                                force_scalar=force,
                                force_vec=force_vec,
                                token_dim=physics_token_dim,
                                contact_force_threshold=args.contact_force_threshold,
                                force_mag_clip=args.force_mag_clip,
                            )
                            obs_phys_token = (
                                torch.tensor(token_np, dtype=torch.float32, device=device)
                                .unsqueeze(0)
                            )
                            mask_value = (
                                1.0 if args.physics_inpaint_mode == "fullmask" else 0.0
                            )
                            obs_phys_mask = torch.tensor(
                                [mask_value], dtype=torch.float32, device=device
                            )
                        actions = _sample_actions(
                            model,
                            scheduler,
                            obs_joint,
                            obs_force,
                            obs_rgb,
                            obs_phys_token,
                            obs_phys_mask,
                            horizon=args.model_horizon,
                            action_dim=args.action_dim,
                            device=device,
                            sample_steps=args.sample_steps,
                        )
                        plan_actions = actions.detach().cpu().numpy()
                        plan_index = 0
                    q_target = plan_actions[plan_index]
                    plan_index += 1

                cmd = _format_action(
                    q_target,
                    q_curr,
                    action_low,
                    action_high,
                    action_mode,
                    args.gripper,
                )
                if args.execution_mode == "controller":
                    obs, _, _, _ = env.step(cmd)
                else:
                    # Diagnostic mode: directly set arm joints to isolate controller mismatch.
                    q_arm = q_target[: len(joint_ids)]
                    env.sim.data.qpos[joint_ids] = q_arm
                    env.sim.data.qvel[joint_ids] = 0.0
                    env.sim.forward()
                    env.sim.step()
                    if hasattr(env, "_get_observations"):
                        obs = env._get_observations(force_update=True)
                    else:
                        obs, _, _, _ = env.step(np.zeros_like(action_low))

            result = {
                "success": bool(env._check_success()),
                "completion_time": time.time() - t_start,
            }
            results.append(result)
            print(
                f"[ep {ep:03d}] success={result['success']} time={result['completion_time']:.2f}s",
                flush=True,
            )
            (out_dir / f"{args.task}_ep{ep:03d}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
    finally:
        _close_env_safely(env)

    summary = {
        "task": args.task,
        "episodes": args.episodes,
        "success_rate": float(np.mean([r["success"] for r in results])),
        "controller": args.controller,
        "action_mode": action_mode,
        "execution_mode": args.execution_mode,
        "policy_source": args.policy_source,
        "sample_steps": int(args.sample_steps),
        "resample_every": int(args.resample_every),
        "horizon": int(args.horizon),
        "controller_delta_limit": float(args.controller_delta_limit),
        "device": device,
        "ckpt": str(args.ckpt),
        "use_rgb": bool(use_rgb),
        "env_use_rgb": bool(env_use_rgb),
        "use_physics_inpainting": bool(use_phys),
        "physics_inpaint_mode": str(args.physics_inpaint_mode),
        "rgb_key": str(args.rgb_key),
    }
    summary["rgb_backend"] = selected_backend
    summary["out_dir"] = str(out_dir)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
