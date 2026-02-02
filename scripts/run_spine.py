"""Run SPINE contact-rich strategies for NutAssemblySquare and Threading."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_controller_config

from spine.simulation.robosuite_spine_envs import SpineNutAssemblySquare, SpineThreading


def _log_contact_geoms(env) -> None:
    link_names = [
        name
        for name in env.sim.model.geom_names
        if "robot0_link6" in name or "robot0_link7" in name
    ]
    if link_names:
        print(f"[debug] link6/link7 geoms: {link_names}")


def noisy_step(env, action: np.ndarray, sigma: float) -> tuple:
    noise = np.random.normal(0.0, sigma, size=action.shape)
    return env.step(action + noise)


def _get_eef_pose(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    site_id = robot.eef_site_id
    pos = env.sim.data.site_xpos[site_id].copy()
    if hasattr(env.sim.data, "site_xquat"):
        quat = env.sim.data.site_xquat[site_id].copy()
    else:
        mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
        quat = T.convert_quat(T.mat2quat(mat), to="wxyz")
    return pos, quat


def _get_body_force(env, body_name: str) -> float:
    body_id = env.sim.model.body_name2id(body_name)
    wrench = env.sim.data.cfrc_ext[body_id]
    return float(np.linalg.norm(wrench[:3]))


def _joint_torque_energy(env, joint_indexes: list[int]) -> float:
    torques = env.sim.data.qfrc_actuator[joint_indexes]
    return float(np.sum(torques**2))


def _get_fixture_height(env, geom_name: str) -> float | None:
    try:
        geom_id = env.sim.model.geom_name2id(geom_name)
    except Exception:
        return None
    geom_pos = env.sim.data.geom_xpos[geom_id]
    geom_size = env.sim.model.geom_size[geom_id]
    return float(geom_pos[2] + geom_size[2])


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


def _build_env(task: str, controller_config: dict, debug: bool) -> suite.Env:
    render = bool(debug)
    if task == "square":
        env = SpineNutAssemblySquare(
            robots="Panda",
            has_renderer=render,
            use_camera_obs=False,
            control_freq=50,
            controller_configs=controller_config,
        )
    elif task == "threading":
        env = SpineThreading(
            robots="Panda",
            has_renderer=render,
            use_camera_obs=False,
            control_freq=50,
            controller_configs=controller_config,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    _log_contact_geoms(env)
    return env


def _guarded_descent(env, sigma: float, force_threshold: float) -> bool:
    """Descend slowly until wrist contact exceeds threshold (with debounce)."""
    reached = False
    consecutive = 0
    for _ in range(200):
        action = np.zeros(7, dtype=np.float32)
        action[2] = -0.01
        action[-1] = -1.0
        noisy_step(env, action, sigma)
        f_wrist = _get_body_force(env, "robot0_link7")
        if f_wrist > force_threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 5:
            reached = True
            break
    return reached


def _policy_spine(task: str):
    """Returns phase-based spine policy for task."""

    def policy(env, phase: int) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[-1] = -1.0
        # Phase 0: approach fixture (small lateral move)
        if phase == 0:
            action[0] = 0.01
            action[1] = -0.01
        # Phase 1: pivot insertion (tilt end-effector)
        elif phase == 1:
            action[3] = 0.03
            action[4] = -0.03
        # Phase 2: stable insertion
        else:
            action[2] = -0.01
        return action

    return policy


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
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip(gain * delta, -0.05, 0.05)
            action[-1] = 1.0
            env.step(action)

    pregrasp = obj_pos + np.array([0.0, 0.0, 0.08])
    grasp = obj_pos + np.array([0.0, 0.0, 0.01])

    step_to(pregrasp, steps=80)
    step_to(grasp, steps=60, gain=2.0)

    for _ in range(30):
        action = np.zeros(7, dtype=np.float32)
        action[-1] = -1.0
        env.step(action)

    riser_height = _get_fixture_height(env, "fixture_riser")
    z_min = riser_height + 0.02 if riser_height is not None else grasp[2] + 0.05

    if mode == "teleport":
        _teleport_object_to_gripper(env, body_id, z_min)
    else:
        _teleport_object_to_gripper(env, body_id, z_min)

    safe = obj_pos + np.array([0.0, 0.0, 0.12])
    step_to(safe, steps=60, gain=2.0)


def run_episode(
    env,
    sigma: float,
    horizon: int,
    force_threshold: float,
    task: str,
    debug: bool,
    oracle_mode: str,
) -> dict:
    obs = env.reset()
    execute_oracle_grasp(env, task, mode=oracle_mode)
    t_start = time.time()
    joint_ids = env.robots[0].joint_indexes

    ee_positions = []
    ideal_positions = []
    contact_forces = []
    torque_energy = 0.0

    ee_pos0, _ = _get_eef_pose(env)
    ee_target = ee_pos0.copy()
    ee_target[2] -= 0.08

    # Guarded descent
    contact_reached = _guarded_descent(env, sigma, force_threshold)
    policy = _policy_spine(task)
    phase = 1 if contact_reached else 2

    for step in range(horizon):
        action = policy(env, phase)
        obs, _, done, _ = noisy_step(env, action, sigma=sigma)
        ee_pos, _ = _get_eef_pose(env)
        if debug and step % 10 == 0:
            f6 = _get_body_force(env, "robot0_link6")
            f7 = _get_body_force(env, "robot0_link7")
            print(f"[debug] step={step} f_link6={f6:.3f} f_link7={f7:.3f}")
        interp = (step + 1) / horizon
        ideal_pos = ee_pos0 * (1 - interp) + ee_target * interp
        ee_positions.append(ee_pos)
        ideal_positions.append(ideal_pos)
        contact_forces.append(
            {
                "link6": _get_body_force(env, "robot0_link6"),
                "link7": _get_body_force(env, "robot0_link7"),
            }
        )
        torque_energy += _joint_torque_energy(env, joint_ids)
        if done:
            break

    ee_positions = np.array(ee_positions)
    ideal_positions = np.array(ideal_positions)
    jitter = float(np.std(np.linalg.norm(ee_positions - ideal_positions, axis=1)))

    result = {
        "success": bool(env._check_success()),
        "completion_time": time.time() - t_start,
        "ee_jitter": jitter,
        "contact_force_wrist": contact_forces,
        "joint_torque_integral": torque_energy / env.control_freq,
        "guarded_contact": contact_reached,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["square", "threading"], required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/spine"))
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--force_threshold", type=float, default=5.0)
    parser.add_argument(
        "--oracle_mode", choices=["waypoints", "teleport"], default="waypoints"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["kp"] = 150
    controller_config["damping_ratio"] = 1.0
    controller_config["input_max"] = 4

    env = _build_env(args.task, controller_config, debug=args.debug)

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for ep in range(args.episodes):
        result = run_episode(
            env,
            sigma=args.sigma,
            horizon=args.horizon,
            force_threshold=args.force_threshold,
            task=args.task,
            debug=args.debug,
            oracle_mode=args.oracle_mode,
        )
        results.append(result)
        (args.out / f"{args.task}_spine_ep{ep:03d}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    summary = {
        "task": args.task,
        "episodes": args.episodes,
        "sigma": args.sigma,
        "success_rate": float(np.mean([r["success"] for r in results])),
        "contact_rate": float(np.mean([r["guarded_contact"] for r in results])),
    }
    (args.out / f"{args.task}_spine_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
