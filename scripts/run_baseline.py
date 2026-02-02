"""Run SPINE baseline (free-floating) policies for NutAssemblySquare and Threading."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

from spine.simulation.robosuite_spine_envs import SpineNutAssemblySquare, SpineThreading


def noisy_step(env, action: np.ndarray, sigma: float) -> tuple:
    noise = np.random.normal(0.0, sigma, size=action.shape)
    return env.step(action + noise)


def _get_eef_pose(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    site_id = robot.eef_site_id
    pos = env.sim.data.site_xpos[site_id].copy()
    quat = env.sim.data.site_xquat[site_id].copy()
    return pos, quat


def _get_body_force(env, body_name: str) -> float:
    body_id = env.sim.model.body_name2id(body_name)
    wrench = env.sim.data.cfrc_ext[body_id]
    return float(np.linalg.norm(wrench[:3]))


def _joint_torque_energy(env, joint_indexes: list[int]) -> float:
    torques = env.sim.data.qfrc_actuator[joint_indexes]
    return float(np.sum(torques**2))


def _build_env(task: str, controller_config: dict) -> suite.Env:
    if task == "square":
        return SpineNutAssemblySquare(
            robots="Panda",
            has_renderer=False,
            use_camera_obs=False,
            control_freq=50,
            controller_configs=controller_config,
        )
    if task == "threading":
        return SpineThreading(
            robots="Panda",
            has_renderer=False,
            use_camera_obs=False,
            control_freq=50,
            controller_configs=controller_config,
        )
    raise ValueError(f"Unknown task: {task}")


def _policy_baseline(task: str) -> Callable[[np.ndarray], np.ndarray]:
    def policy(obs: np.ndarray) -> np.ndarray:
        # Placeholder policy: hover above target then descend along z.
        # This uses a simple proportional step in OSC pose space.
        action = np.zeros(7, dtype=np.float32)
        # move downward slowly
        action[2] = -0.02
        # keep gripper closed
        action[-1] = -1.0
        return action

    return policy


def run_episode(env, policy, sigma: float, horizon: int) -> dict:
    obs = env.reset()
    t_start = time.time()
    joint_ids = env.robots[0].joint_indexes

    ee_positions = []
    ideal_positions = []
    contact_forces = []
    torque_energy = 0.0

    # ideal trajectory: straight-line descent in z for jitter metric
    ee_pos0, _ = _get_eef_pose(env)
    ee_target = ee_pos0.copy()
    ee_target[2] -= 0.08

    for step in range(horizon):
        action = policy(obs)
        obs, _, done, _ = noisy_step(env, action, sigma=sigma)
        ee_pos, _ = _get_eef_pose(env)
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
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["square", "threading"], required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--horizon", type=int, default=200)
    args = parser.parse_args()

    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["kp"] = 150
    controller_config["damping_ratio"] = 1.0
    controller_config["input_max"] = 4

    env = _build_env(args.task, controller_config)
    policy = _policy_baseline(args.task)

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for ep in range(args.episodes):
        result = run_episode(env, policy, sigma=args.sigma, horizon=args.horizon)
        results.append(result)
        (args.out / f"{args.task}_baseline_ep{ep:03d}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    summary = {
        "task": args.task,
        "episodes": args.episodes,
        "sigma": args.sigma,
        "success_rate": float(np.mean([r["success"] for r in results])),
    }
    (args.out / f"{args.task}_baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
