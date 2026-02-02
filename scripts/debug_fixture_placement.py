"""Visual sanity check for fixture placement and braced posture."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

from spine.simulation.robosuite_spine_envs import SpineNutAssemblySquare, SpineThreading
from scripts.run_spine import (
    _guarded_descent,
    _get_body_force,
    _get_eef_pose,
    execute_oracle_grasp,
)


def _build_env(task: str) -> suite.Env:
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["kp"] = 150
    controller_config["damping_ratio"] = 1.0
    controller_config["input_max"] = 4

    if task == "square":
        return SpineNutAssemblySquare(
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            use_camera_obs=False,
            control_freq=50,
            controller_configs=controller_config,
        )
    if task == "threading":
        return SpineThreading(
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            use_camera_obs=False,
            control_freq=50,
            controller_configs=controller_config,
        )
    raise ValueError(f"Unknown task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["square", "threading"], required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/debug_fixture.png"))
    parser.add_argument("--force_threshold", type=float, default=5.0)
    parser.add_argument(
        "--oracle_mode", choices=["waypoints", "teleport"], default="teleport"
    )
    args = parser.parse_args()

    env = _build_env(args.task)
    env.reset()
    execute_oracle_grasp(env, args.task, mode=args.oracle_mode)

    # try to seat the wrist on the fixture
    _guarded_descent(env, sigma=0.0, force_threshold=args.force_threshold)

    ee_pos, _ = _get_eef_pose(env)
    f6 = _get_body_force(env, "robot0_link6")
    f7 = _get_body_force(env, "robot0_link7")
    print(f"[debug] ee_pos={ee_pos} f_link6={f6:.3f} f_link7={f7:.3f}")

    img = env.sim.render(width=1024, height=768, camera_name="frontview")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(args.out.as_posix(), img)
    print(f"Saved fixture debug screenshot -> {args.out}")


if __name__ == "__main__":
    main()
