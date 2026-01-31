#!/usr/bin/env python
"""Minimal robosuite playback for MimicGen trajectories on Panda.

Adds optional HDF5 alignment for MimicGen pick_place: load demo from hdf5,
set object (cube) initial pose, and replay gripper qpos alongside arm joints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import time
import h5py
import mujoco

# Force GLFW and disable robosuite's default GPU (EGL) path to avoid EGL issues in headless setups.
os.environ.setdefault("MUJOCO_GL", "glfw")
import robosuite.macros as macros

macros.MUJOCO_GPU_RENDERING = False
import robosuite as suite


def load_joint_trajectory(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    joints = [frame["joint_positions"] for frame in data["frames"]]
    arr = np.array(joints, dtype=float)
    if arr.shape[1] != 7:
        raise ValueError(f"Expected 7 joints for Panda, got shape {arr.shape}")
    return arr


def load_hdf5_demo(h5_path: Path, demo: str):
    """Load MimicGen hdf5 demo for object pose + gripper."""
    with h5py.File(h5_path, "r") as h:
        g = h["data"][demo]
        obs = g["obs"]
        joint_pos = np.array(obs["robot0_joint_pos"], dtype=float)
        grip_qpos = np.array(obs["robot0_gripper_qpos"], dtype=float)  # (N,2)
        obj_pose = None
        obj_goal = None
        eef_quat = None
        if "object" in obs:
            obj = np.array(obs["object"], dtype=float)
            # Heuristic: first 7 dims = cube pose (pos+quat), next 3 = goal pos
            obj_pose = obj[:, :7]
            obj_goal = obj[:, 7:10]
        if "robot0_eef_quat" in obs:
            eef_quat = np.array(obs["robot0_eef_quat"], dtype=float)
        return joint_pos, grip_qpos, obj_pose, obj_goal, eef_quat


def yaw_to_quat(rad: float) -> np.ndarray:
    """Return wxyz quaternion for rotation about z by rad."""
    half = rad * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of wxyz quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def rotmat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion."""
    import math

    tr = np.trace(mat)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (mat[2, 1] - mat[1, 2]) / S
        qy = (mat[0, 2] - mat[2, 0]) / S
        qz = (mat[1, 0] - mat[0, 1]) / S
    else:
        if (mat[0, 0] > mat[1, 1]) and (mat[0, 0] > mat[2, 2]):
            S = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
            qw = (mat[2, 1] - mat[1, 2]) / S
            qx = 0.25 * S
            qy = (mat[0, 1] + mat[1, 0]) / S
            qz = (mat[0, 2] + mat[2, 0]) / S
        elif mat[1, 1] > mat[2, 2]:
            S = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
            qw = (mat[0, 2] - mat[2, 0]) / S
            qx = (mat[0, 1] + mat[1, 0]) / S
            qy = 0.25 * S
            qz = (mat[1, 2] + mat[2, 1]) / S
        else:
            S = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
            qw = (mat[1, 0] - mat[0, 1]) / S
            qx = (mat[0, 2] + mat[2, 0]) / S
            qy = (mat[1, 2] + mat[2, 1]) / S
            qz = 0.25 * S
    return np.array([qw, qx, qy, qz], dtype=float)


def patch_cv2_destroy() -> None:
    """Monkey-patch cv2.destroyAllWindows to ignore GUI errors in headless builds."""
    try:
        import cv2

        if getattr(cv2, "_spine_patched_destroy", False):
            return
        original: Callable[[], None] = cv2.destroyAllWindows

        def safe_destroy() -> None:
            try:
                original()
            except Exception:
                # Ignore GUI teardown errors (e.g., headless OpenCV builds).
                pass

        cv2.destroyAllWindows = safe_destroy  # type: ignore[assignment]
        cv2._spine_patched_destroy = True  # type: ignore[attr-defined]
    except Exception:
        # If OpenCV not present or patch fails, continue; not fatal.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay MimicGen joint trajectory on robosuite Panda.")
    parser.add_argument("trajectory", type=Path, help="Path to trajectory JSON with joint_positions")
    parser.add_argument("--env", type=str, default="Lift", help="robosuite env name")
    parser.add_argument("--horizon", type=int, default=None, help="Optional horizon to truncate playback")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering (opens Mujoco viewer)")
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="Playback speed multiplier relative to control_freq (1.0 = real-time)",
    )
    parser.add_argument("--mjcf", type=Path, default=None, help="Custom MJCF for robot (e.g., FR3)")
    parser.add_argument("--robot-name", type=str, default="Panda", help="Robot name (default Panda)")
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=None,
        help="Optional MimicGen hdf5 to align object pose and gripper (demo name must exist).",
    )
    parser.add_argument("--demo", type=str, default=None, help="Demo name inside hdf5 (e.g., demo_0)")
    parser.add_argument(
        "--renderer",
        type=str,
        default="mujoco",
        choices=["mujoco", "opencv"],
        help="Renderer backend for on-screen display. Use 'mujoco' to avoid OpenCV GUI requirements.",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="If set, save an MP4/AVI to this path using an offscreen camera render.",
    )
    parser.add_argument(
        "--record-camera",
        type=str,
        default="agentview",
        help="Camera name to use when recording (robosuite camera).",
    )
    parser.add_argument(
        "--record-res",
        type=str,
        default="960x540",
        help="Resolution WIDTHxHEIGHT for recording (e.g., 960x540). Pass your viewer size to match on-screen.",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=20.0,
        help="FPS for VideoWriter; default matches control_freq for near real-time playback.",
    )
    parser.add_argument(
        "--base-yaw-deg",
        type=float,
        default=None,
        help="Optional yaw rotation (deg) applied to robot base to align world frames.",
    )
    parser.add_argument(
        "--base-xyz",
        type=str,
        default=None,
        help="Optional base translation 'x,y,z' to apply to robot root body.",
    )
    parser.add_argument(
        "--debug-fk",
        action="store_true",
        help="Print FK vs hdf5 eef quat (first frame) to inspect orientation mismatch.",
    )
    parser.add_argument(
        "--eef-quat-offset",
        type=str,
        default=None,
        help="Optional w,x,y,z quaternion to rotate simulated eef frame to match hdf5 (e.g., '1,0,0,0').",
    )
    parser.add_argument(
        "--auto-eef-offset",
        action="store_true",
        help="Automatically compute eef quat offset from first frame (hdf5 vs FK) and apply to sim.",
    )
    parser.add_argument(
        "--eef-site",
        type=str,
        default="gripper0_grip_site",
        help="Site name to use for FK orientation (default gripper0_grip_site).",
    )
    parser.add_argument(
        "--per-frame-eef-offset",
        action="store_true",
        help="If set, compute quat offset each frame (from hdf5 vs FK) and apply dynamically.",
    )
    args = parser.parse_args()

    rec_width, rec_height = (640, 480)
    if args.record_res:
        try:
            w_str, h_str = args.record_res.lower().split("x")
            rec_width, rec_height = int(w_str), int(h_str)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid --record-res '{args.record_res}', expected WIDTHxHEIGHT") from exc

    # Avoid OpenCV GUI teardown crashes in headless builds when viewer closes.
    if args.render:
        patch_cv2_destroy()

    q_traj = load_joint_trajectory(args.trajectory)
    grip_traj = None
    obj_pose = None
    obj_goal = None
    eef_quat = None
    eef_quat_all = None
    if args.hdf5 and args.demo:
        jp_h5, grip_h5, obj_h5, obj_goal_h5, eef_quat_h5 = load_hdf5_demo(args.hdf5, args.demo)
        q_traj = jp_h5
        grip_traj = grip_h5
        obj_pose = obj_h5[0] if obj_h5 is not None else None
        obj_goal = obj_goal_h5[0] if obj_goal_h5 is not None else None
        if eef_quat_h5 is not None:
            eef_quat_all = eef_quat_h5
            eef_quat = eef_quat_h5[0]
    first_qpos = q_traj[0].copy()
    if args.horizon:
        q_traj = q_traj[: args.horizon]
        if grip_traj is not None:
            grip_traj = grip_traj[: args.horizon]
        if eef_quat_all is not None:
            eef_quat_all = eef_quat_all[: args.horizon]

    record = args.record_path is not None
    writer = None
    recorded_frames = 0
    record_path = args.record_path
    if record:
        import cv2  # Lazy import so non-record runs don't require it
        record_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = record_path.suffix.lower()
        primary_codec = "mp4v" if suffix == ".mp4" else "MJPG"
        fallback_codec = "MJPG"

        def make_writer(codec: str) -> cv2.VideoWriter:
            return cv2.VideoWriter(
                str(record_path), cv2.VideoWriter_fourcc(*codec), args.record_fps, (rec_width, rec_height)
            )

        writer = make_writer(primary_codec)
        if not writer.isOpened():
            writer = make_writer(fallback_codec)
        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to open VideoWriter for {record_path}. "
                "Try a different extension (.avi) or install ffmpeg/libx264."
            )

    env_kwargs = dict(
        env_name=args.env,
        robots=args.robot_name,
        has_renderer=args.render,
        renderer=args.renderer,
        has_offscreen_renderer=record,
        render_camera=args.record_camera if record else None,
        use_camera_obs=False,
        control_freq=20,
    )
    if args.mjcf:
        env_kwargs["gripper_types"] = "None"
        env_kwargs["robot_configs"] = {"model_file": str(args.mjcf)}
    env = suite.make(**env_kwargs)
    env.reset()
    # Set robot to first frame to avoid mismatch with default home pose
    robot = env.robots[0]
    # Optional base transform alignment
    if args.base_yaw_deg is not None or args.base_xyz is not None:
        root_body = robot.robot_model.root_body
        try:
            bid = env.sim.model.body_name2id(root_body)
            if args.base_yaw_deg is not None:
                quat = yaw_to_quat(np.deg2rad(args.base_yaw_deg))
                env.sim.model.body_quat[bid] = quat
            if args.base_xyz is not None:
                try:
                    x_str, y_str, z_str = args.base_xyz.split(",")
                    env.sim.model.body_pos[bid] = np.array([float(x_str), float(y_str), float(z_str)])
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"Invalid --base-xyz '{args.base_xyz}', expected 'x,y,z'") from exc
            env.sim.forward()
        except Exception:
            # Silently ignore if base transform fails
            pass
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = first_qpos  # type: ignore[attr-defined]
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0  # type: ignore[attr-defined]
    env.sim.forward()
    # Align object initial pose if provided and env exposes sim
    if obj_pose is not None:
        # Try several names: Lift usually uses a free joint for cube/object0
        pose6 = np.concatenate([obj_pose[:3], obj_pose[3:]])
        tried = []
        for jname in ["object0:joint", "cube:joint", "cube:joint0"]:
            try:
                env.sim.data.set_joint_qpos(jname, pose6)
                env.sim.forward()
                break
            except Exception as exc:  # noqa: BLE001
                tried.append((jname, str(exc)))
        else:
            # Fallback: set body pose if joint names failed
            try:
                cube_id = env.sim.model.body_name2id("cube")
                env.sim.model.body_pos[cube_id] = obj_pose[:3]
                env.sim.model.body_quat[cube_id] = obj_pose[3:]
                env.sim.forward()
            except Exception:
                # If still fails, ignore; logging could be added here if needed
                pass
    # Optional goal placement if available and env supports it (Lift uses goal for sparse reward)
    if obj_goal is not None:
        try:
            env.sim.model.body_pos[env.cube_goal_body_id] = obj_goal
            env.sim.forward()
        except Exception:
            pass

    if eef_quat is not None:
        # Compute FK quat for current joints and compare/apply offset
        try:
            site_id = env.sim.model.site_name2id(args.eef_site)
            mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
            quat_sim = rotmat_to_quat(mat)
            if args.eef_quat_offset:
                try:
                    qw, qx, qy, qz = [float(x) for x in args.eef_quat_offset.split(",")]
                    offset = np.array([qw, qx, qy, qz])
                    quat_sim = quat_multiply(offset, quat_sim)
                except Exception as exc:  # noqa: BLE001
                    print(f"Invalid --eef-quat-offset '{args.eef_quat_offset}': {exc}")
            if args.auto_eef_offset:
                inv_sim = quat_conjugate(quat_sim)
                q_offset = quat_multiply(eef_quat, inv_sim)
                quat_sim = quat_multiply(q_offset, quat_sim)
                print("DEBUG FK: auto applied eef offset wxyz:", q_offset)
            if args.debug_fk:
                print("DEBUG FK: hdf5 eef quat first frame:", eef_quat)
                print("DEBUG FK: sim  eef quat first frame (after offset):", quat_sim)
                if not args.auto_eef_offset:
                    inv_sim = quat_conjugate(quat_sim)
                    q_offset = quat_multiply(eef_quat, inv_sim)
                    print("DEBUG FK: suggested eef_quat_offset wxyz:", q_offset)
        except Exception as exc:  # noqa: BLE001
            if args.debug_fk:
                print(f"DEBUG FK failed: {exc}")

    dt = 1.0 / env.control_freq

    try:
        for idx, q in enumerate(q_traj):
            robot = env.robots[0]
            env.sim.data.qpos[robot._ref_joint_pos_indexes] = q  # type: ignore[attr-defined]
            env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0  # type: ignore[attr-defined]
            if grip_traj is not None:
                # two gripper joints for Panda
                gq = grip_traj[idx]
                env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = gq  # type: ignore[attr-defined]
            env.sim.forward()
            env.step(np.zeros(robot.dof))
            if args.render:
                env.render()
                # Slow down to human-viewable speed; playback_rate>1 speeds up.
                time.sleep(max(0.0, dt / max(args.playback_rate, 1e-6)))
            if record and writer is not None:
                import cv2  # type: ignore

                # mujoco expects (width, height); returned frame shape is (height, width, 3)
                frame = env.sim.render(rec_width, rec_height, camera_name=args.record_camera)
                frame_uint8 = np.asarray(frame, dtype=np.uint8)
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                writer.write(np.ascontiguousarray(frame_bgr))
                recorded_frames += 1
    finally:
        # Explicitly close viewer to avoid EGL teardown warnings.
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
        if writer is not None:
            writer.release()

    print(f"Replayed {len(q_traj)} steps in env {args.env}")
    if record and writer is not None:
        size_bytes = record_path.stat().st_size if record_path.exists() else 0
        print(
            f"Saved video to {record_path} at {rec_width}x{rec_height}@{args.record_fps}fps "
            f"({recorded_frames} frames, {size_bytes/1024:.1f} KB)."
        )


if __name__ == "__main__":
    main()
