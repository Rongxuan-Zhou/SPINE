"""Utility to project kinematic trajectories with a lightweight CITO/VSCM pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from spine.perception.kinematics.data_structures import (
    KinematicFrame,
    KinematicTrajectory,
    TrajectoryMetadata,
)
from .configs import CITOParameters
from .solver import CITOPlanner, Trajectory


def _load_kinematic_json(path: Path) -> KinematicTrajectory:
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata", {})
    frames_payload = payload.get("frames", [])
    frames: List[KinematicFrame] = []
    for f in frames_payload:
        frames.append(
            KinematicFrame(
                timestamp=float(f["timestamp"]),
                joint_positions=f.get("joint_positions", []),
                end_effector_pose=f["end_effector_pose"],
                base_frame=f.get("base_frame", "world"),
            )
        )
    traj = KinematicTrajectory(frames=frames, metadata=TrajectoryMetadata(**metadata))
    return traj


def _to_state_array(traj: KinematicTrajectory) -> np.ndarray:
    return np.stack(
        [np.array(f.end_effector_pose[:3], dtype=float) for f in traj.frames], axis=0
    )


def project_trajectory(
    input_path: Path,
    output_path: Path,
    table_height: float = 0.0,
    dt: float = 0.05,
    params: CITOParameters | None = None,
) -> Path:
    """Runs a simple CITO/VSCM projection over EE positions and writes an updated JSON."""
    traj = _load_kinematic_json(input_path)
    states = _to_state_array(traj)
    if states.shape[0] < 2:
        raise ValueError("Trajectory must contain at least 2 frames")
    controls = np.zeros((states.shape[0] - 1, states.shape[1]), dtype=float)
    target_states = states.copy()
    planner = CITOPlanner(params or CITOParameters(max_iters=5, step_size=0.05))

    def dyn_fn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x + dt * u

    def contact_fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        dist = float(x[2] - table_height)
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        return dist, normal

    init_traj = Trajectory(states=states, controls=controls)
    result = planner.optimize(init_traj, dyn_fn, contact_fn, target_states)
    projected_states = result.trajectory.states

    new_frames: List[dict[str, object]] = []
    for frame, pos in zip(traj.frames, projected_states):
        pose = list(pos) + list(frame.end_effector_pose[3:])
        new_frames.append(
            {
                "timestamp": frame.timestamp,
                "joint_positions": list(frame.joint_positions),
                "end_effector_pose": pose,
                "base_frame": frame.base_frame,
            }
        )
    metadata = traj.metadata.as_dict()
    metadata.setdefault("notes", {})  # type: ignore[assignment]
    metadata["notes"]["cito_projection"] = "ee_xyz projected with VSCM penalty"  # type: ignore[index]

    output = {"metadata": metadata, "frames": new_frames}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2))
    return output_path


__all__ = ["project_trajectory"]
