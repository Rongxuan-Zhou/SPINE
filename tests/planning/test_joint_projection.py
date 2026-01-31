import json
from pathlib import Path

import numpy as np

from spine.planning.cito.joint_projection import JointProjectionConfig, project_joint_trajectory


def _write_joint_json(path: Path, q_list: list[list[float]]) -> None:
    frames = []
    for i, q in enumerate(q_list):
        frames.append(
            {
                "timestamp": i * 0.05,
                "joint_positions": q,
                "end_effector_pose": [0, 0, 0, 0, 0, 0, 1],
            }
        )
    payload = {"metadata": {"source": "dummy", "clip_id": "joint_test"}, "frames": frames}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_joint_projection_clamps_velocity(tmp_path: Path) -> None:
    # Large jump should be clamped
    q0 = [0.0] * 7
    q1 = [3.5] * 7  # exceeds limit
    inp = tmp_path / "in.json"
    out = tmp_path / "out.json"
    _write_joint_json(inp, [q0, q1])

    cfg = JointProjectionConfig(vel_limit=1.0, acc_limit=5.0, dt=0.05)
    project_joint_trajectory(inp, out, cfg)

    payload = json.loads(out.read_text())
    q_proj = np.array(payload["frames"][1]["joint_positions"])
    assert np.all(q_proj <= 3.0)  # joint limit clamp
    max_step = np.abs(q_proj - np.array(q0)).max()
    # Allow step to be limited by either vel bound or remaining joint range (e.g., joint4 upper=-0.0698)
    allowable = max(cfg.vel_limit * cfg.dt, np.abs(-0.0698 - 0.0))
    assert max_step <= allowable + 1e-6


def test_joint_projection_requires_7dof(tmp_path: Path) -> None:
    inp = tmp_path / "in.json"
    out = tmp_path / "out.json"
    _write_joint_json(inp, [[0.0, 0.1]])  # wrong DOF
    try:
        project_joint_trajectory(inp, out)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-7DOF input")
