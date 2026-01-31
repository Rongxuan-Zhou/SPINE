import json
from pathlib import Path

from spine.planning.cito.projection import project_trajectory


def _write_kinematic(path: Path) -> None:
    payload = {
        "metadata": {"source": "dummy", "clip_id": "penetrating"},
        "frames": [
            {
                "timestamp": 0.0,
                "end_effector_pose": [0, 0, -0.05, 0, 0, 0, 1],
                "joint_positions": [0] * 7,
            },
            {
                "timestamp": 0.05,
                "end_effector_pose": [0, 0, -0.02, 0, 0, 0, 1],
                "joint_positions": [0] * 7,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_project_trajectory_raises_penetration(tmp_path: Path) -> None:
    inp = tmp_path / "traj.json"
    out = tmp_path / "proj.json"
    _write_kinematic(inp)

    project_trajectory(inp, out, table_height=0.0, dt=0.05)

    payload = json.loads(out.read_text())
    z_values = [frame["end_effector_pose"][2] for frame in payload["frames"]]
    # Should reduce penetration (z closer to or above 0)
    assert z_values[0] > -0.05
    assert len(payload["frames"]) == 2
