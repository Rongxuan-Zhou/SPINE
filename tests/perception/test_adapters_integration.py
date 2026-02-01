import json
from pathlib import Path

import pytest

from spine.perception.kinematics import (
    DexCapAdapter,
    DexCapConfig,
    KinematicGenerator,
    R2R2RAdapter,
    R2R2RConfig,
)


def _write_r2r2r_json(path: Path) -> None:
    payload = {
        "frames": [
            {
                "timestamp": 0.0,
                "world_pose": [0, 0, 0, 0, 0, 0, 1],
                "joint_positions": [0.1] * 7,
            },
            {
                "timestamp": 0.05,
                "world_pose": [0.01, 0, 0, 0, 0, 0, 1],
                "joint_positions": [0.2] * 7,
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_dexcap_episode(root: Path) -> Path:
    episode_dir = root / "episode_01"
    episode_dir.mkdir(parents=True, exist_ok=True)
    for idx, joints in enumerate(([0.3] * 7, [0.4] * 7)):
        frame_dir = episode_dir / f"frame_{idx:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        (frame_dir / "pose_2.txt").write_text("0 0 0 0 0 0 1", encoding="utf-8")
        (frame_dir / "right_hand_joint.txt").write_text(
            " ".join(str(x) for x in joints), encoding="utf-8"
        )
    return episode_dir


@pytest.fixture(autouse=True)
def _skip_dep_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPINE_SKIP_DEP_CHECK", "1")


def test_r2r2r_adapter_and_generator(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip_r2r2r.json"
    _write_r2r2r_json(clip_path)
    adapter = R2R2RAdapter(
        R2R2RConfig(capture_root=tmp_path, enable_gaussian_splatting=False)
    )
    generator = KinematicGenerator(
        output_dir=tmp_path / "out", adapters=[adapter], max_trajectories=None
    )

    outputs = generator.run()

    assert outputs and outputs[0].exists()
    content = json.loads(outputs[0].read_text())
    assert content["metadata"]["source"] == "r2r2r"
    assert len(content["frames"]) == 2


def test_dexcap_adapter_and_generator(tmp_path: Path) -> None:
    _write_dexcap_episode(tmp_path)
    adapter = DexCapAdapter(DexCapConfig(dataset_root=tmp_path))
    generator = KinematicGenerator(
        output_dir=tmp_path / "out_dexcap", adapters=[adapter], max_trajectories=None
    )

    outputs = generator.run()

    assert outputs and outputs[0].exists()
    payload = json.loads(outputs[0].read_text())
    assert payload["metadata"]["source"] == "dexcap"
    assert payload["frames"][0]["joint_positions"] == [pytest.approx(0.3)] * 7
