from pathlib import Path

import h5py

from spine.perception.kinematics.parsing import (
    parse_dexcap_episode_frames,
    parse_trajectory_json,
)
from spine.perception.kinematics.data_structures import TrajectoryMetadata


def test_parse_generic_json(tmp_path: Path) -> None:
    payload = {
        "frames": [
            {
                "timestamp": 0.0,
                "end_effector_pose": [0, 0, 0, 0, 0, 0, 1],
                "joint_positions": [0.1, 0.2],
            },
            {"time": 0.1, "ee_pose": [0, 0, 0, 0, 0, 0, 1], "joints": [0.3, 0.4]},
        ]
    }
    json_path = tmp_path / "clip.json"
    json_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    metadata = TrajectoryMetadata(source="dummy", clip_id="clip")

    traj = parse_trajectory_json(json_path, metadata)

    assert len(traj.frames) == 2
    assert traj.frames[0].timestamp == 0.0
    assert traj.frames[1].joint_positions == [0.3, 0.4]


def test_parse_dexcap_episode_frames(tmp_path: Path) -> None:
    episode = tmp_path / "episode_a"
    episode.mkdir()
    frame0 = episode / "frame_0"
    frame0.mkdir()
    (frame0 / "pose_2.txt").write_text("0 0 0 0 0 0 1", encoding="utf-8")
    (frame0 / "right_hand_joint.txt").write_text("0.5 0.6", encoding="utf-8")
    frame1 = episode / "frame_1"
    frame1.mkdir()
    (frame1 / "pose_2.txt").write_text("0 0 0 0 0 0 1", encoding="utf-8")
    metadata = TrajectoryMetadata(source="dexcap", clip_id="episode_a")

    traj = parse_dexcap_episode_frames(episode, metadata, frame_dt=0.02)

    assert len(traj.frames) == 2
    assert traj.frames[0].timestamp == 0.0
    assert traj.frames[1].timestamp == 0.02
    assert traj.frames[0].joint_positions == [0.5, 0.6]


def test_parse_mimicgen_hdf5(tmp_path: Path) -> None:
    h5_path = tmp_path / "demo.hdf5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("data/demo_0/obs")
        grp.create_dataset("robot0_eef_pos", data=[[0, 0, 0], [1, 1, 1]])
        grp.create_dataset("robot0_eef_quat", data=[[0, 0, 0, 1], [0, 0, 0, 1]])
        grp.create_dataset("robot0_joint_pos", data=[[0.1, 0.2], [0.3, 0.4]])
    from spine.perception.kinematics.sources_mimicgen import MimicGenAdapter
    from spine.perception.kinematics.configs import MimicGenConfig, AugmentationConfig

    adapter = MimicGenAdapter(
        MimicGenConfig(
            dataset_root=tmp_path,
            augmentations=AugmentationConfig(time_warp_factor=0.0),
        )
    )

    traj = next(adapter._load_hdf5(h5_path))

    assert len(traj.frames) == 2
    assert traj.frames[0].end_effector_pose[:3] == [0.0, 0.0, 0.0]
    assert traj.frames[1].joint_positions == [0.3, 0.4]
