import json
from pathlib import Path
from typing import Iterable, List

import pytest

from spine.perception.kinematics import (
    DexCapConfig,
    KinematicFrame,
    KinematicGenerator,
    KinematicGeneratorConfig,
    KinematicSourceAdapter,
    KinematicTrajectory,
    TrajectoryMetadata,
)


class _DummyAdapter(KinematicSourceAdapter):
    def __init__(self, name: str, trajectories: Iterable[KinematicTrajectory]):
        super().__init__(name)
        self._trajectories = list(trajectories)

    def generate(self) -> Iterable[KinematicTrajectory]:
        return iter(self._trajectories)


def _make_trajectory(clip_id: str, timestamps: List[float]) -> KinematicTrajectory:
    frames = [
        KinematicFrame(
            timestamp=t,
            joint_positions=[0.0, 1.0],
            end_effector_pose=[0, 0, 0, 0, 0, 0, 1],
        )
        for t in timestamps
    ]
    metadata = TrajectoryMetadata(
        source="dummy", clip_id=clip_id, augmentations={"position_noise_m": 0.01}
    )
    return KinematicTrajectory(frames=frames, metadata=metadata)


def test_generator_writes_json(tmp_path: Path) -> None:
    traj = _make_trajectory("clip_a", [0.0, 0.1, 0.2])
    adapter = _DummyAdapter("dummy_source", [traj])
    generator = KinematicGenerator(
        output_dir=tmp_path, adapters=[adapter], max_trajectories=None
    )

    paths = generator.run()

    assert len(paths) == 1
    saved = paths[0]
    assert saved.exists()
    with saved.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    assert payload["metadata"]["clip_id"] == "clip_a"
    assert len(payload["frames"]) == 3


def test_generator_respects_max_count(tmp_path: Path) -> None:
    trajectories = [_make_trajectory(f"clip_{i}", [0.0, 0.1]) for i in range(3)]
    adapter = _DummyAdapter("dummy_source", trajectories)
    generator = KinematicGenerator(
        output_dir=tmp_path, adapters=[adapter], max_trajectories=2
    )

    paths = generator.run()

    assert len(paths) == 2
    assert all(path.exists() for path in paths)


def test_trajectory_validation_rejects_bad_timestamps() -> None:
    traj = _make_trajectory("bad_clip", [0.0, -0.1])
    with pytest.raises(ValueError):
        traj.validate()


def test_config_uses_dexcap_key() -> None:
    config = KinematicGeneratorConfig(
        output_dir=Path("out"),
        dexcap=DexCapConfig(dataset_root=Path("/tmp/dexcap")),
    )

    sources = config.active_sources()

    assert sources == ["dexcap"]
    payload = config.as_dict()
    assert "dexcap" in payload
    assert payload["dexcap"]["dataset_root"] == "/tmp/dexcap"
