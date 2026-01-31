import pytest

from spine.perception.kinematics.fr3_mapping import (
    FR3_BASE_FRAME,
    FR3_FLANGE_FRAME,
    FR3_JOINT_NAMES,
    FR3_TCP_OFFSET_M,
    validate_fr3_joint_positions,
)


def test_fr3_joint_order_length() -> None:
    assert FR3_JOINT_NAMES == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    validate_fr3_joint_positions([0] * 7)
    with pytest.raises(ValueError):
        validate_fr3_joint_positions([0, 1])


def test_fr3_frames_defined() -> None:
    assert FR3_BASE_FRAME == "base"
    assert FR3_FLANGE_FRAME == "link8"
    assert FR3_TCP_OFFSET_M[2] > 0
