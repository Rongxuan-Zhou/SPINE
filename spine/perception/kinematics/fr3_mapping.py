"""FR3 joint/frame references for retargeting."""

from __future__ import annotations

from typing import Sequence

# Joint order matches menagerie XML / franka_description with explicit prefix.
FR3_JOINT_NAMES: list[str] = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

# Frames of interest
FR3_BASE_FRAME = "base"
FR3_FLANGE_FRAME = "link8"  # flange before hand; tcp is offset from this
FR3_TCP_OFFSET_M = (0.0, 0.0, 0.1034)
FR3_TCP_RPY = (0.0, 0.0, 0.0)


def validate_fr3_joint_positions(joints: Sequence[float]) -> None:
    """Checks that provided joint vector matches FR3 ordering and length."""
    if len(joints) != len(FR3_JOINT_NAMES):
        raise ValueError(
            f"FR3 expects {len(FR3_JOINT_NAMES)} joints, got {len(joints)}"
        )


__all__ = [
    "FR3_BASE_FRAME",
    "FR3_FLANGE_FRAME",
    "FR3_JOINT_NAMES",
    "FR3_TCP_OFFSET_M",
    "FR3_TCP_RPY",
    "validate_fr3_joint_positions",
]
