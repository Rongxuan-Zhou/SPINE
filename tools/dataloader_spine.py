from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _physics_tokens_from_force(
    force: np.ndarray,
    contact_force_threshold: float,
    force_mag_clip: float,
) -> np.ndarray:
    force = np.asarray(force, dtype=np.float32).reshape(-1)
    fnorm = np.abs(force)
    contact = (fnorm > float(contact_force_threshold)).astype(np.float32)
    normal_z = np.full_like(contact, 0.5, dtype=np.float32)
    denom = np.log1p(float(force_mag_clip))
    mag = np.log1p(np.clip(fnorm, 0.0, None)) / max(denom, 1e-6)
    mag = np.clip(mag, 0.0, 1.0).astype(np.float32)
    return np.stack([contact, normal_z, mag], axis=1)


class SpineH5Dataset(Dataset):
    """SPINE HDF5 loader supporting RGB + physics-token inpainting."""

    def __init__(
        self,
        h5_path: str | Path,
        force_dim: int = 1,
        horizon: int = 16,
        use_rgb: bool = True,
        rgb_key: str = "agentview_rgb",
        rgb_size: int = 84,
        use_physics_inpainting: bool = True,
        physics_token_dim: int = 3,
        physics_mask_prob: float = 0.5,
        contact_force_threshold: float = 2.0,
        force_mag_clip: float = 50.0,
    ):
        self.h5_path = str(h5_path)
        self.force_dim = int(force_dim)
        self.horizon = int(horizon)
        self.use_rgb = bool(use_rgb)
        self.rgb_key = str(rgb_key)
        self.rgb_size = int(rgb_size)
        self.use_physics_inpainting = bool(use_physics_inpainting)
        self.physics_token_dim = int(physics_token_dim)
        self.physics_mask_prob = float(physics_mask_prob)
        self.contact_force_threshold = float(contact_force_threshold)
        self.force_mag_clip = float(force_mag_clip)

        with h5py.File(self.h5_path, "r") as f:
            if "data" not in f:
                raise KeyError(f"{self.h5_path} missing 'data' group")
            self.keys = sorted(f["data"].keys())
        if not self.keys:
            raise ValueError(f"{self.h5_path} has zero demos")

    def __len__(self) -> int:
        return len(self.keys)

    def _slice_window(self, arr: np.ndarray, start: int, end: int) -> np.ndarray:
        chunk = arr[start:end]
        if chunk.shape[0] >= self.horizon:
            return chunk[: self.horizon]
        pad = np.repeat(chunk[-1:], self.horizon - chunk.shape[0], axis=0)
        return np.concatenate([chunk, pad], axis=0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        key = self.keys[idx]
        with h5py.File(self.h5_path, "r") as f:
            demo = f["data"][key]
            obs = demo["obs"]
            q = np.asarray(obs["joint_positions"], dtype=np.float32)
            force = np.asarray(obs["ee_forces"], dtype=np.float32)
            if force.ndim == 1:
                force = force[:, None]
            actions_raw = np.asarray(demo["actions"], dtype=np.float32)
            rgb = None
            if self.use_rgb:
                if self.rgb_key in obs:
                    rgb = np.asarray(obs[self.rgb_key], dtype=np.float32)
                elif "agentview_image" in obs:
                    rgb = np.asarray(obs["agentview_image"], dtype=np.float32)
                elif "robot0_eye_in_hand_image" in obs:
                    rgb = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.float32)
                else:
                    raise KeyError(
                        f"[{key}] no RGB key '{self.rgb_key}' or fallback in dataset"
                    )
            if "physics_tokens" in obs:
                physics_tokens = np.asarray(obs["physics_tokens"], dtype=np.float32)
            else:
                physics_tokens = _physics_tokens_from_force(
                    force[:, 0],
                    contact_force_threshold=self.contact_force_threshold,
                    force_mag_clip=self.force_mag_clip,
                )

        seq_len = min(q.shape[0], actions_raw.shape[0], force.shape[0], physics_tokens.shape[0])
        if self.use_rgb and rgb is not None:
            seq_len = min(seq_len, rgb.shape[0])
        if seq_len < 2:
            raise ValueError(f"[{key}] sequence too short: {seq_len}")

        if seq_len > self.horizon:
            start = np.random.randint(0, seq_len - self.horizon + 1)
        else:
            start = 0
        end = min(start + self.horizon, seq_len)

        q_w = self._slice_window(q, start, end)
        force_w = self._slice_window(force, start, end)
        act_w = self._slice_window(actions_raw, start, end)
        phys_w = self._slice_window(physics_tokens, start, end)

        obs_joint = torch.from_numpy(q_w[0])
        obs_force = torch.from_numpy(force_w[0][: self.force_dim])
        target_actions = torch.from_numpy(act_w)
        phys_target = torch.from_numpy(phys_w[0][: self.physics_token_dim])

        obs_rgb = torch.zeros(3, self.rgb_size, self.rgb_size, dtype=torch.float32)
        if self.use_rgb and rgb is not None:
            rgb_w = self._slice_window(rgb, start, end)
            rgb0 = rgb_w[0]
            if rgb0.max() > 1.0:
                rgb0 = rgb0 / 255.0
            rgb_t = torch.from_numpy(rgb0).permute(2, 0, 1).contiguous().float()
            if rgb_t.shape[-2] != self.rgb_size or rgb_t.shape[-1] != self.rgb_size:
                rgb_t = torch.nn.functional.interpolate(
                    rgb_t.unsqueeze(0),
                    size=(self.rgb_size, self.rgb_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            obs_rgb = rgb_t

        if self.use_physics_inpainting:
            mask_val = (
                1.0 if np.random.rand() < self.physics_mask_prob else 0.0
            )
        else:
            mask_val = 0.0

        return {
            "obs_joint": obs_joint.float(),
            "obs_force": obs_force.float(),
            "obs_rgb": obs_rgb.float(),
            "target_actions": target_actions.float(),
            "obs_phys_token": phys_target.float(),
            "obs_phys_mask": torch.tensor(mask_val, dtype=torch.float32),
        }
