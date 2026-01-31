import h5py
import torch
from torch.utils.data import Dataset


class SpineH5Dataset(Dataset):
    """
    Minimal loader for SPINE packaged HDF5 (joint_positions, ee_forces, actions).
    - Randomly samples a horizon window.
    - Returns noisy_action (initially target + tiny noise), obs_joint/obs_force (first frame),
      and target_actions (ground-truth actions to denoise).
    """

    def __init__(self, h5_path, force_dim=1, horizon=16, normalize=True):
        self.h5_path = h5_path
        self.keys = []
        with h5py.File(h5_path, "r") as f:
            self.keys = list(f["data"].keys())
            if normalize:
                # 为了启动快，使用占位均值/方差；如需精确可预先统计后写入
                # 例：Square 数据可填 self.force_mean=1.585, std=2.0 (approx)
                self.force_mean = torch.tensor(0.0, dtype=torch.float32)
                self.force_std = torch.tensor(1.0, dtype=torch.float32)
            else:
                self.force_mean = torch.tensor(0.0, dtype=torch.float32)
                self.force_std = torch.tensor(1.0, dtype=torch.float32)

        self.horizon = horizon
        self.force_dim = force_dim

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            k = self.keys[idx]
            q = torch.tensor(
                f["data"][k]["obs"]["joint_positions"][()], dtype=torch.float32
            )  # (T,9)
            force = torch.tensor(
                f["data"][k]["obs"]["ee_forces"][()], dtype=torch.float32
            )  # (T,1)

        # 归一化（简单占位，可改为预计算统计）
        force = (force - self.force_mean) / (self.force_std + 1e-6)

        # 随机截取长度 horizon
        if q.shape[0] > self.horizon:
            start = torch.randint(0, q.shape[0] - self.horizon, (1,)).item()
        else:
            start = 0
        q = q[start : start + self.horizon]
        force = force[start : start + self.horizon]

        # 动作标签：用下一帧关节（末帧重复）
        actions = torch.cat([q[1:], q[-1:].clone()], dim=0)  # (H,9)
        noisy_action = actions + 0.01 * torch.randn_like(actions)

        # 观测取窗口首帧
        obs_joint = q[0]  # (9,)
        obs_force = force[0][: self.force_dim]  # (force_dim,)

        return noisy_action, obs_joint, obs_force, actions
