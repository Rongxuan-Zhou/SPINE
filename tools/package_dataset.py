import argparse
import glob
import os

import h5py
import numpy as np


def package_dataset(root: str, output: str, split: str = "train"):
    """
    将散落的 *_q_opt.npy / *_force.npy 打包为 RoboSuite/RoboMimic 兼容的 HDF5。
    - root: 包含批处理输出的目录（如 data/spine_dataset_square）
    - output: 输出 HDF5 路径
    - split: 写入 mask/train 或 mask/valid
    """
    q_files = sorted(glob.glob(os.path.join(root, "*_q_opt.npy")))
    f_files = sorted(glob.glob(os.path.join(root, "*_force.npy")))
    bases_q = {os.path.basename(f).replace("_q_opt.npy", "") for f in q_files}
    bases_f = {os.path.basename(f).replace("_force.npy", "") for f in f_files}
    bases = sorted(bases_q.intersection(bases_f))

    if not bases:
        print(f"❌ No paired q_opt/force found in {root}")
        return

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with h5py.File(output, "w") as h5f:
        data_grp = h5f.create_group("data")
        mask_grp = h5f.create_group("mask")
        mask_train = mask_grp.create_dataset(split, shape=(len(bases),), dtype=h5py.string_dtype())

        for idx, base in enumerate(bases):
            q_path = os.path.join(root, f"{base}_q_opt.npy")
            f_path = os.path.join(root, f"{base}_force.npy")
            q = np.load(q_path)
            forces = np.load(f_path)

            # 创建 demo 组
            demo_grp = data_grp.create_group(f"demo_{idx}")
            obs_grp = demo_grp.create_group("obs")
            # 关节 (T, 9)
            obs_grp.create_dataset("joint_positions", data=q, compression="gzip")
            # 力 (T, 1) - 如需扩维到 3/6，可在此处修改
            obs_grp.create_dataset("ee_forces", data=forces.reshape(-1, 1), compression="gzip")
            # actions: 简单用下一帧的关节差分或直接下一帧关节。这里用下一帧关节 (末帧重复)
            actions = np.concatenate([q[1:], q[-1:]], axis=0)
            demo_grp.create_dataset("actions", data=actions, compression="gzip")

            mask_train[idx] = f"demo_{idx}"

    print(f"✅ Packaged {len(bases)} demos to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Package SPINE outputs into HDF5 (RoboMimic/RoboSuite style)."
    )
    parser.add_argument("--root", type=str, required=True, help="Dataset root (e.g., data/spine_dataset_square)")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--split", type=str, default="train", help="Mask split name (train/valid)")
    args = parser.parse_args()

    package_dataset(args.root, args.output, args.split)


if __name__ == "__main__":
    main()
