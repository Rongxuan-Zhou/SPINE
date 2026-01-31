import json
import glob
import numpy as np
import os
import argparse


def convert_mimicgen():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Specific JSON file path")
    parser.add_argument("--gripper", type=float, default=0.0, help="Padding value for gripper DOF")
    parser.add_argument("--dt", type=float, default=0.05, help="Assumed dt for duration estimate")
    parser.add_argument("--output", type=str, default="data/fr3_q_ref.npy", help="Output npy path")
    args = parser.parse_args()

    # 1. 寻找源文件
    if args.input:
        target_files = [args.input]
    else:
        files = sorted(glob.glob("data/kinematics/mimicgen/*.json"), key=os.path.getmtime)
        if not files:
            print("❌ No MimicGen JSON files found in data/kinematics/mimicgen/")
            return
        target_files = [files[-1]]  # 最新

    target_file = target_files[0]
    print(f"📂 Processing: {target_file}")

    with open(target_file, "r") as f:
        data = json.load(f)

    # 2. 提取关节位置
    traj = []
    for frame in data.get("frames", []):
        q = frame.get("joint_positions", [])
        traj.append(q)
    q_ref = np.array(traj)  # (T, 7) typically

    # 3. 维度适配 (7DOF -> 9DOF)
    if q_ref.shape[1] == 7:
        print(f"⚠️ Detected 7DOF data. Padding gripper dims with {args.gripper} ...")
        T = q_ref.shape[0]
        gripper_state = np.full((T, 2), args.gripper)
        q_ref = np.hstack([q_ref, gripper_state])

    # 4. 保存
    save_path = args.output
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, q_ref)
    print(f"✅ Saved to {save_path}")
    print(f"   Shape: {q_ref.shape}")
    print(f"   Duration: {q_ref.shape[0] * args.dt:.2f}s (assuming dt={args.dt})")


if __name__ == "__main__":
    convert_mimicgen()
