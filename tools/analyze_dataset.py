import argparse
import glob
import os

import numpy as np
import pinocchio as pin


def load_model(urdf_path: str, frame_name: str):
    model = pin.buildModelFromUrdf(urdf_path)
    if not model.existFrame(frame_name):
        raise ValueError(
            f"Frame '{frame_name}' not found in model. "
            f"Available: {[f.name for f in model.frames]}"
        )
    data = model.createData()
    fid = model.getFrameId(frame_name)
    return model, data, fid


def analyze_dataset(root: str, urdf: str, frame: str, table_height: float):
    force_files = sorted(glob.glob(os.path.join(root, "*_force.npy")))
    q_files = sorted(glob.glob(os.path.join(root, "*_q_opt.npy")))

    if not force_files or not q_files:
        print(f"❌ No force/q_opt files found in {root}")
        return

    # ensure pairing
    base_force = {os.path.basename(f).replace("_force.npy", "") for f in force_files}
    base_q = {os.path.basename(f).replace("_q_opt.npy", "") for f in q_files}
    bases = sorted(base_force.intersection(base_q))

    model, data, fid = load_model(urdf, frame)

    mean_forces = []
    max_forces = []
    zmins = []
    zero_force_count = 0

    for base in bases:
        f_path = os.path.join(root, f"{base}_force.npy")
        q_path = os.path.join(root, f"{base}_q_opt.npy")
        forces = np.load(f_path).flatten()
        q_traj = np.load(q_path)

        mean_f = float(np.mean(forces))
        max_f = float(np.max(forces))
        mean_forces.append(mean_f)
        max_forces.append(max_f)
        if np.isclose(mean_f, 0.0):
            zero_force_count += 1

        # compute min z
        z_vals = []
        for q in q_traj:
            pin.framesForwardKinematics(model, data, q)
            z_vals.append(data.oMf[fid].translation[2])
        zmins.append(float(np.min(z_vals)))

    mean_force_all = float(np.mean(mean_forces))
    max_force_all = float(np.mean(max_forces))
    inpainted = sum(1 for m in max_forces if m > 0.5)
    total = len(bases)
    residual_penetration = sum(1 for z in zmins if z < table_height - 1e-3)

    print(f"📊 Dataset: {root}")
    print(f"  Trajectories analyzed : {total}")
    print(f"  Mean(force) overall   : {mean_force_all:.3f} N")
    print(f"  Mean(max force)       : {max_force_all:.3f} N")
    print(
        f"  Inpainting rate       : {inpainted}/{total} "
        f"({100*inpainted/total:.1f}%)  [max force > 0.5N]"
    )
    print(f"  Zero-force trajectories: {zero_force_count}")
    print(
        f"  Residual penetration  : {residual_penetration} "
        f"traj with z_min < table({table_height})"
    )
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SPINE batch outputs (force/q_opt)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="One or more output directories (e.g., data/spine_dataset_square)",
    )
    parser.add_argument("--urdf", type=str, default="models/fr3.urdf")
    parser.add_argument("--frame", type=str, default="fr3_link8")
    parser.add_argument(
        "--table-heights",
        nargs="+",
        type=float,
        default=None,
        help="Table heights matching datasets order. If not set, uses 0.0 for all.",
    )
    args = parser.parse_args()

    table_heights = (
        args.table_heights
        if args.table_heights and len(args.table_heights) == len(args.datasets)
        else [0.0] * len(args.datasets)
    )
    if args.table_heights and len(args.table_heights) != len(args.datasets):
        print("⚠️ table-heights length mismatch; using 0.0 for all.")

    for ds, th in zip(args.datasets, table_heights):
        analyze_dataset(ds, args.urdf, args.frame, th)


if __name__ == "__main__":
    main()
