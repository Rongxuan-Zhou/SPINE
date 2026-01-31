import glob
import os
import shlex
import subprocess
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/kinematics/mimicgen")
    parser.add_argument("--output_dir", type=str, default="data/spine_dataset")
    parser.add_argument("--urdf", type=str, default="models/fr3.urdf")
    parser.add_argument("--table_height", type=float, default=0.20)
    parser.add_argument("--comp_penalty", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--python_bin", type=str, default="python", help="Python executable (e.g., conda run -n spine_opt python)")
    parser.add_argument("--limit", type=int, default=None, help="Stop after producing this many successful outputs (paired).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip inputs that already have outputs in output_dir.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"❌ No JSON files found in {args.input_dir}")
        return

    log_path = os.path.join(args.output_dir, "batch_log.txt")
    stats = {"success": 0, "failed": 0, "timeout": 0, "skipped": 0}
    existing_pairs = set()
    if args.skip_existing:
        q_files = glob.glob(os.path.join(args.output_dir, "*_q_opt.npy"))
        f_files = glob.glob(os.path.join(args.output_dir, "*_force.npy"))
        q_bases = {os.path.basename(p).replace("_q_opt.npy", "") for p in q_files}
        f_bases = {os.path.basename(p).replace("_force.npy", "") for p in f_files}
        existing_pairs = q_bases & f_bases
    existing_count = len(existing_pairs)
    target_total = args.limit if args.limit is not None else None
    if target_total is not None and existing_count >= target_total:
        print(f"✅ Already have {existing_count} paired outputs (>= limit {target_total}); nothing to do.")
        return

    python_cmd = shlex.split(args.python_bin)
    print(f"🏭 SPINE Factory | input={len(json_files)} | out={args.output_dir}")
    print(f"  table_height={args.table_height}, comp_penalty={args.comp_penalty}")
    if args.skip_existing:
        print(f"  skip_existing=True (already paired: {existing_count})")
    if target_total is not None:
        print(f"  limit={target_total}")

    with open(log_path, "w") as flog:
        for jf in tqdm(json_files, desc="SPINE Batch"):
            base = os.path.basename(jf).replace(".json", "")
            if args.skip_existing and base in existing_pairs:
                stats["skipped"] += 1
                continue
            if target_total is not None and (existing_count + stats["success"]) >= target_total:
                break
            # 1) 转换 JSON -> npy 到固定路径，避免冲突
            ref_path = "data/current_batch_ref.npy"
            cmd_convert = [
                *python_cmd, "tools/convert_mimicgen_to_npy.py",
                "--input", jf,
                "--output", ref_path,
            ]
            subprocess.run(cmd_convert, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 2) 运行 CITO（自动窗口检测）
            cmd_cito = [
                *python_cmd, "tools/casadi_cito_mvp.py",
                "--urdf", args.urdf,
                "--q-ref", ref_path,
                "--table-height", str(args.table_height),
                "--comp-penalty", str(args.comp_penalty),
                "--dt", str(args.dt),
            ]
            try:
                res = subprocess.run(cmd_cito, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                stats["timeout"] += 1
                flog.write(f"{base}: TIMEOUT\n")
                continue

            if res.returncode == 0 and "Optimization Success" in res.stdout:
                stats["success"] += 1
                # 3) 归档结果
                out_q = os.path.join(args.output_dir, f"{base}_q_opt.npy")
                out_f = os.path.join(args.output_dir, f"{base}_force.npy")
                if os.path.exists("data/fr3_opt_result.npy"):
                    os.rename("data/fr3_opt_result.npy", out_q)
                if os.path.exists("data/fr3_opt_forces.npy"):
                    os.rename("data/fr3_opt_forces.npy", out_f)
            else:
                stats["failed"] += 1
                flog.write(f"{base}: FAILED\n{res.stdout}\n{res.stderr}\n")

    print("----- Batch Summary -----")
    print(f"Success: {stats['success']}")
    print(f"Failed : {stats['failed']}")
    print(f"Timeout: {stats['timeout']}")
    print(f"Skipped: {stats['skipped']}")
    if target_total is not None:
        print(f"Total paired (existing+new): {existing_count + stats['success']}")
    print(f"Log   : {log_path}")


if __name__ == "__main__":
    main()
