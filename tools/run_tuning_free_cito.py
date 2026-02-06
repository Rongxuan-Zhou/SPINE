#!/usr/bin/env python
"""Run tuning-free CITO on a single MimicGen HDF5 demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from spine.planning.cito.tuning_free import TuningFreeCITO, TuningFreeCITOConfig


def _read_model_xml(demo_group) -> str:
    if "model_file" not in demo_group.attrs:
        raise ValueError("HDF5 demo missing model_file attr (MJCF XML).")
    xml = demo_group.attrs["model_file"]
    if isinstance(xml, bytes):
        return xml.decode("utf-8")
    return str(xml)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tuning-free CITO on one demo.")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--contact-site", type=str, default=None)
    parser.add_argument("--table-height", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--output-q", type=Path, required=True)
    parser.add_argument("--output-lambda", type=Path, required=True)
    parser.add_argument("--output-tau", type=Path, default=None)
    parser.add_argument("--output-vel", type=Path, default=None)
    parser.add_argument("--output-diag", type=Path, default=None)
    parser.add_argument("--scvx-iters", type=int, default=5)
    parser.add_argument("--penalty-loops", type=int, default=6)
    parser.add_argument("--trust-region", type=float, default=0.1)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        demo = f["data"][args.demo]
        q_ref = np.array(demo["obs"]["robot0_joint_pos"], dtype=float)
        model_xml = _read_model_xml(demo)

    cfg = TuningFreeCITOConfig(
        dt=args.dt,
        scvx_iters=args.scvx_iters,
        penalty_loops=args.penalty_loops,
        trust_region=args.trust_region,
    )
    solver = TuningFreeCITO(
        model_xml=model_xml,
        contact_site=args.contact_site,
        table_height=args.table_height,
        config=cfg,
    )
    result = solver.solve(q_ref)

    args.output_q.parent.mkdir(parents=True, exist_ok=True)
    args.output_lambda.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_q, result.q_opt)
    np.save(args.output_lambda, result.lambda_opt)
    if args.output_tau:
        args.output_tau.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_tau, result.tau_opt)
    if args.output_vel:
        args.output_vel.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_vel, result.v_opt)
    if args.output_diag:
        args.output_diag.parent.mkdir(parents=True, exist_ok=True)
        args.output_diag.write_text(
            json.dumps(result.diagnostics, indent=2), encoding="utf-8"
        )

    print(f"✅ Saved q_opt to {args.output_q}")
    print(f"✅ Saved lambda_opt to {args.output_lambda}")
    if args.output_tau:
        print(f"✅ Saved tau_opt to {args.output_tau}")
    if args.output_vel:
        print(f"✅ Saved v_opt to {args.output_vel}")
    if args.output_diag:
        print(f"✅ Saved diagnostics to {args.output_diag}")


if __name__ == "__main__":
    main()
