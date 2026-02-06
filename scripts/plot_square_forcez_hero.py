#!/usr/bin/env python
"""Plot Square force Z: Sim Force vs CITO (lambda)."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_style import apply_style


def _read_force_z(mimicgen_hdf5: Path, demo_key: str) -> np.ndarray:
    with h5py.File(mimicgen_hdf5, "r") as f:
        demo = f["data"][demo_key]
        force = np.array(demo["obs"]["robot0_ee_force"], dtype=float)
    return force[:, 2]


def _read_cito_lambda(refine_hdf5: Path, demo_key: str) -> np.ndarray:
    with h5py.File(refine_hdf5, "r") as f:
        demo = f["data"][demo_key]
        lam = np.array(demo["lambda_opt"], dtype=float)
        use_table_plane = int(f.attrs.get("use_table_plane", 1))
        contact_box_geom = f.attrs.get("contact_box_geom", "")
        if isinstance(contact_box_geom, bytes):
            contact_box_geom = contact_box_geom.decode("utf-8")
        has_box = bool(str(contact_box_geom).strip())
        demo_attrs = dict(f.attrs)
    if lam.ndim == 1:
        return lam
    if lam.shape[1] == 1:
        return lam[:, 0]
    # Prefer box-contact forces if both table + box constraints are active.
    if use_table_plane and has_box and lam.shape[1] % 2 == 0:
        box_cols = list(range(1, lam.shape[1], 2))
        lam_box = lam[:, box_cols]
        return np.linalg.norm(lam_box, axis=1)
    return np.linalg.norm(lam, axis=1)


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    window = int(window)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Square force Z hero figure.")
    parser.add_argument("--mimicgen-hdf5", type=Path, required=True)
    parser.add_argument("--refine-hdf5", type=Path, required=True)
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dual-axis", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--sim-label", type=str, default="Sim Force Z (Left)")
    parser.add_argument(
        "--cito-label", type=str, default="CITO Contact Force (Right)"
    )
    args = parser.parse_args()

    apply_style()

    sim_force_z = _read_force_z(args.mimicgen_hdf5, args.demo)
    cito_lambda = _read_cito_lambda(args.refine_hdf5, args.demo)

    T = min(len(sim_force_z), len(cito_lambda))
    sim_force_z = _smooth(sim_force_z[:T], args.smooth_window)
    cito_lambda = _smooth(cito_lambda[:T], args.smooth_window)

    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
    line1 = ax.plot(
        t,
        sim_force_z,
        color="#d62728",
        linewidth=1.2,
        label=args.sim_label,
    )
    if args.dual_axis:
        ax2 = ax.twinx()
        line2 = ax2.plot(
            t,
            cito_lambda,
            color="#1f77b4",
            linewidth=1.2,
            label=args.cito_label,
        )
        ax.set_ylabel("Sim Force (N)")
        ax2.set_ylabel("CITO Contact Force (N)")
        lines = line1 + line2
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, loc="upper right", frameon=True)
    else:
        ax.plot(
            t,
            cito_lambda,
            color="#1f77b4",
            linewidth=1.2,
            label="CITO Contact Force (lambda)",
        )
        ax.set_ylabel("Force (N)")
        ax.legend(loc="upper right", frameon=True)
    ax.set_xlabel("Timestep")
    title = args.title if args.title else "Square: Sim Force vs. CITO Force"
    ax.set_title(title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved figure to {args.output}")


if __name__ == "__main__":
    main()
