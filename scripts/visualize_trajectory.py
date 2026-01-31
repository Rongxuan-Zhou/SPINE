#!/usr/bin/env python
"""Quick visualization for kinematic trajectory JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import importlib


def load_xyz(path: Path) -> tuple[list[float], list[float], list[float], list[float]]:
    data = json.loads(path.read_text())
    frames = data.get("frames", [])
    t = [f.get("timestamp", i) for i, f in enumerate(frames)]
    xs, ys, zs = [], [], []
    for f in frames:
        pose: Sequence[float] = f["end_effector_pose"]
        xs.append(pose[0])
        ys.append(pose[1])
        zs.append(pose[2])
    return t, xs, ys, zs


def plot_trajectory(path: Path) -> None:
    t, xs, ys, zs = load_xyz(path)
    fig = plt.figure(figsize=(7.5, 3.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(xs, ys, zs, label=path.name, lw=1.2, alpha=0.9)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(loc="upper left", fontsize=6, frameon=False)
    # flatten z spikes for readability
    ax3d.set_box_aspect([1, 1, 0.8])

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(t, zs, label="z", lw=1.0)
    ax.plot(t, xs, label="x", lw=1.0)
    ax.plot(t, ys, label="y", lw=1.0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position (m)")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(True, alpha=0.2)
    fig.suptitle(path.name, fontsize=12)
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize kinematic trajectory JSON and save PNG.")
    parser.add_argument("paths", nargs="+", type=Path, help="Trajectory JSON paths")
    parser.add_argument("--output_dir", type=Path, default=Path("results"), help="Directory to save PNGs")
    parser.add_argument("--science", action="store_true", help="Use SciencePlots style if available.")
    args = parser.parse_args()

    if args.science:
        try:
            if importlib.util.find_spec("scienceplots"):
                plt.style.use(["science", "ieee", "no-latex"])
        except Exception:
            pass

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for p in args.paths:
        fig = plot_trajectory(p)
        out_path = args.output_dir / f"{p.stem}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
