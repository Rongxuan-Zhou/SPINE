#!/usr/bin/env python
"""Interactive Plotly overlay of multiple kinematic trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay trajectories in one interactive Plotly figure.")
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input trajectory JSON files (from generator/projection scripts).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path")
    return parser.parse_args()


def load_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"{path} 不包含 frames")
    t = np.array([f["timestamp"] for f in frames], dtype=float)
    xyz = np.array([f["end_effector_pose"][:3] for f in frames], dtype=float)
    return t, xyz[:, 0], xyz[:, 1], xyz[:, 2]


def main() -> None:
    args = parse_args()
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("3D trajectory", "Position vs time"),
        horizontal_spacing=0.08,
    )

    for path in args.inputs:
        t, x, y, z = load_xyz(path)
        name = path.stem
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name=name,
                legendgroup=name,
                line=dict(width=4),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=t, y=x, mode="lines", name=f"{name}-x", legendgroup=name, line=dict(color="#ff7f0e")),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=t, y=y, mode="lines", name=f"{name}-y", legendgroup=name, line=dict(color="#2ca02c")),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=t, y=z, mode="lines", name=f"{name}-z", legendgroup=name, line=dict(color="#1f77b4")),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="Trajectory overlay",
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
        xaxis_title="t (s)",
        yaxis_title="position (m)",
        legend=dict(itemsizing="constant"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output))
    print(f"Saved interactive comparison to {args.output}")


if __name__ == "__main__":
    main()
