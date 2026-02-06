"""Render annotated dimension diagrams (SVG/PNG) for MimicGen assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def _draw_dim_arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    text: str,
    text_offset: Tuple[float, float] = (0.0, 0.0),
    fontsize: int = 10,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="black"),
    )
    mid = ((start[0] + end[0]) / 2 + text_offset[0], (start[1] + end[1]) / 2 + text_offset[1])
    ax.text(mid[0], mid[1], text, ha="center", va="center", fontsize=fontsize, color="black")


def _setup_axis(ax: plt.Axes, title: str) -> None:
    ax.set_aspect("equal", "box")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


def _draw_square(ax: plt.Axes, payload: dict[str, float]) -> None:
    outer = payload["nut_outer_mm"]
    inner = payload["nut_inner_mm"]
    peg = payload["peg_side_mm"]
    clearance = payload["clearance_mm"]

    outer_half = outer / 2.0
    inner_half = inner / 2.0
    peg_half = peg / 2.0

    # Nut outer
    ax.add_patch(
        Rectangle(
            (-outer_half, -outer_half),
            outer,
            outer,
            fill=False,
            lw=2.0,
            edgecolor="#1f77b4",
        )
    )
    # Inner hole
    ax.add_patch(
        Rectangle(
            (-inner_half, -inner_half),
            inner,
            inner,
            fill=False,
            lw=2.0,
            edgecolor="#ff7f0e",
        )
    )
    # Peg (dashed)
    ax.add_patch(
        Rectangle(
            (-peg_half, -peg_half),
            peg,
            peg,
            fill=False,
            lw=1.6,
            linestyle="--",
            edgecolor="#2ca02c",
        )
    )

    # Dimension arrows
    offset = outer_half + 6.0
    _draw_dim_arrow(
        ax,
        (-outer_half, offset),
        (outer_half, offset),
        f"Nut outer = {outer:.2f} mm",
        text_offset=(0, 3),
    )
    _draw_dim_arrow(
        ax,
        (-inner_half, -offset),
        (inner_half, -offset),
        f"Nut inner = {inner:.2f} mm",
        text_offset=(0, -3),
    )
    _draw_dim_arrow(
        ax,
        (offset, -peg_half),
        (offset, peg_half),
        f"Peg = {peg:.2f} mm",
        text_offset=(4, 0),
    )

    ax.text(
        0,
        -offset - 10,
        f"Clearance = {clearance:.2f} mm",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )


def _draw_threading(ax: plt.Axes, payload: dict[str, float]) -> None:
    outer_x = payload["ring_outer_x_mm"]
    outer_y = payload["ring_outer_y_mm"]
    inner_x = payload["ring_inner_x_mm"]
    inner_y = payload["ring_inner_y_mm"]
    thickness = payload["ring_thickness_mm"]
    needle_d = payload["needle_diameter_mm"]

    outer_half_x = outer_x / 2.0
    outer_half_y = outer_y / 2.0
    inner_half_x = inner_x / 2.0
    inner_half_y = inner_y / 2.0

    # Outer ring bbox
    ax.add_patch(
        Rectangle(
            (-outer_half_x, -outer_half_y),
            outer_x,
            outer_y,
            fill=False,
            lw=2.0,
            edgecolor="#d62728",
        )
    )
    # Inner hole
    ax.add_patch(
        Rectangle(
            (-inner_half_x, -inner_half_y),
            inner_x,
            inner_y,
            fill=False,
            lw=2.0,
            edgecolor="#9467bd",
        )
    )
    # Needle diameter circle (reference)
    ax.add_patch(
        Circle(
            (outer_half_x + needle_d * 0.8, 0.0),
            needle_d / 2.0,
            fill=False,
            lw=1.6,
            edgecolor="#2ca02c",
        )
    )
    ax.text(
        outer_half_x + needle_d * 0.8,
        -needle_d / 2.0 - 4,
        f"Needle Ø {needle_d:.2f} mm",
        ha="center",
        fontsize=10,
    )

    offset_y = outer_half_y + 6.0
    _draw_dim_arrow(
        ax,
        (-outer_half_x, offset_y),
        (outer_half_x, offset_y),
        f"Ring outer X = {outer_x:.2f} mm",
        text_offset=(0, 3),
    )
    offset_x = outer_half_x + 6.0
    _draw_dim_arrow(
        ax,
        (offset_x, -outer_half_y),
        (offset_x, outer_half_y),
        f"Ring outer Y = {outer_y:.2f} mm",
        text_offset=(4, 0),
    )
    _draw_dim_arrow(
        ax,
        (-inner_half_x, -offset_y),
        (inner_half_x, -offset_y),
        f"Hole X = {inner_x:.2f} mm",
        text_offset=(0, -3),
    )
    _draw_dim_arrow(
        ax,
        (-offset_x, -inner_half_y),
        (-offset_x, inner_half_y),
        f"Hole Y = {inner_y:.2f} mm",
        text_offset=(-4, 0),
    )

    ax.text(
        0,
        -offset_y - 10,
        f"Ring thickness ≈ {thickness:.2f} mm",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("artifacts/mimicgen_physical_dims.json"),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("artifacts/mimicgen_dims"),
    )
    args = parser.parse_args()

    payload = json.loads(args.json.read_text(encoding="utf-8"))
    square = payload["square"]
    threading = payload["threading"]

    square_plot = {
        "nut_outer_mm": float(square["spine_target"]["nut_outer_mm_scaled"][0]),
        "nut_inner_mm": float(square["spine_target"]["nut_inner_mm"]),
        "peg_side_mm": float(square["spine_target"]["peg_side_mm"]),
        "clearance_mm": float(square["spine_target"]["clearance_mm"]),
    }

    threading_plot = {
        "ring_outer_x_mm": float(threading["ring_bbox_mm"][0]),
        "ring_outer_y_mm": float(threading["ring_bbox_mm"][1]),
        "ring_inner_x_mm": float(threading["ring_inner_hole_mm"][0]),
        "ring_inner_y_mm": float(threading["ring_inner_hole_mm"][1]),
        "ring_thickness_mm": float(threading["ring_thickness_mm"]),
        "needle_diameter_mm": float(threading["needle_diameter_mm"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _setup_axis(axes[0], "Square (SPINE target)")
    _draw_square(axes[0], square_plot)
    _setup_axis(axes[1], "Threading (MimicGen)")
    _draw_threading(axes[1], threading_plot)

    fig.tight_layout()

    png_path = args.out_prefix.with_suffix(".png")
    svg_path = args.out_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
