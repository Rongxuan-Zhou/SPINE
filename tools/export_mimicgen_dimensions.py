"""Export physical dimensions for MimicGen Square / Threading assets."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from robosuite.utils.mjcf_utils import xml_path_completion
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("robosuite is required to export asset dimensions") from exc


def _bbox_from_geoms(geoms: list[ET.Element]) -> tuple[np.ndarray, np.ndarray]:
    mins = np.array([1e9, 1e9, 1e9], dtype=float)
    maxs = -mins
    for geom in geoms:
        size = np.array([float(x) for x in geom.get("size", "").split()])
        pos = np.array([float(x) for x in geom.get("pos", "0 0 0").split()])
        if size.shape[0] != 3:
            continue
        mins = np.minimum(mins, pos - size)
        maxs = np.maximum(maxs, pos + size)
    return mins, maxs


def _parse_vec(value: str | None) -> list[float] | None:
    if not value:
        return None
    parts = value.split()
    if not parts:
        return None
    return [float(x) for x in parts]


def _square_assets() -> dict[str, Any]:
    square_xml = xml_path_completion("objects/square-nut.xml")
    pegs_xml = xml_path_completion("arenas/pegs_arena.xml")

    square_root = ET.parse(square_xml).getroot()
    obj_body = square_root.find(".//body[@name='object']")
    if obj_body is None:
        raise RuntimeError("square-nut.xml missing body 'object'")
    square_geoms = list(obj_body.findall("geom"))
    square_min, square_max = _bbox_from_geoms(square_geoms)

    # Ring-only geoms (exclude handle) -> used to estimate inner hole size
    ring_geoms: list[ET.Element] = []
    for geom in square_geoms:
        pos = _parse_vec(geom.get("pos")) or [0.0, 0.0, 0.0]
        # handle is offset at x ~= 0.054, so keep only near-center ring pieces
        if abs(pos[0]) <= 0.04 and abs(pos[1]) <= 0.04:
            ring_geoms.append(geom)

    ring_min, ring_max = _bbox_from_geoms(ring_geoms)
    ring_outer = ring_max - ring_min

    # Estimate ring wall thickness via smallest half-extent on x/y
    ring_sizes: list[list[float]] = []
    for geom in ring_geoms:
        size_vec = _parse_vec(geom.get("size"))
        if size_vec is None or len(size_vec) != 3:
            continue
        ring_sizes.append(size_vec)

    if ring_sizes:
        min_half_x = min(size[0] for size in ring_sizes)
        min_half_y = min(size[1] for size in ring_sizes)
        wall_x = 2.0 * min_half_x
        wall_y = 2.0 * min_half_y
        inner_dims = ring_outer.copy()
        inner_dims[0] = max(0.0, ring_outer[0] - 2.0 * wall_x)
        inner_dims[1] = max(0.0, ring_outer[1] - 2.0 * wall_y)
        inner_dims[2] = ring_outer[2]
    else:
        wall_x = 0.0
        wall_y = 0.0
        inner_dims = np.zeros(3, dtype=float)

    pegs_root = ET.parse(pegs_xml).getroot()
    peg_body = pegs_root.find(".//body[@name='peg1']")
    if peg_body is None:
        raise RuntimeError("pegs_arena.xml missing body 'peg1'")
    peg_geom = peg_body.find("geom")
    peg_size = np.array([float(x) for x in peg_geom.get("size", "").split()])
    peg_full = peg_size * 2.0

    # SPINE target scaling: peg side 20mm, clearance 0.8mm (nut inner side 20.8mm)
    peg_side_default = float(peg_size[0] * 2.0)  # meters
    target_peg_side = 0.02
    target_clearance = 0.0008
    target_nut_inner = target_peg_side + target_clearance
    inner_default = float(inner_dims[0])
    nut_scale = target_nut_inner / inner_default if inner_default > 0 else 1.0
    peg_scale = target_peg_side / peg_side_default if peg_side_default > 0 else 1.0
    nut_outer_scaled = ring_outer * nut_scale
    nut_thickness_scaled = ring_outer[2] * nut_scale
    peg_length_scaled = float(peg_full[2] * peg_scale)
    clearance_default = inner_default - peg_side_default

    return {
        "square_nut_bbox_m": (square_max - square_min).tolist(),
        "square_nut_bbox_mm": ((square_max - square_min) * 1000).tolist(),
        "square_nut_bbox_min_m": square_min.tolist(),
        "square_nut_bbox_max_m": square_max.tolist(),
        "square_ring_outer_m": ring_outer.tolist(),
        "square_ring_outer_mm": (ring_outer * 1000).tolist(),
        "square_ring_inner_m": inner_dims.tolist(),
        "square_ring_inner_mm": (inner_dims * 1000).tolist(),
        "square_ring_wall_mm": [wall_x * 1000, wall_y * 1000, ring_outer[2] * 1000],
        "peg1_half_size_m": peg_size.tolist(),
        "peg1_full_size_mm": (peg_size * 2 * 1000).tolist(),
        "square_clearance_default_mm": clearance_default * 1000,
        "spine_target": {
            "peg_side_mm": target_peg_side * 1000,
            "peg_length_mm": peg_length_scaled * 1000,
            "nut_inner_mm": target_nut_inner * 1000,
            "clearance_mm": target_clearance * 1000,
            "peg_scale": peg_scale,
            "nut_scale": nut_scale,
            "nut_outer_mm_scaled": (nut_outer_scaled * 1000).tolist(),
            "nut_thickness_mm_scaled": float(nut_thickness_scaled * 1000),
        },
    }


def _threading_assets() -> dict[str, Any]:
    from mimicgen.models.robosuite.objects.composite.needle import NeedleObject
    from mimicgen.models.robosuite.objects.composite.ring_tripod import RingTripodObject

    needle = NeedleObject("needle")
    ring_tripod = RingTripodObject("tripod")

    # Needle bbox
    needle_min = np.array([1e9, 1e9, 1e9], dtype=float)
    needle_max = -needle_min
    for loc, size in zip(needle.geom_locations, needle.geom_sizes):
        loc = np.array(loc, dtype=float)
        size = np.array(size, dtype=float)
        if size.shape[0] != 3:
            continue
        needle_min = np.minimum(needle_min, loc - size)
        needle_max = np.maximum(needle_max, loc + size)

    # Ring bbox (use ring_* geoms only)
    ring_min = np.array([1e9, 1e9, 1e9], dtype=float)
    ring_max = -ring_min
    ring_sizes = []
    for name, loc, size in zip(
        ring_tripod.geom_names, ring_tripod.geom_locations, ring_tripod.geom_sizes
    ):
        if not str(name).startswith("ring_"):
            continue
        loc = np.array(loc, dtype=float)
        size = np.array(size, dtype=float)
        if size.shape[0] != 3:
            continue
        ring_sizes.append(size)
        ring_min = np.minimum(ring_min, loc - size)
        ring_max = np.maximum(ring_max, loc + size)

    ring_dims = ring_max - ring_min
    ring_unit = np.min(np.array(ring_sizes), axis=0) if ring_sizes else np.zeros(3)
    ring_thickness = float(ring_unit[1] * 2.0)
    ring_inner = np.array(
        [
            ring_dims[0],
            ring_dims[1] - 2 * ring_thickness,
            ring_dims[2] - 2 * ring_thickness,
        ]
    )
    needle_bbox = needle_max - needle_min
    needle_diameter = float(min(needle_bbox[0], needle_bbox[2]))

    return {
        "needle_bbox_m": needle_bbox.tolist(),
        "needle_bbox_mm": (needle_bbox * 1000).tolist(),
        "needle_bbox_min_m": needle_min.tolist(),
        "needle_bbox_max_m": needle_max.tolist(),
        "needle_diameter_mm": needle_diameter * 1000,
        "ring_bbox_m": ring_dims.tolist(),
        "ring_bbox_mm": (ring_dims * 1000).tolist(),
        "ring_bbox_min_m": ring_min.tolist(),
        "ring_bbox_max_m": ring_max.tolist(),
        "ring_inner_hole_mm": (ring_inner * 1000).tolist(),
        "ring_thickness_mm": ring_thickness * 1000,
    }


def _write_csv(payload: dict[str, Any], csv_path: Path) -> None:
    rows: list[dict[str, str]] = []

    def add_row(task: str, component: str, metric: str, value: Any, note: str = "") -> None:
        rows.append(
            {
                "task": task,
                "component": component,
                "metric": metric,
                "value_mm": f"{value:.4f}" if isinstance(value, float) else str(value),
                "note": note,
            }
        )

    square = payload["square"]
    add_row("square", "peg", "full_size_mm", square["peg1_full_size_mm"], "MuJoCo peg1 geom full extents")
    add_row("square", "nut", "outer_bbox_mm", square["square_ring_outer_mm"], "ring-only outer bbox")
    add_row("square", "nut", "inner_hole_mm", square["square_ring_inner_mm"], "ring-only inner clearance")
    add_row("square", "nut", "wall_thickness_mm", square["square_ring_wall_mm"], "approx wall thickness")
    add_row("square", "fit", "default_clearance_mm", square["square_clearance_default_mm"], "inner - peg side")

    spine_target = square["spine_target"]
    add_row("square", "spine_target", "peg_side_mm", spine_target["peg_side_mm"], "target peg side")
    add_row(
        "square",
        "spine_target",
        "peg_length_mm",
        spine_target["peg_length_mm"],
        "scaled peg length",
    )
    add_row(
        "square",
        "spine_target",
        "nut_inner_mm",
        spine_target["nut_inner_mm"],
        "target nut inner side",
    )
    add_row(
        "square",
        "spine_target",
        "clearance_mm",
        spine_target["clearance_mm"],
        "target clearance",
    )
    add_row(
        "square",
        "spine_target",
        "nut_outer_mm_scaled",
        spine_target["nut_outer_mm_scaled"],
        "scaled outer bbox",
    )
    add_row(
        "square",
        "spine_target",
        "nut_thickness_mm_scaled",
        spine_target["nut_thickness_mm_scaled"],
        "scaled thickness",
    )

    threading = payload["threading"]
    add_row("threading", "needle", "bbox_mm", threading["needle_bbox_mm"], "needle overall bbox")
    add_row("threading", "needle", "diameter_mm", threading["needle_diameter_mm"], "approx shaft diameter")
    add_row("threading", "ring", "outer_bbox_mm", threading["ring_bbox_mm"], "ring outer bbox")
    add_row("threading", "ring", "inner_hole_mm", threading["ring_inner_hole_mm"], "inner hole clearance")
    add_row("threading", "ring", "thickness_mm", threading["ring_thickness_mm"], "ring wall thickness")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv, fieldnames=["task", "component", "metric", "value_mm", "note"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/mimicgen_physical_dims.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/mimicgen_physical_dims.csv"),
    )
    args = parser.parse_args()

    payload = {
        "square": _square_assets(),
        "threading": _threading_assets(),
        "notes": {
            "units": "meters (m) and millimeters (mm)",
            "box_size": "MuJoCo box sizes are half-extents",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(payload, args.csv)
    print(f"Wrote dimensions -> {args.out}")
    print(f"Wrote CAD table -> {args.csv}")


if __name__ == "__main__":
    main()
