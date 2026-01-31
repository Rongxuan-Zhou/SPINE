#!/usr/bin/env python
"""Run a lightweight CITO/VSCM projection on a kinematic trajectory JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from spine.planning.cito import CITOParameters, project_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project kinematic trajectory with CITO/VSCM penalty."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input kinematic JSON (from run_kinematic_generator.py)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSON path after projection"
    )
    parser.add_argument(
        "--table-height",
        type=float,
        default=0.0,
        help="Table height (m) used for penetration penalty",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Timestep (s) for simple point-mass dynamics",
    )
    parser.add_argument(
        "--max-iters", type=int, default=5, help="Optimization iterations"
    )
    parser.add_argument(
        "--step-size", type=float, default=0.05, help="Gradient step size"
    )
    parser.add_argument(
        "--trust-region",
        type=float,
        default=0.05,
        help="Max update magnitude per step (m)",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    params = CITOParameters(
        max_iters=args.max_iters,
        step_size=args.step_size,
        trust_region=args.trust_region,
    )
    output_path = project_trajectory(
        input_path=args.input,
        output_path=args.output,
        table_height=args.table_height,
        dt=args.dt,
        params=params,
    )
    print(f"Projected trajectory saved to {output_path}")


if __name__ == "__main__":
    main()
