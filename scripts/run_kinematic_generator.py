#!/usr/bin/env python
"""CLI entrypoint to run the kinematic generator from a YAML config."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spine.perception.kinematics import (
    DexCapAdapter,
    KinematicGenerator,
    KinematicGeneratorConfig,
    MimicGenAdapter,
    R2R2RAdapter,
    load_kinematic_generator_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_kinematic_generator")


def build_adapters(config: KinematicGeneratorConfig):
    adapters = []
    if config.r2r2r:
        adapters.append(R2R2RAdapter(config.r2r2r))
    if config.mimicgen:
        adapters.append(MimicGenAdapter(config.mimicgen))
    if config.dexcap:
        adapters.append(DexCapAdapter(config.dexcap))
    return adapters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run kinematic generators and emit JSON trajectories."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kinematics/kinematic_generator.yaml"),
        help="Path to kinematic generator YAML config.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional hard cap overriding config.max_trajectories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_kinematic_generator_config(args.config)
    if args.max_trajectories is not None:
        cfg.max_trajectories = args.max_trajectories
    adapters = build_adapters(cfg)
    if not adapters:
        logger.warning("未配置任何数据源，直接退出")
        return
    generator = KinematicGenerator(
        output_dir=cfg.output_dir,
        adapters=adapters,
        max_trajectories=cfg.max_trajectories,
    )
    paths: List[Path] = generator.run()
    logger.info("生成完成，保存 %d 条轨迹到 %s", len(paths), cfg.output_dir)


if __name__ == "__main__":
    main()
