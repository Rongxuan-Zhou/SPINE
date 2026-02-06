#!/usr/bin/env python
"""Build Phase2 training HDF5 datasets from CITO refine outputs.

Outputs per split:
- `*_noforce.hdf5`: RGB + proprio + zero force labels
- `*_simforce.hdf5`: RGB + proprio + simulated force labels
- `*_spine_refine.hdf5`: RGB + proprio + CITO force labels
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np


def _pad_q(q: np.ndarray, pad_dim: int) -> np.ndarray:
    if q.shape[1] == pad_dim:
        return q
    if q.shape[1] > pad_dim:
        return q[:, :pad_dim]
    pad = np.zeros((q.shape[0], pad_dim - q.shape[1]), dtype=q.dtype)
    return np.concatenate([q, pad], axis=1)


def _force_from_sim(force: np.ndarray, mode: str) -> np.ndarray:
    if force.ndim == 1:
        return force
    if mode == "z":
        return force[:, 2]
    if mode == "x":
        return force[:, 0]
    if mode == "y":
        return force[:, 1]
    # default: norm
    return np.linalg.norm(force, axis=1)


def _force_from_lambda(lam: np.ndarray, mode: str) -> np.ndarray:
    if lam.ndim == 1:
        return lam
    if lam.shape[1] == 1:
        return lam[:, 0]
    if mode == "sum":
        return lam.sum(axis=1)
    # default: norm
    return np.linalg.norm(lam, axis=1)


def _pick_rgb(obs: h5py.Group, key: str) -> np.ndarray:
    if key in obs:
        return np.array(obs[key], dtype=np.uint8)
    for fallback in ("agentview_image", "robot0_eye_in_hand_image"):
        if fallback in obs:
            return np.array(obs[fallback], dtype=np.uint8)
    raise KeyError(f"Missing RGB key '{key}' (and fallback keys) in source obs.")


def _physics_tokens_from_scalar(
    force_scalar: np.ndarray,
    contact_force_threshold: float,
    force_mag_clip: float,
) -> np.ndarray:
    force_scalar = np.asarray(force_scalar, dtype=np.float32).reshape(-1)
    fnorm = np.abs(force_scalar)
    contact = (fnorm > float(contact_force_threshold)).astype(np.float32)
    denom = np.log1p(float(force_mag_clip))
    mag = np.log1p(np.clip(fnorm, 0.0, None)) / max(denom, 1e-6)
    mag = np.clip(mag, 0.0, 1.0).astype(np.float32)
    normal_z = np.full_like(contact, 0.5, dtype=np.float32)
    return np.stack([contact, normal_z, mag], axis=1)


def _physics_tokens_from_vec(
    force_vec: np.ndarray,
    contact_force_threshold: float,
    force_mag_clip: float,
) -> np.ndarray:
    force_vec = np.asarray(force_vec, dtype=np.float32)
    if force_vec.ndim == 1:
        return _physics_tokens_from_scalar(
            force_vec,
            contact_force_threshold=contact_force_threshold,
            force_mag_clip=force_mag_clip,
        )
    fxyz = force_vec[:, :3]
    fnorm = np.linalg.norm(fxyz, axis=1)
    contact = (fnorm > float(contact_force_threshold)).astype(np.float32)
    denom = np.log1p(float(force_mag_clip))
    mag = np.log1p(np.clip(fnorm, 0.0, None)) / max(denom, 1e-6)
    mag = np.clip(mag, 0.0, 1.0).astype(np.float32)
    normal_z = np.full_like(contact, 0.5, dtype=np.float32)
    valid = fnorm > 1e-6
    normal_z[valid] = 0.5 * (fxyz[valid, 2] / fnorm[valid] + 1.0)
    normal_z = np.clip(normal_z, 0.0, 1.0).astype(np.float32)
    return np.stack([contact, normal_z, mag], axis=1)


def _write_dataset(
    output_path: Path,
    demos: Iterable[
        Tuple[str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
    ],
    pad_dim: int,
    source_refine: Path,
    source_mimicgen: Path,
    force_mode: str,
    rgb_out_key: str,
    include_rgb: bool,
    spine_rgb_source: str | None,
    contact_force_threshold: float,
    force_mag_clip: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    demos = list(demos)

    with h5py.File(output_path, "w") as h5f:
        data_grp = h5f.create_group("data")
        mask_grp = h5f.create_group("mask")
        mask_train = mask_grp.create_dataset(
            "train", shape=(len(demos),), dtype=h5py.string_dtype()
        )

        h5f.attrs["source_refine"] = str(source_refine)
        h5f.attrs["source_mimicgen"] = str(source_mimicgen)
        h5f.attrs["pad_dim"] = int(pad_dim)
        h5f.attrs["force_mode"] = str(force_mode)
        h5f.attrs["include_rgb"] = bool(include_rgb)
        h5f.attrs["rgb_key"] = str(rgb_out_key)
        h5f.attrs["physics_token_dim"] = 3
        h5f.attrs["contact_force_threshold"] = float(contact_force_threshold)
        h5f.attrs["force_mag_clip"] = float(force_mag_clip)
        if spine_rgb_source is not None:
            h5f.attrs["spine_rgb_source"] = str(spine_rgb_source)

        for idx, (src_key, q, force, rgb, physics_tokens) in enumerate(demos):
            q = _pad_q(q, pad_dim)
            actions = np.concatenate([q[1:], q[-1:]], axis=0)

            demo_key = f"demo_{idx}"
            demo_grp = data_grp.create_group(demo_key)
            demo_grp.attrs["source_demo"] = src_key
            obs_grp = demo_grp.create_group("obs")
            obs_grp.create_dataset("joint_positions", data=q, compression="gzip")
            obs_grp.create_dataset(
                "ee_forces", data=force.reshape(-1, 1), compression="gzip"
            )
            if physics_tokens is None:
                physics_tokens = _physics_tokens_from_scalar(
                    force,
                    contact_force_threshold=contact_force_threshold,
                    force_mag_clip=force_mag_clip,
                )
            obs_grp.create_dataset(
                "physics_tokens",
                data=np.asarray(physics_tokens, dtype=np.float32),
                compression="gzip",
            )
            if include_rgb:
                if rgb is None:
                    raise ValueError(
                        f"RGB expected but missing for demo '{src_key}' in {output_path}"
                    )
                obs_grp.create_dataset(rgb_out_key, data=rgb, compression="gzip")
            demo_grp.create_dataset("actions", data=actions, compression="gzip")
            mask_train[idx] = demo_key

    print(f"✅ Wrote {len(demos)} demos to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2 trainsets.")
    parser.add_argument("--refine-hdf5", type=Path, required=True)
    parser.add_argument("--mimicgen-hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="n50")
    parser.add_argument("--pad-dim", type=int, default=9)
    parser.add_argument("--rgb-key", type=str, default="agentview_image")
    parser.add_argument("--rgb-out-key", type=str, default="agentview_rgb")
    parser.add_argument("--spine-rgb-hdf5", type=Path, default=None)
    parser.add_argument("--spine-rgb-key", type=str, default="agentview_image")
    parser.add_argument(
        "--include-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write RGB observations into output trainsets (default: true).",
    )
    parser.add_argument(
        "--force-mode",
        type=str,
        default="norm",
        choices=["norm", "z", "x", "y", "sum"],
    )
    parser.add_argument("--contact-force-threshold", type=float, default=2.0)
    parser.add_argument("--force-mag-clip", type=float, default=50.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.spine_rgb_hdf5 is not None and not args.spine_rgb_hdf5.exists():
        raise FileNotFoundError(args.spine_rgb_hdf5)

    with h5py.File(args.refine_hdf5, "r") as f_ref, h5py.File(
        args.mimicgen_hdf5, "r"
    ) as f_mg:
        ref_data = f_ref["data"]
        mg_data = f_mg["data"]
        demo_keys = sorted(ref_data.keys())

        sim_demos = []
        noforce_demos = []
        spine_demos = []

        f_spine_rgb = None
        spine_rgb_data = None
        if args.spine_rgb_hdf5 is not None:
            f_spine_rgb = h5py.File(args.spine_rgb_hdf5, "r")
            spine_rgb_data = f_spine_rgb["data"]

        for key in demo_keys:
            if key not in mg_data:
                continue
            mg_demo = mg_data[key]
            q_ref = np.array(mg_demo["obs"]["robot0_joint_pos"], dtype=float)
            sim_force_raw = np.array(mg_demo["obs"]["robot0_ee_force"], dtype=float)
            sim_force = _force_from_sim(sim_force_raw, args.force_mode)
            rgb_mg = _pick_rgb(mg_demo["obs"], args.rgb_key) if args.include_rgb else None

            ref_demo = ref_data[key]
            q_opt = np.array(ref_demo["q_opt"], dtype=float)
            lam = np.array(ref_demo["lambda_opt"], dtype=float)
            spine_force = _force_from_lambda(lam, args.force_mode)
            rgb_spine = rgb_mg
            if args.include_rgb and spine_rgb_data is not None and key in spine_rgb_data:
                rgb_spine = _pick_rgb(spine_rgb_data[key]["obs"], args.spine_rgb_key)

            # Align lengths
            t_sim = min(
                len(q_ref),
                len(sim_force),
                len(rgb_mg) if rgb_mg is not None else len(q_ref),
            )
            t_spine = min(
                len(q_opt),
                len(spine_force),
                len(rgb_spine) if rgb_spine is not None else len(q_opt),
            )

            q_ref = q_ref[:t_sim]
            sim_force = sim_force[:t_sim]
            sim_force_raw = sim_force_raw[:t_sim]
            if rgb_mg is not None:
                rgb_mg = rgb_mg[:t_sim]

            q_opt = q_opt[:t_spine]
            spine_force = spine_force[:t_spine]
            if rgb_spine is not None:
                rgb_spine = rgb_spine[:t_spine]

            sim_tokens = _physics_tokens_from_vec(
                sim_force_raw,
                contact_force_threshold=args.contact_force_threshold,
                force_mag_clip=args.force_mag_clip,
            )
            noforce_force = np.zeros_like(sim_force)
            noforce_tokens = _physics_tokens_from_scalar(
                noforce_force,
                contact_force_threshold=args.contact_force_threshold,
                force_mag_clip=args.force_mag_clip,
            )
            spine_tokens = _physics_tokens_from_scalar(
                spine_force,
                contact_force_threshold=args.contact_force_threshold,
                force_mag_clip=args.force_mag_clip,
            )

            sim_demos.append((key, q_ref, sim_force, rgb_mg, sim_tokens))
            noforce_demos.append((key, q_ref, noforce_force, rgb_mg, noforce_tokens))
            spine_demos.append((key, q_opt, spine_force, rgb_spine, spine_tokens))

        if f_spine_rgb is not None:
            f_spine_rgb.close()

    tag = args.tag
    _write_dataset(
        output_dir / f"{tag}_simforce.hdf5",
        sim_demos,
        args.pad_dim,
        args.refine_hdf5,
        args.mimicgen_hdf5,
        args.force_mode,
        args.rgb_out_key,
        args.include_rgb,
        str(args.spine_rgb_hdf5) if args.spine_rgb_hdf5 is not None else None,
        args.contact_force_threshold,
        args.force_mag_clip,
    )
    _write_dataset(
        output_dir / f"{tag}_noforce.hdf5",
        noforce_demos,
        args.pad_dim,
        args.refine_hdf5,
        args.mimicgen_hdf5,
        args.force_mode,
        args.rgb_out_key,
        args.include_rgb,
        str(args.spine_rgb_hdf5) if args.spine_rgb_hdf5 is not None else None,
        args.contact_force_threshold,
        args.force_mag_clip,
    )
    _write_dataset(
        output_dir / f"{tag}_spine_refine.hdf5",
        spine_demos,
        args.pad_dim,
        args.refine_hdf5,
        args.mimicgen_hdf5,
        args.force_mode,
        args.rgb_out_key,
        args.include_rgb,
        str(args.spine_rgb_hdf5) if args.spine_rgb_hdf5 is not None else None,
        args.contact_force_threshold,
        args.force_mag_clip,
    )


if __name__ == "__main__":
    main()
