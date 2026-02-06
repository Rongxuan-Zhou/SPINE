#!/usr/bin/env python
"""Run Phase 2 tuning-free CITO batch on MimicGen HDF5 demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import mujoco
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spine.planning.cito.tuning_free import TuningFreeCITO, TuningFreeCITOConfig

import multiprocessing as mp


def _sorted_demo_keys(keys: Iterable[str]) -> List[str]:
    def sort_key(name: str):
        if name.startswith("demo_"):
            parts = name.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
        return name

    return sorted(list(keys), key=sort_key)


def _read_model_xml(source_hdf5: Path, demo_key: str = "demo_0") -> str:
    with h5py.File(source_hdf5, "r") as f:
        demo = f["data"][demo_key]
        if "model_file" not in demo.attrs:
            raise ValueError(f"{source_hdf5} missing model_file attr on {demo_key}")
        xml = demo.attrs["model_file"]
    if isinstance(xml, bytes):
        return xml.decode("utf-8")
    return str(xml)


def _patch_asset_paths(model_xml: str, assets_root: Path, mimicgen_assets_root: Path) -> str:
    """Replace asset file paths with local assets roots."""
    import re

    root = str(assets_root.resolve())
    mg_root = str(mimicgen_assets_root.resolve())
    token_map = {
        "robosuite/robosuite/models/assets/": root,
        "robosuite/models/assets/": root,
        "mimicgen_environments/mimicgen_envs/models/robosuite/assets/": mg_root,
        "mimicgen_envs/models/robosuite/assets/": mg_root,
    }

    def _rewrite_path(path: str) -> str:
        for token, base in token_map.items():
            if token in path:
                suffix = path.split(token, 1)[1]
                return f"{base}/{suffix}"
        return path

    def _repl(match: re.Match) -> str:
        raw = match.group(1)
        fixed = _rewrite_path(raw)
        return f'file="{fixed}"'

    return re.sub(r'file="([^"]+)"', _repl, model_xml)


def _infer_table_height(model: mujoco.MjModel) -> float:
    table_candidates: List[Tuple[float, str]] = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name:
            continue
        lname = name.lower()
        if "table" not in lname:
            continue
        if any(tag in lname for tag in ["leg", "legs"]):
            continue
        pos = model.geom_pos[gid]
        size = model.geom_size[gid]
        # For box geoms, top surface is pos_z + size_z
        top_z = float(pos[2] + size[2])
        table_candidates.append((top_z, name))
    if not table_candidates:
        # fallback: use 0.0
        return 0.0
    table_candidates.sort(key=lambda x: x[0], reverse=True)
    return table_candidates[0][0]


def _ensure_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        return parent[name]
    return parent.create_group(name)


def _solve_one(
    conn,
    model_xml: str,
    contact_site: str | None,
    contact_sites: list[str],
    contact_geoms: list[str],
    contact_box_geom: str | None,
    contact_box_body: str | None,
    use_table_plane: bool,
    table_height: float,
    cfg_dict: dict,
    q_ref: np.ndarray,
) -> None:
    try:
        cfg = TuningFreeCITOConfig(**cfg_dict)
        solver = TuningFreeCITO(
            model_xml=model_xml,
            contact_site=contact_site,
            contact_sites=contact_sites or None,
            contact_geoms=contact_geoms or None,
            contact_box_geom=contact_box_geom,
            contact_box_body=contact_box_body,
            use_table_plane=use_table_plane,
            table_height=table_height,
            config=cfg,
        )
        result = solver.solve(q_ref)
        conn.send(
            {
                "q_opt": result.q_opt,
                "v_opt": result.v_opt,
                "tau_opt": result.tau_opt,
                "lambda_opt": result.lambda_opt,
                "diagnostics": result.diagnostics,
            }
        )
    except Exception as exc:
        conn.send({"error": str(exc)})
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch tuning-free CITO on HDF5 demos.")
    parser.add_argument("--input-hdf5", type=Path, required=True)
    parser.add_argument("--output-hdf5", type=Path, required=True)
    parser.add_argument("--source-hdf5", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--contact-site", type=str, default=None)
    parser.add_argument(
        "--contact-sites",
        type=str,
        default=None,
        help="Comma-separated site names for multi-point contacts.",
    )
    parser.add_argument(
        "--contact-geoms",
        type=str,
        default=None,
        help="Comma-separated geom names for multi-point contacts.",
    )
    parser.add_argument(
        "--contact-box-geom",
        type=str,
        default=None,
        help="Geom name to use as box/cylinder obstacle for contact constraints.",
    )
    parser.add_argument(
        "--contact-box-body",
        type=str,
        default=None,
        help="Body name whose first geom is used as box/cylinder obstacle.",
    )
    parser.add_argument(
        "--no-table-plane",
        action="store_true",
        help="Disable table plane contact constraint.",
    )
    parser.add_argument("--table-height", type=float, default=None)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--scvx-iters", type=int, default=5)
    parser.add_argument("--penalty-loops", type=int, default=6)
    parser.add_argument("--trust-region", type=float, default=0.1)
    parser.add_argument("--solver", type=str, default="OSQP")
    parser.add_argument("--solver-verbose", action="store_true")
    parser.add_argument("--demo-timeout", type=float, default=None)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    def _split_list(raw: str | None) -> list[str]:
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    contact_sites = _split_list(args.contact_sites)
    contact_geoms = _split_list(args.contact_geoms)

    assets_root = args.assets_root
    if assets_root is None:
        assets_root = Path(__file__).resolve().parents[1] / ".venv" / "lib"
        # best-effort search for robosuite assets
        candidates = list(Path(".venv").rglob("robosuite/models/assets"))
        if candidates:
            assets_root = candidates[0]
        else:
            assets_root = Path("/home/rongxuan_zhou/SPINE/.venv/lib/python3.10/site-packages/robosuite/models/assets")

    base_xml = _read_model_xml(args.source_hdf5)
    mimicgen_assets_root = Path(__file__).resolve().parents[1] / "external" / "mimicgen" / "mimicgen" / "models" / "robosuite" / "assets"
    patched_xml = _patch_asset_paths(base_xml, assets_root, mimicgen_assets_root)
    model = mujoco.MjModel.from_xml_string(patched_xml)
    table_height = args.table_height if args.table_height is not None else _infer_table_height(model)

    cfg = TuningFreeCITOConfig(
        dt=args.dt,
        scvx_iters=args.scvx_iters,
        penalty_loops=args.penalty_loops,
        trust_region=args.trust_region,
        solver=args.solver,
        solver_verbose=args.solver_verbose,
    )
    base_solver = TuningFreeCITO(
        model_xml=patched_xml,
        contact_site=args.contact_site,
        contact_sites=contact_sites or None,
        contact_geoms=contact_geoms or None,
        contact_box_geom=args.contact_box_geom,
        contact_box_body=args.contact_box_body,
        use_table_plane=not args.no_table_plane,
        table_height=table_height,
        config=cfg,
    )
    resolved_contact_site = base_solver.contact_site

    args.output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    with h5py.File(args.input_hdf5, "r") as fin, h5py.File(args.output_hdf5, "a") as fout:
        data_in = fin["data"]
        data_out = _ensure_group(fout, "data")
        demo_keys = _sorted_demo_keys(data_in.keys())
        demo_keys = demo_keys[args.start :]
        if args.limit is not None:
            demo_keys = demo_keys[: args.limit]

        # store metadata
        fout.attrs["source_hdf5"] = str(args.input_hdf5)
        fout.attrs["model_source"] = str(args.source_hdf5)
        fout.attrs["assets_root"] = str(assets_root)
        fout.attrs["table_height"] = float(table_height)
        fout.attrs["contact_site"] = "" if resolved_contact_site is None else resolved_contact_site
        fout.attrs["contact_sites"] = json.dumps(contact_sites, ensure_ascii=True)
        fout.attrs["contact_geoms"] = json.dumps(contact_geoms, ensure_ascii=True)
        fout.attrs["contact_box_geom"] = (
            "" if args.contact_box_geom is None else str(args.contact_box_geom)
        )
        fout.attrs["contact_box_body"] = (
            "" if args.contact_box_body is None else str(args.contact_box_body)
        )
        fout.attrs["use_table_plane"] = int(not args.no_table_plane)
        fout.attrs["config_json"] = json.dumps(cfg.__dict__, indent=2)

        for demo in demo_keys:
            if (demo in data_out) and (not args.overwrite):
                continue
            demo_in = data_in[demo]
            if "obs" not in demo_in or "robot0_joint_pos" not in demo_in["obs"]:
                continue
            q_ref = np.array(demo_in["obs"]["robot0_joint_pos"], dtype=float)
            if args.demo_timeout is None:
                try:
                    result = base_solver.solve(q_ref)
                except Exception as exc:
                    failures.append(f"{demo}: {exc}")
                    if not args.allow_fail:
                        raise
                    continue
                result_payload = {
                    "q_opt": result.q_opt,
                    "v_opt": result.v_opt,
                    "tau_opt": result.tau_opt,
                    "lambda_opt": result.lambda_opt,
                    "diagnostics": result.diagnostics,
                }
            else:
                ctx = mp.get_context("spawn")
                parent_conn, child_conn = ctx.Pipe(duplex=False)
                proc = ctx.Process(
                    target=_solve_one,
                    args=(
                        child_conn,
                        patched_xml,
                        args.contact_site,
                        contact_sites,
                        contact_geoms,
                        args.contact_box_geom,
                        args.contact_box_body,
                        not args.no_table_plane,
                        table_height,
                        cfg.__dict__,
                        q_ref,
                    ),
                )
                proc.start()
                proc.join(timeout=args.demo_timeout)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                    failures.append(f"{demo}: TIMEOUT ({args.demo_timeout}s)")
                    if not args.allow_fail:
                        raise RuntimeError(f"{demo}: TIMEOUT ({args.demo_timeout}s)")
                    continue
                if parent_conn.poll():
                    result_payload = parent_conn.recv()
                else:
                    result_payload = {"error": "No result returned"}
                if "error" in result_payload:
                    failures.append(f"{demo}: {result_payload['error']}")
                    if not args.allow_fail:
                        raise RuntimeError(result_payload["error"])
                    continue

            if demo in data_out:
                del data_out[demo]
            demo_out = data_out.create_group(demo)
            demo_out.create_dataset("q_ref", data=q_ref)
            demo_out.create_dataset("q_opt", data=result_payload["q_opt"])
            demo_out.create_dataset("v_opt", data=result_payload["v_opt"])
            demo_out.create_dataset("tau_opt", data=result_payload["tau_opt"])
            demo_out.create_dataset("lambda_opt", data=result_payload["lambda_opt"])
            demo_out.attrs["success"] = int(bool(demo_in.attrs.get("success", 0)))
            demo_out.attrs["length"] = int(q_ref.shape[0])
            demo_out.attrs["table_height"] = float(table_height)
            demo_out.attrs["contact_site"] = (
                "" if resolved_contact_site is None else resolved_contact_site
            )
            demo_out.attrs["diagnostics"] = json.dumps(result_payload["diagnostics"])

    print(f"✅ Saved CITO outputs to {args.output_hdf5}")
    if failures:
        fail_path = args.output_hdf5.with_suffix(".failures.txt")
        if fail_path.exists():
            existing = fail_path.read_text(encoding="utf-8").strip().splitlines()
        else:
            existing = []
        fail_path.write_text("\n".join(existing + failures), encoding="utf-8")
        print(f"⚠️  Failures: {len(failures)} (see {fail_path})")


if __name__ == "__main__":
    main()
