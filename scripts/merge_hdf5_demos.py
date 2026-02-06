#!/usr/bin/env python
"""Merge demo groups from multiple HDF5 files into a single output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import h5py


def _load_merge_inputs(attrs: h5py.AttributeManager) -> List[str]:
    if "merge_inputs" not in attrs:
        return []
    raw = attrs.get("merge_inputs")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return list(json.loads(raw))
    except Exception:
        return []


def _write_merge_inputs(attrs: h5py.AttributeManager, inputs: List[str]) -> None:
    attrs["merge_inputs"] = json.dumps(inputs, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge HDF5 demo groups.")
    parser.add_argument("--output-hdf5", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = args.output_hdf5
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run. Output:", output_path)
        for inp in args.inputs:
            print("  input:", inp)
        return

    with h5py.File(output_path, "a") as fout:
        data_out = fout.require_group("data")
        merged_inputs = _load_merge_inputs(fout.attrs)
        for inp in args.inputs:
            with h5py.File(inp, "r") as fin:
                if "data" not in fin:
                    continue
                data_in = fin["data"]
                for demo in data_in.keys():
                    if (demo in data_out) and (not args.overwrite):
                        continue
                    if demo in data_out:
                        del data_out[demo]
                    fin.copy(data_in[demo], data_out, name=demo)
            merged_inputs.append(str(inp))

        _write_merge_inputs(fout.attrs, merged_inputs)

    print(f"✅ Merged {len(args.inputs)} files into {output_path}")


if __name__ == "__main__":
    main()
