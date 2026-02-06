"""Minimal tensor_utils stub for MimicGen dataset writing."""

from __future__ import annotations


def list_of_flat_dict_to_dict_of_list(list_dicts):
    """Convert list of flat dicts to dict of lists."""
    out = {}
    for d in list_dicts:
        for k, v in d.items():
            out.setdefault(k, []).append(v)
    return out
