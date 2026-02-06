"""Minimal robomimic file_utils stub for MimicGen generation."""

from __future__ import annotations

import json
import os

import h5py

from .env_utils import set_env_specific_obs_processing


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic", set_env_specific_obs_processors=True):
    dataset_path = os.path.expandvars(os.path.expanduser(dataset_path))
    with h5py.File(dataset_path, "r") as f:
        if ds_format == "robomimic":
            env_meta = json.loads(f["data"].attrs["env_args"])
        elif ds_format == "r2d2":
            env_meta = dict(f.attrs)
        else:
            raise ValueError(f"Unsupported dataset format: {ds_format}")
    if set_env_specific_obs_processors:
        set_env_specific_obs_processing(env_meta=env_meta)
    return env_meta


def url_is_alive(url: str) -> bool:
    """Conservative stub for url liveness check."""
    # Avoid network checks in stub mode; assume unreachable so callers can handle locally.
    return False
