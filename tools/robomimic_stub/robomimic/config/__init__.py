"""Minimal robomimic.config stub."""

from .config import Config


def config_factory(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("robomimic.config.config_factory is not available in the stub.")

