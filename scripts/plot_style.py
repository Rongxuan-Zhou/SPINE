"""Shared plotting style for SPINE figures."""

from __future__ import annotations

import os
from pathlib import Path

_mpl_cache = Path("artifacts/.mplconfig").resolve()
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt  # noqa: E402
try:  # noqa: E402
    import scienceplots  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    # Keep plots functional in minimal environments; aesthetics are optional.
    scienceplots = None  # type: ignore[assignment]


SCIENCE_IEEE_STYLE = ["science", "ieee"]


def apply_style() -> None:
    """Apply the default Science + IEEE matplotlib style."""
    if scienceplots is not None:
        plt.style.use(SCIENCE_IEEE_STYLE)
    else:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": False,
        }
    )
