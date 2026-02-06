#!/usr/bin/env python
"""Generate Phase 1 data QA plots for MimicGen outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_style


TASKS = [
    ("coffee", "Coffee"),
    ("square", "Square"),
    ("threading", "Threading"),
]

WRENCH_KEY = "robot0_ee_wrench"
CONTACT_THRESHOLD = 5.0
BASELINE_QUANTILE = 10.0
CONTACT_ALIGN_PRE = 10
CONTACT_ALIGN_POST = 10
OUT_DIR = Path("artifacts/2026-02-04_phase1")


def _sorted_demo_keys(keys: List[str]) -> List[str]:
    def sort_key(name: str):
        if name.startswith("demo_"):
            parts = name.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
        return name
    return sorted(keys, key=sort_key)


def _baseline_correct(force: np.ndarray) -> np.ndarray:
    baseline = float(np.percentile(force, BASELINE_QUANTILE))
    return np.clip(force - baseline, 0.0, None)


def _contact_event_indices(contact: np.ndarray) -> np.ndarray:
    if contact.size == 0:
        return np.array([], dtype=np.int64)
    starts = list(np.where(contact[1:] & ~contact[:-1])[0] + 1)
    if contact[0]:
        starts = [0] + starts
    return np.asarray(starts, dtype=np.int64)


def _contact_aligned_hf(force_adj: np.ndarray, pre: int, post: int) -> List[float]:
    contact = force_adj > CONTACT_THRESHOLD
    indices = _contact_event_indices(contact)
    energies: List[float] = []
    for idx in indices:
        start = max(0, int(idx) - pre)
        end = min(len(force_adj), int(idx) + post + 1)
        if end - start < 2:
            continue
        window = force_adj[start:end]
        diff = np.diff(window)
        energies.append(float(np.mean(diff**2)))
    return energies


def _contact_aligned_hf_per_traj(force_adj: np.ndarray, pre: int, post: int) -> Tuple[float, int]:
    energies = _contact_aligned_hf(force_adj, pre, post)
    if not energies:
        return 0.0, 0
    return float(np.mean(energies)), len(energies)


def _contact_aligned_force_windows(force_adj: np.ndarray, pre: int, post: int) -> List[np.ndarray]:
    contact = force_adj > CONTACT_THRESHOLD
    indices = _contact_event_indices(contact)
    windows: List[np.ndarray] = []
    for idx in indices:
        start = int(idx) - pre
        end = int(idx) + post
        if start < 0 or end >= len(force_adj):
            continue
        windows.append(force_adj[start : end + 1])
    return windows


def _contact_aligned_force_stats(windows: List[np.ndarray]) -> Dict[str, np.ndarray]:
    if not windows:
        return {"count": np.array([0], dtype=np.int64)}
    stacked = np.stack(windows, axis=0)
    return {
        "mean": np.mean(stacked, axis=0),
        "std": np.std(stacked, axis=0),
        "count": np.array([stacked.shape[0]], dtype=np.int64),
    }


def _load_metrics(hdf5_path: Path) -> Dict[str, List[float]]:
    metrics = {
        "length": [],
        "hf_energy": [],
        "contact_ratio": [],
    }
    if not hdf5_path.exists():
        return metrics
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return metrics
        demos = _sorted_demo_keys(list(f["data"].keys()))
        for demo in demos:
            grp = f[f"data/{demo}"]
            obs = grp.get("obs", None)
            if obs is None or WRENCH_KEY not in obs:
                continue
            wrench = np.asarray(obs[WRENCH_KEY], dtype=np.float32)
            if wrench.ndim != 2 or wrench.shape[1] < 3:
                continue
            force = np.linalg.norm(wrench[:, :3], axis=1)
            force_adj = _baseline_correct(force)
            length = int(force.shape[0])
            diff = np.diff(force_adj) if length > 1 else np.array([0.0], dtype=np.float32)
            hf_energy = float(np.mean(diff**2))
            contact_ratio = float(np.mean(force_adj > CONTACT_THRESHOLD))
            metrics["length"].append(length)
            metrics["hf_energy"].append(hf_energy)
            metrics["contact_ratio"].append(contact_ratio)
    return metrics


def _load_contact_aligned_hf_per_traj(hdf5_path: Path, pre: int, post: int) -> Tuple[List[float], Dict[str, int]]:
    per_traj: List[float] = []
    stats = {"demos": 0, "demos_with_events": 0, "event_count": 0}
    if not hdf5_path.exists():
        return per_traj, stats
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return per_traj, stats
        demos = _sorted_demo_keys(list(f["data"].keys()))
        for demo in demos:
            grp = f[f"data/{demo}"]
            obs = grp.get("obs", None)
            if obs is None or WRENCH_KEY not in obs:
                continue
            wrench = np.asarray(obs[WRENCH_KEY], dtype=np.float32)
            if wrench.ndim != 2 or wrench.shape[1] < 3:
                continue
            force = np.linalg.norm(wrench[:, :3], axis=1)
            force_adj = _baseline_correct(force)
            stats["demos"] += 1
            mean_energy, event_count = _contact_aligned_hf_per_traj(force_adj, pre, post)
            if event_count > 0:
                stats["demos_with_events"] += 1
                per_traj.append(mean_energy)
            stats["event_count"] += event_count
    return per_traj, stats


def _load_contact_aligned_force_stats(hdf5_path: Path, pre: int, post: int) -> Dict[str, np.ndarray]:
    windows: List[np.ndarray] = []
    if not hdf5_path.exists():
        return _contact_aligned_force_stats(windows)
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return _contact_aligned_force_stats(windows)
        demos = _sorted_demo_keys(list(f["data"].keys()))
        for demo in demos:
            grp = f[f"data/{demo}"]
            obs = grp.get("obs", None)
            if obs is None or WRENCH_KEY not in obs:
                continue
            wrench = np.asarray(obs[WRENCH_KEY], dtype=np.float32)
            if wrench.ndim != 2 or wrench.shape[1] < 3:
                continue
            force = np.linalg.norm(wrench[:, :3], axis=1)
            force_adj = _baseline_correct(force)
            windows.extend(_contact_aligned_force_windows(force_adj, pre, post))
    return _contact_aligned_force_stats(windows)


def _load_timeseries(hdf5_path: Path) -> Tuple[np.ndarray, str]:
    if not hdf5_path.exists():
        return np.array([]), "missing"
    with h5py.File(hdf5_path, "r") as f:
        demos = _sorted_demo_keys(list(f["data"].keys()))
        if not demos:
            return np.array([]), "empty"
        demo = demos[0]
        obs = f[f"data/{demo}/obs"]
        if WRENCH_KEY not in obs:
            return np.array([]), f"{demo} missing wrench"
        wrench = np.asarray(obs[WRENCH_KEY], dtype=np.float32)
        force = np.linalg.norm(wrench[:, :3], axis=1)
        force_adj = _baseline_correct(force)
        return force_adj, demo


def _boxplot_two_sets(ax, data_a, data_b, labels, title, ylabel):
    positions = []
    box_data = []
    colors = []
    for i, label in enumerate(labels):
        positions.extend([i - 0.18, i + 0.18])
        box_data.extend([data_a[i], data_b[i]])
        colors.extend(["#4C78A8", "#F58518"])
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(
        [bp["boxes"][0], bp["boxes"][1]],
        ["success", "failed"],
        loc="upper right",
        frameon=False,
    )


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    success_counts = []
    failure_counts = []
    labels = [label for _, label in TASKS]

    hf_success = []
    hf_failed = []
    len_success = []
    len_failed = []
    contact_success = []
    contact_failed = []
    aligned_success = []
    aligned_failed = []

    for task, label in TASKS:
        base = Path("data/mimicgen_generated") / task / f"spine_{task}"
        demo_path = base / "demo.hdf5"
        failed_path = base / "demo_failed.hdf5"

        metrics_success = _load_metrics(demo_path)
        metrics_failed = _load_metrics(failed_path)
        aligned_success_vals, aligned_success_stats = _load_contact_aligned_hf_per_traj(
            demo_path, CONTACT_ALIGN_PRE, CONTACT_ALIGN_POST
        )
        aligned_failed_vals, aligned_failed_stats = _load_contact_aligned_hf_per_traj(
            failed_path, CONTACT_ALIGN_PRE, CONTACT_ALIGN_POST
        )
        aligned_success_force_stats = _load_contact_aligned_force_stats(
            demo_path, CONTACT_ALIGN_PRE, CONTACT_ALIGN_POST
        )
        aligned_failed_force_stats = _load_contact_aligned_force_stats(
            failed_path, CONTACT_ALIGN_PRE, CONTACT_ALIGN_POST
        )

        success_counts.append(len(metrics_success["length"]))
        failure_counts.append(len(metrics_failed["length"]))

        hf_success.append(metrics_success["hf_energy"])
        hf_failed.append(metrics_failed["hf_energy"])
        len_success.append(metrics_success["length"])
        len_failed.append(metrics_failed["length"])
        contact_success.append(metrics_success["contact_ratio"])
        contact_failed.append(metrics_failed["contact_ratio"])
        aligned_success.append(aligned_success_vals)
        aligned_failed.append(aligned_failed_vals)

        summary[task] = {
            "success_count": len(metrics_success["length"]),
            "failure_count": len(metrics_failed["length"]),
            "hf_energy_mean_success": float(np.mean(metrics_success["hf_energy"])) if metrics_success["hf_energy"] else 0.0,
            "hf_energy_mean_failed": float(np.mean(metrics_failed["hf_energy"])) if metrics_failed["hf_energy"] else 0.0,
            "length_mean_success": float(np.mean(metrics_success["length"])) if metrics_success["length"] else 0.0,
            "length_mean_failed": float(np.mean(metrics_failed["length"])) if metrics_failed["length"] else 0.0,
            "contact_ratio_mean_success": float(np.mean(metrics_success["contact_ratio"])) if metrics_success["contact_ratio"] else 0.0,
            "contact_ratio_mean_failed": float(np.mean(metrics_failed["contact_ratio"])) if metrics_failed["contact_ratio"] else 0.0,
            "contact_event_count_success": aligned_success_stats["event_count"],
            "contact_event_count_failed": aligned_failed_stats["event_count"],
            "contact_event_demos_success": aligned_success_stats["demos"],
            "contact_event_demos_failed": aligned_failed_stats["demos"],
            "contact_event_demos_with_events_success": aligned_success_stats["demos_with_events"],
            "contact_event_demos_with_events_failed": aligned_failed_stats["demos_with_events"],
            "contact_aligned_hf_per_traj_mean_success": float(np.mean(aligned_success_vals)) if aligned_success_vals else 0.0,
            "contact_aligned_hf_per_traj_mean_failed": float(np.mean(aligned_failed_vals)) if aligned_failed_vals else 0.0,
            "contact_aligned_force_window_count_success": int(aligned_success_force_stats["count"][0]),
            "contact_aligned_force_window_count_failed": int(aligned_failed_force_stats["count"][0]),
        }

    # Plot 1: success/failure counts
    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, success_counts, width, label="success", color="#4C78A8")
    ax.bar(x + width / 2, failure_counts, width, label="failed", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Trajectories")
    ax.set_title("Phase 1: Success vs Failed Counts")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(OUT_DIR / "phase1_success_failure_counts.png", bbox_inches="tight")
    plt.close(fig)

    # Plot 2: HF energy distribution
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    _boxplot_two_sets(
        ax,
        hf_success,
        hf_failed,
        labels,
        "Force High-Frequency Energy (diff^2 mean)",
        "HF energy",
    )
    fig.savefig(OUT_DIR / "phase1_force_hf_energy.png", bbox_inches="tight")
    plt.close(fig)

    # Plot 3: trajectory length distribution
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    _boxplot_two_sets(
        ax,
        len_success,
        len_failed,
        labels,
        "Trajectory Length Distribution",
        "Length (steps)",
    )
    fig.savefig(OUT_DIR / "phase1_traj_length.png", bbox_inches="tight")
    plt.close(fig)

    # Plot 4: contact ratio distribution
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    _boxplot_two_sets(
        ax,
        contact_success,
        contact_failed,
        labels,
        f"Contact Ratio (|F| > {CONTACT_THRESHOLD:.1f}N, baseline p{BASELINE_QUANTILE:.0f})",
        "Contact ratio",
    )
    all_contact = [val for series in (contact_success + contact_failed) for val in series]
    if all_contact:
        low = float(np.percentile(all_contact, 5))
        high = float(np.percentile(all_contact, 95))
        pad = max(0.02, 0.1 * (high - low))
        ymin = max(0.0, low - pad)
        ymax = min(1.0, high + pad)
        if ymax - ymin < 0.15:
            ymin = max(0.0, ymin - 0.1)
            ymax = min(1.0, ymax + 0.1)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0.0, 1.0)
    fig.savefig(OUT_DIR / "phase1_contact_ratio.png", bbox_inches="tight")
    plt.close(fig)

    # Plot 5: contact-aligned HF energy distribution (per-trajectory mean, contact-only)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    _boxplot_two_sets(
        ax,
        aligned_success,
        aligned_failed,
        labels,
        f"Contact-aligned HF Energy per Trajectory (contact-only, ±{CONTACT_ALIGN_PRE}/{CONTACT_ALIGN_POST} steps)",
        "HF energy (mean over events)",
    )
    fig.savefig(OUT_DIR / "phase1_contact_aligned_hf_energy.png", bbox_inches="tight")
    plt.close(fig)

    # Plot 6: contact-aligned mean force time-series
    fig, axes = plt.subplots(len(TASKS), 1, figsize=(7.6, 9.2), sharex=True, constrained_layout=True)
    if len(TASKS) == 1:
        axes = [axes]
    x = np.arange(-CONTACT_ALIGN_PRE, CONTACT_ALIGN_POST + 1)
    for ax, (task, label) in zip(axes, TASKS):
        base = Path("data/mimicgen_generated") / task / f"spine_{task}"
        stats_success = _load_contact_aligned_force_stats(base / "demo.hdf5", CONTACT_ALIGN_PRE, CONTACT_ALIGN_POST)
        stats_failed = _load_contact_aligned_force_stats(base / "demo_failed.hdf5", CONTACT_ALIGN_PRE, CONTACT_ALIGN_POST)
        if "mean" in stats_success:
            ax.plot(x, stats_success["mean"], color="#4C78A8", linewidth=1.6, label="success")
            ax.fill_between(
                x,
                stats_success["mean"] - stats_success["std"],
                stats_success["mean"] + stats_success["std"],
                color="#4C78A8",
                alpha=0.18,
            )
        if "mean" in stats_failed:
            ax.plot(x, stats_failed["mean"], color="#F58518", linewidth=1.6, linestyle="--", label="failed")
            ax.fill_between(
                x,
                stats_failed["mean"] - stats_failed["std"],
                stats_failed["mean"] + stats_failed["std"],
                color="#F58518",
                alpha=0.18,
            )
        ax.axvline(0, color="#444444", linewidth=1.0, linestyle=":", alpha=0.8)
        ax.set_title(f"{label}: Contact-aligned Force")
        ax.set_ylabel("|F| (N)")
        ax.margins(x=0)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.02))
    axes[-1].set_xlabel("Steps from contact onset")
    fig.savefig(OUT_DIR / "phase1_contact_aligned_force_timeseries.png", bbox_inches="tight")
    plt.close(fig)

    # Plot 7: force time-series examples
    fig, axes = plt.subplots(len(TASKS), 1, figsize=(7.8, 9.6), sharex=False, constrained_layout=True)
    if len(TASKS) == 1:
        axes = [axes]
    for ax, (task, label) in zip(axes, TASKS):
        base = Path("data/mimicgen_generated") / task / f"spine_{task}"
        force_success, demo_success = _load_timeseries(base / "demo.hdf5")
        force_failed, demo_failed = _load_timeseries(base / "demo_failed.hdf5")
        ax.plot(force_success, label=f"success ({demo_success})", color="#4C78A8", linewidth=1.6)
        ax.plot(force_failed, label=f"failed ({demo_failed})", color="#F58518", linewidth=1.6, linestyle="--")
        ax.axhline(CONTACT_THRESHOLD, color="#444444", linewidth=1.0, linestyle=":", alpha=0.8)
        ax.set_title(f"{label}: Force Norm (baseline-corrected)")
        ax.set_ylabel("|F| (N)")
        ax.margins(x=0)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.02))
    axes[-1].set_xlabel("Timestep")
    fig.savefig(OUT_DIR / "phase1_force_timeseries_examples.png", bbox_inches="tight")
    plt.close(fig)

    # Save summary json
    summary_path = OUT_DIR / "phase1_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "force_threshold": CONTACT_THRESHOLD,
                "baseline_quantile": BASELINE_QUANTILE,
                "contact_align_pre": CONTACT_ALIGN_PRE,
                "contact_align_post": CONTACT_ALIGN_POST,
                "wrench_key": WRENCH_KEY,
                "tasks": summary,
            },
            f,
            indent=2,
        )

    print(f"Saved plots to {OUT_DIR}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
