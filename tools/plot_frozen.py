import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Optional styling (SciencePlots)
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except Exception:
    pass


def plot_beautified(
    data_path="results/final_evidence_data_threading.pt",
    save_path="results/paper_figure_5_final.png",
    y_max=None,
    baseline=0.055,
):
    try:
        data = torch.load(data_path, weights_only=False)
    except FileNotFoundError:
        print("❌ Data file not found! Run freeze_evidence.py first.")
        return

    prof_contact = np.array(data["prof_contact"])
    prof_free = np.array(data["prof_free"])
    peak = float(data["peak_val"])
    fc_val = data.get("force_contact_val", 5.0)
    ff_val = data.get("force_free_val", 0.0)

    # 美化字体
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})

    y_top = y_max if y_max is not None else min(0.40, max(0.16, peak * 1.25))
    print(f"🎨 Plotting frozen data | Peak: {peak:.4f} | y_top: {y_top:.3f}")

    x = np.arange(1, 17)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    # Left: Free Space (F_ext ~ 0 N)
    axes[0].plot(
        x, prof_free, "o-", color="#1f77b4", linewidth=1.8, markersize=5, label="Attention $\\alpha_t$"
    )
    axes[0].axhline(
        y=baseline, color="gray", linestyle=":", alpha=0.6, label="Chance Level (1/18)"
    )
    axes[0].fill_between(x, 0, prof_free, color="#1f77b4", alpha=0.1)
    axes[0].set_title(
        f"Scenario: Free Space\n($F_{{ext}} \\approx {ff_val:.2f}$ N)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylim(0, y_top)
    axes[0].grid(True, alpha=0.2, linestyle="--")
    axes[0].set_ylabel("Attention to Force ($w_{force}$)", fontsize=11)
    axes[0].set_xlabel("Prediction Step ($t$)", fontsize=11)
    axes[0].text(
        8,
        y_top * 0.92,
        f"Avg: {np.mean(prof_free):.4f}",
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
    )
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)

    # Right: Hard Contact (F_ext ~ 5 N)
    axes[1].plot(
        x, prof_contact, "o-", color="#d62728", linewidth=2.3, markersize=6, label="Attention $\\alpha_t$"
    )
    axes[1].axhline(
        y=baseline, color="gray", linestyle=":", alpha=0.6, label="Chance Level (1/18)"
    )
    axes[1].fill_between(
        x,
        baseline,
        prof_contact,
        where=(prof_contact >= baseline),
        color="#d62728",
        alpha=0.2,
        interpolate=True,
    )
    axes[1].set_title(
        f"Scenario: Hard Contact\n($F_{{ext}} \\approx {fc_val:.2f}$ N)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_ylim(0, y_top)
    axes[1].grid(True, alpha=0.2, linestyle="--")
    axes[1].set_xlabel("Prediction Step ($t$)", fontsize=11)
    axes[1].text(
        8,
        y_top * 0.92,
        f"Avg: {np.mean(prof_contact):.4f}",
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
    )
    # Peak annotation -> right side
    max_idx = np.argmax(prof_contact)
    peak_x = max_idx + 1
    peak_y = prof_contact[max_idx]
    axes[1].annotate(
        f"Peak: {peak_y:.3f}",
        xy=(peak_x, peak_y),
        xytext=(peak_x + 1.5, peak_y - 0.005),  # move right/up to avoid legend overlap
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.2, headwidth=5),
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Final Polish saved to {save_path}")


if __name__ == "__main__":
    plot_beautified()
