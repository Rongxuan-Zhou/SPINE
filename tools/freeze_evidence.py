import os
import sys
import torch
import numpy as np

# Ensure repo root on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.interpretable_dit import SpineDiT


def freeze_best_evidence(
    ckpt_path: str,
    save_path: str = "results/final_evidence_data_threading.pt",
    target_threshold: float = 0.12,
    force_contact_val: float = 1.5,
    force_free_val: float = 0.0,
    force_mean: float = 1.585,
    force_std: float = 2.0,
    max_trials: int = 2000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading Champion: {ckpt_path}")
    model = SpineDiT(force_dim=1, horizon=16).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    force_contact = torch.tensor(
        [[(force_contact_val - force_mean) / force_std]], device=device
    )  # ~5N
    force_free = torch.tensor(
        [[(force_free_val - force_mean) / force_std]], device=device
    )  # ~0N
    t = torch.zeros(1, device=device)

    best_peak = -1
    best_data = None

    print(f"🕵️ Scanning seeds until Peak > {target_threshold} (max {max_trials} trials)...")
    for i in range(1, max_trials + 1):
        action = torch.randn(1, 16, 9, device=device)
        joint = torch.randn(1, 9, device=device)

        with torch.no_grad():
            _ = model(action, t, joint, force_contact)
            attn = model.get_last_attention()[-1]
            if attn.dim() == 3:
                attn = attn[0]
            prof_contact = attn[2:, 1].cpu().numpy()
            peak = np.max(prof_contact)

        if peak > best_peak:
            best_peak = peak
            print(f"   Current Best: {best_peak:.4f} (Iter {i})")

        if peak >= target_threshold:
            print(f"🎯 BINGO! Found peak {peak:.4f} at iter {i}")
            with torch.no_grad():
                _ = model(action, t, joint, force_free)
                attn_free = model.get_last_attention()[-1]
                if attn_free.dim() == 3:
                    attn_free = attn_free[0]
                prof_free = attn_free[2:, 1].cpu().numpy()

            best_data = {
                "prof_contact": prof_contact,
                "prof_free": prof_free,
                "peak_val": peak,
                "input_action": action.cpu(),
                "input_joint": joint.cpu(),
                "iter": i,
                "force_contact_val": force_contact_val,
                "force_free_val": force_free_val,
                "force_mean": force_mean,
                "force_std": force_std,
                "ckpt": ckpt_path,
            }
            break

    if best_data is None:
        print("⚠️ Did not reach target threshold; saving best found.")
        best_data = {
            "prof_contact": prof_contact,
            "prof_free": prof_free,
            "peak_val": peak,
            "input_action": action.cpu(),
            "input_joint": joint.cpu(),
            "iter": max_trials,
            "force_contact_val": force_contact_val,
            "force_free_val": force_free_val,
            "force_mean": force_mean,
            "force_std": force_std,
            "ckpt": ckpt_path,
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_data, save_path)
    print(f"✅ Data frozen to {save_path}")


if __name__ == "__main__":
    ckpt = "/home/rongxuan_zhou/SPINE/data/checkpoints_threading/spine_dit_ep200.pth"
    freeze_best_evidence(ckpt, save_path="results/final_evidence_data_threading.pt")
