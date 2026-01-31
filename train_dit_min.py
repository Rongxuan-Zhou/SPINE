"""
Diffusion policy training (Transformer-based) for SPINE datasets with W&B logging.
- Uses joint_positions + ee_forces (no vision).
- DDPM training: random timesteps, predict noise.
- Production defaults: dataset=data/spine_threading.hdf5, batch=128, epochs=200.
- Checkpoints saved to /home/rongxuan_zhou/data/checkpoints (per user request).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tools.dataloader_spine import SpineH5Dataset
from models.interpretable_dit import SpineDiT
from models.scheduler import DDPMScheduler


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 配置
    dataset_path = "data/spine_threading.hdf5"
    batch_size = 128
    n_epochs = 200
    n_diffusion_steps = 100
    horizon = 16
    ckpt_dir = "data/checkpoints_threading"
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg = {
        "dataset": dataset_path,
        "batch_size": batch_size,
        "epochs": n_epochs,
        "diffusion_steps": n_diffusion_steps,
        "horizon": horizon,
        "lr": 1e-4,
    }

    # 数据与模型
    ds = SpineH5Dataset(dataset_path, force_dim=1, horizon=horizon, normalize=True)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    model = SpineDiT(force_dim=1, horizon=horizon).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=n_diffusion_steps, device=device
    )
    loss_fn = nn.MSELoss()

    print(f"🚀 Start Training on {dataset_path}")

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for step, (_, obs_joint, obs_force, target_actions) in enumerate(loader):
            obs_joint = obs_joint.to(device)
            obs_force = obs_force.to(device)
            target_actions = target_actions.to(device)  # x_0 (GT actions)
            B = target_actions.shape[0]

            # --- Diffusion forward ---
            timesteps = torch.randint(0, n_diffusion_steps, (B,), device=device).long()
            noise = torch.randn_like(target_actions)
            noisy_actions = noise_scheduler.add_noise(target_actions, noise, timesteps)

            # --- Model predicts noise ---
            pred_noise = model(noisy_actions, timesteps, obs_joint, obs_force)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                with torch.no_grad():
                    attn = model.get_last_attention()
                    if attn is not None:
                        force_attn = attn.mean(0)[2:, 1].mean().item()
                        joint_attn = attn.mean(0)[2:, 0].mean().item()
                        print(
                            f"Ep {epoch} Step {step} | Loss {loss.item():.4f} "
                            f"| Attn(F) {force_attn:.4f} Attn(J) {joint_attn:.4f}"
                        )

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"spine_dit_ep{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
