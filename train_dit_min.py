#!/usr/bin/env python
"""Minimal DiT training entry for SPINE Phase-3 (RGB + physics inpainting)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.interpretable_dit import SpineDiT
from models.scheduler import DDPMScheduler
from tools.dataloader_spine import SpineH5Dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPINE DiT on one HDF5 dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--force-dim", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint-dim", type=int, default=9)
    parser.add_argument("--action-dim", type=int, default=9)
    parser.add_argument(
        "--use-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rgb-key", type=str, default="agentview_rgb")
    parser.add_argument("--rgb-size", type=int, default=84)
    parser.add_argument(
        "--use-physics-inpainting",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--physics-token-dim", type=int, default=3)
    parser.add_argument("--physics-mask-prob", type=float, default=0.5)
    parser.add_argument("--loss-phys-weight", type=float, default=0.5)
    parser.add_argument("--contact-force-threshold", type=float, default=2.0)
    parser.add_argument("--force-mag-clip", type=float, default=50.0)
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from latest `spine_dit_ep*.pth` in ckpt-dir if present.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_latest_checkpoint(ckpt_dir: Path) -> tuple[Path | None, int]:
    files = sorted(ckpt_dir.glob("spine_dit_ep*.pth"))
    if not files:
        return None, 0
    latest = files[-1]
    stem = latest.stem
    ep = 0
    if "ep" in stem:
        try:
            ep = int(stem.split("ep")[-1])
        except ValueError:
            ep = 0
    return latest, ep


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = SpineH5Dataset(
        h5_path=args.dataset,
        force_dim=args.force_dim,
        horizon=args.horizon,
        use_rgb=args.use_rgb,
        rgb_key=args.rgb_key,
        rgb_size=args.rgb_size,
        use_physics_inpainting=args.use_physics_inpainting,
        physics_token_dim=args.physics_token_dim,
        physics_mask_prob=args.physics_mask_prob,
        contact_force_threshold=args.contact_force_threshold,
        force_mag_clip=args.force_mag_clip,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = SpineDiT(
        action_dim=args.action_dim,
        joint_dim=args.joint_dim,
        force_dim=args.force_dim,
        horizon=args.horizon,
        use_rgb=args.use_rgb,
        use_physics_inpainting=args.use_physics_inpainting,
        physics_token_dim=args.physics_token_dim,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.diffusion_steps,
        device=device,
    )
    mse = nn.MSELoss()

    start_epoch = 0
    if args.resume_latest:
        latest, ep = _find_latest_checkpoint(args.ckpt_dir)
        if latest is not None:
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state)
            start_epoch = int(ep)
            print(f"[resume] {latest} -> start_epoch={start_epoch}", flush=True)

    cfg = {
        "dataset": str(args.dataset),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "diffusion_steps": int(args.diffusion_steps),
        "horizon": int(args.horizon),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "joint_dim": int(args.joint_dim),
        "action_dim": int(args.action_dim),
        "force_dim": int(args.force_dim),
        "use_rgb": bool(args.use_rgb),
        "use_physics_inpainting": bool(args.use_physics_inpainting),
        "physics_token_dim": int(args.physics_token_dim),
        "physics_mask_prob": float(args.physics_mask_prob),
        "loss_phys_weight": float(args.loss_phys_weight),
    }
    print(f"Start training on {args.dataset}", flush=True)
    print(f"Config: {cfg}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_action = 0.0
        total_phys = 0.0
        steps = 0

        for step, batch in enumerate(loader):
            obs_joint = batch["obs_joint"].to(device)
            obs_force = batch["obs_force"].to(device)
            target_actions = batch["target_actions"].to(device)
            obs_rgb = batch["obs_rgb"]
            if obs_rgb is not None:
                obs_rgb = obs_rgb.to(device)
            obs_phys_token = batch["obs_phys_token"].to(device)
            obs_phys_mask = batch["obs_phys_mask"].to(device)

            batch_size = target_actions.shape[0]
            timesteps = torch.randint(
                0,
                args.diffusion_steps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(target_actions)
            noisy_actions = scheduler.add_noise(target_actions, noise, timesteps)

            pred_noise, aux = model(
                noisy_actions,
                timesteps,
                obs_joint,
                obs_force,
                obs_rgb=obs_rgb,
                obs_phys_token=obs_phys_token,
                obs_phys_mask=obs_phys_mask,
                return_aux=True,
            )
            loss_action = mse(pred_noise, noise)

            loss_phys = torch.zeros((), device=device)
            if args.use_physics_inpainting and "phys_pred" in aux:
                mask = (obs_phys_mask > 0.5).float().unsqueeze(-1)
                if torch.sum(mask) > 0:
                    sq = (aux["phys_pred"] - obs_phys_token) ** 2
                    loss_phys = torch.sum(sq * mask) / torch.sum(mask)
            loss = loss_action + float(args.loss_phys_weight) * loss_phys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_action += float(loss_action.item())
            total_phys += float(loss_phys.item())
            steps += 1

            if step % 50 == 0:
                attn = model.get_last_attention()
                if attn is not None:
                    force_idx = 1
                    joint_attn = float(attn.mean(0)[2:, 0].mean().item())
                    force_attn = float(attn.mean(0)[2:, force_idx].mean().item())
                else:
                    joint_attn = 0.0
                    force_attn = 0.0
                print(
                    f"Ep {epoch} Step {step} | "
                    f"Loss {loss.item():.4f} "
                    f"(act {loss_action.item():.4f}, phys {loss_phys.item():.4f}) | "
                    f"Attn(F) {force_attn:.4f} Attn(J) {joint_attn:.4f}",
                    flush=True,
                )

        avg_loss = total_loss / max(steps, 1)
        avg_action = total_action / max(steps, 1)
        avg_phys = total_phys / max(steps, 1)
        print(
            f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f} "
            f"(act {avg_action:.4f}, phys {avg_phys:.4f})",
            flush=True,
        )

        ckpt_path = args.ckpt_dir / f"spine_dit_ep{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
