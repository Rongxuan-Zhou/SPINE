import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class InterpretableAttentionBlock(nn.Module):
    """Transformer block that stores attention maps for interpretability."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(
            x, x, x, key_padding_mask=padding_mask, need_weights=True
        )
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x, attn_w


class SpineDiT(nn.Module):
    """Diffusion Transformer with optional RGB and physics-token inpainting."""

    def __init__(
        self,
        action_dim: int = 9,
        joint_dim: int = 9,
        force_dim: int = 1,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        horizon: int = 16,
        use_rgb: bool = False,
        use_physics_inpainting: bool = False,
        physics_token_dim: int = 3,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.use_rgb = bool(use_rgb)
        self.use_physics_inpainting = bool(use_physics_inpainting)
        self.physics_token_dim = int(physics_token_dim)

        self.joint_enc = nn.Linear(joint_dim, d_model)
        self.force_enc = nn.Linear(force_dim, d_model)
        self.action_enc = nn.Linear(action_dim, d_model)

        self.rgb_enc = None
        if self.use_rgb:
            self.rgb_enc = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, d_model),
            )

        self.physics_enc = None
        self.physics_head = None
        self.physics_mask_token = None
        if self.use_physics_inpainting:
            self.physics_enc = nn.Linear(self.physics_token_dim, d_model)
            self.physics_head = nn.Linear(d_model, self.physics_token_dim)
            self.physics_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        token_count = 2 + self.horizon
        if self.use_rgb:
            token_count += 1
        if self.use_physics_inpainting:
            token_count += 1
        self.action_token_start = token_count - self.horizon

        self.pos_emb = nn.Parameter(torch.zeros(1, token_count, d_model))
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            [InterpretableAttentionBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, action_dim)
        self.latest_attn: list[torch.Tensor] = []

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        obs_joint: torch.Tensor,
        obs_force: torch.Tensor,
        obs_rgb: torch.Tensor | None = None,
        obs_phys_token: torch.Tensor | None = None,
        obs_phys_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Predict noise over action tokens with optional physics auxiliary output."""
        batch_size = noisy_action.shape[0]
        t_emb = self.time_mlp(timestep).unsqueeze(1)

        tokens = [
            self.joint_enc(obs_joint).unsqueeze(1),
            self.force_enc(obs_force).unsqueeze(1),
        ]

        if self.use_rgb:
            if obs_rgb is None:
                raise ValueError("obs_rgb is required when use_rgb=True")
            if obs_rgb.dtype != torch.float32:
                obs_rgb = obs_rgb.float()
            if obs_rgb.max() > 1.0:
                obs_rgb = obs_rgb / 255.0
            assert self.rgb_enc is not None
            rgb_tok = self.rgb_enc(obs_rgb).unsqueeze(1)
            tokens.append(rgb_tok)

        if self.use_physics_inpainting:
            assert self.physics_enc is not None
            assert self.physics_mask_token is not None
            if obs_phys_token is None:
                obs_phys_token = torch.zeros(
                    batch_size,
                    self.physics_token_dim,
                    dtype=obs_joint.dtype,
                    device=obs_joint.device,
                )
            phys_tok = self.physics_enc(obs_phys_token).unsqueeze(1)
            if obs_phys_mask is not None:
                mask = obs_phys_mask.view(batch_size, 1, 1).to(phys_tok.dtype)
                mask_tok = self.physics_mask_token.expand(batch_size, 1, -1)
                phys_tok = (1.0 - mask) * phys_tok + mask * mask_tok
            tokens.append(phys_tok)

        tokens.append(self.action_enc(noisy_action))
        x = torch.cat(tokens, dim=1)
        x = x + self.pos_emb[:, : x.shape[1], :] + t_emb

        self.latest_attn = []
        for blk in self.blocks:
            x, attn = blk(x)
            self.latest_attn.append(attn.detach().cpu())

        action_feat = x[:, self.action_token_start :, :]
        pred_noise = self.head(action_feat)

        if not return_aux:
            return pred_noise

        aux: dict[str, torch.Tensor] = {}
        if self.use_physics_inpainting:
            assert self.physics_head is not None
            phys_index = self.action_token_start - 1
            aux["phys_pred"] = self.physics_head(x[:, phys_index, :])
        return pred_noise, aux

    def get_last_attention(self) -> torch.Tensor | None:
        return self.latest_attn[-1] if self.latest_attn else None
