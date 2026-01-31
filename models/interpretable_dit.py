import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class InterpretableAttentionBlock(nn.Module):
    """
    Transformer block that returns attention map for interpretability.
    """
    def __init__(self, d_model, n_head, dropout=0.1):
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

    def forward(self, x, padding_mask=None):
        attn_out, attn_w = self.attn(
            x, x, x, key_padding_mask=padding_mask, need_weights=True
        )
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x, attn_w


class SpineDiT(nn.Module):
    """
    Transformer diffusion backbone for SPINE.
    Tokens order: [Joint, Force, Action_0...Action_T]
    """
    def __init__(
        self,
        action_dim=9,
        joint_dim=9,
        force_dim=1,
        d_model=256,
        n_heads=4,
        n_layers=4,
        horizon=16,
    ):
        super().__init__()
        self.joint_enc = nn.Linear(joint_dim, d_model)
        self.force_enc = nn.Linear(force_dim, d_model)
        self.action_enc = nn.Linear(action_dim, d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, 2 + horizon, d_model))
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
        self.latest_attn = []

    def forward(self, noisy_action, timestep, obs_joint, obs_force):
        """
        noisy_action: (B, T, action_dim)
        timestep: (B,) long
        obs_joint: (B, joint_dim)
        obs_force: (B, force_dim)
        """
        B, T, _ = noisy_action.shape
        t_emb = self.time_mlp(timestep).unsqueeze(1)  # (B,1,d)

        token_j = self.joint_enc(obs_joint).unsqueeze(1)
        token_f = self.force_enc(obs_force).unsqueeze(1)
        token_a = self.action_enc(noisy_action)

        x = torch.cat([token_j, token_f, token_a], dim=1)
        x = x + self.pos_emb[:, : x.shape[1], :] + t_emb

        self.latest_attn = []
        for blk in self.blocks:
            x, attn = blk(x)
            self.latest_attn.append(attn.detach().cpu())

        action_feat = x[:, 2:, :]  # only action tokens
        return self.head(action_feat)

    def get_last_attention(self):
        return self.latest_attn[-1] if self.latest_attn else None
