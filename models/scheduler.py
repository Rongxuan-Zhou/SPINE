import torch


class DDPMScheduler:
    """Simple linear-beta DDPM scheduler."""

    def __init__(self, num_train_timesteps=100, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.n_steps = num_train_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        """
        Forward noising: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
        original_samples: (B, T, D)
        noise: same shape
        timesteps: (B,) long
        """
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps]).flatten().view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alphas_cumprod[timesteps]).flatten().view(-1, 1, 1)
        noisy = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy
