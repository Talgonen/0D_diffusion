import torch

class LinearScheduler:
    def __init__(self, beta_start, beta_end, diffusion_timesteps):
        self.betas = torch.linspace(beta_start, beta_end, diffusion_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

class CosineScheduler:
    def __init__(self, diffusion_timesteps, s=0.008):
        self.diffusion_steps = diffusion_timesteps
        self.s = s
        self.betas = self.cosine_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def cosine_beta_schedule(self):
        timesteps = self.diffusion_steps
        s = self.s
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)