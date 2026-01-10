import torch.nn.functional as F
import torch
from geomloss import SamplesLoss

class DiffusionProcess:
    def __init__(self, scheduler, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device

        self.scheduler = scheduler
        self.betas = self.scheduler.betas.to(self.device)
        self.alphas = self.scheduler.alphas.to(self.device)
        self.alpha_bars = self.scheduler.alphas_cumprod.to(self.device)

    def loss_fn(self, model, x0, noise):
        """
        x0: [batch, channels, data] - original data
        """
        batch_size = x0.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        pred_noise = model(xt, t.to(torch.float32))
        mse_loss = F.mse_loss(pred_noise, noise)
        return mse_loss

    @torch.no_grad()
    def DDPM_sample(self, model, x):
        """
        shape: (batch, channels, data)
        """
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            pred_noise = model(x, t_batch.to(torch.float32))
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1 - alpha_bar)
            x = coef1 * (x - coef2 * pred_noise)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x += sigma * noise
        return x
    
    @torch.no_grad()
    def DDIM_sample(self, model, x, ddim_steps=50, eta=0.0):
        """
        shape: (batch, channels, data)
        eta: controls the amount of noise added during sampling (0.0 = deterministic, 1.0 = stochastic)
        """
        timesteps = list(range(0, self.timesteps, self.timesteps // ddim_steps))
        timesteps_next = list(timesteps[:-1])
        for t, t_next in zip(reversed(timesteps), reversed(timesteps_next)):
            t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
            alpha_bar = self.alpha_bars[t]
            if t > 0:
                alpha_bar_next = self.alpha_bars[t_next]
            else:
                alpha_bar_next = torch.tensor(1.0, device=self.device)

            pred_noise = model(x, t_batch.to(torch.float32))
            x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

            dir_xt = torch.sqrt(1 - alpha_bar_next) * pred_noise
            x_next = torch.sqrt(alpha_bar_next) * x0_pred + dir_xt

            if t > 0:
                sigma_t = eta * torch.sqrt((1 - alpha_bar / alpha_bar_next) * (1 - alpha_bar_next) / (1 - alpha_bar))
                noise = torch.randn_like(x)
                x_next += sigma_t * noise

            x = x_next
        return x