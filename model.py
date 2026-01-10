from Processes.Diffusion import LinearScheduler, DiffusionProcess
from networks import SimpleDenoiser
from torch.distributions import Normal

import torch

class Model:
    def __init__(self, diff_timesteps):
        self.diff_timesteps = diff_timesteps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Network - pass diff_timesteps for proper t normalization
        self.model = SimpleDenoiser(1, 256, 8, max_timesteps=self.diff_timesteps).to(self.device)
        linear_scheduler = LinearScheduler(beta_start=2.5e-5, beta_end=0.005, diffusion_timesteps=self.diff_timesteps)
        self.scheduler = linear_scheduler
        self.ddpm = DiffusionProcess(self.scheduler, self.diff_timesteps, self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Network parameter count: {param_count}")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

    def train(self, trainloader, epochs, test_mode_epoch=None):
        self.model.train()
        for epoch in range(1, epochs):
            total_loss = 0
            num_batches = 0
            for batch in trainloader:
                x = batch[0].to(self.device)
                noise = torch.randn_like(x)
                # noise = self._truncated_normal_like(x, lower=0, upper=torch.inf)
                mse_loss  = self.ddpm.loss_fn(self.model, x, noise)
                loss = mse_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
            if test_mode_epoch and epoch % test_mode_epoch == 0:
                yield avg_loss, epoch
                
    def _truncated_normal_like(self, tensor, lower, upper):
        """
        Generate samples from a truncated normal distribution with the same shape as the input tensor.
        lower: lower bound
        upper: upper bound
        """
        # Create standard normal distribution
        normal = Normal(0, 1)
        
        # Get CDF values at boundaries
        lower_cdf = normal.cdf(torch.tensor(lower, device=tensor.device))
        upper_cdf = normal.cdf(torch.tensor(upper, device=tensor.device))
        
        # Sample uniformly between these CDF values
        u = torch.rand_like(tensor) * (upper_cdf - lower_cdf) + lower_cdf
        
        # Apply inverse CDF (icdf)
        samples = normal.icdf(u)
        
        return samples
    

    def DDPM_sample(self, x):
        self.model.eval()
        return self.ddpm.DDPM_sample(self.model, x.to(self.device))
    
    def DDIM_sample(self, x):
        self.model.eval()
        return self.ddpm.DDIM_sample(self.model, x.to(self.device))
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        load_status = self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded status: {load_status}")