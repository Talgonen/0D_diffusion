import torch
from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent

class GaussianMixture:
    def __init__(self, k=3, dim=2, amp=1.0, means=None, stds=None, random_state=None):
        self.amp = amp
        self.generator = torch.Generator().manual_seed(random_state) if random_state is not None else None
        
        if means is None:
            self.k = k
            self.dim = dim
            self.means = torch.empty(k, dim).uniform_(-amp, amp, generator=self.generator)
            self.weights = torch.ones(k) / k
        else:
            self.means = torch.tensor(means, dtype=torch.float32) if not isinstance(means, torch.Tensor) else means.float()
            self.k = self.means.shape[0]
            self.dim = dim
            self.weights = torch.ones(self.k) / self.k  # Uniform weights
        
        if stds is None:
            self.stds = torch.ones_like(self.means)
        else:
            self.stds = torch.tensor(stds, dtype=torch.float32) if not isinstance(stds, torch.Tensor) else stds.float()
        
        # Create mixture distribution
        mix = Categorical(self.weights)
        comp = Independent(Normal(self.means, self.stds), 1) if self.means.dim() > 1 else Normal(self.means, self.stds)
        self.gmm = MixtureSameFamily(mix, comp)

    def sample(self, n):
        return self.gmm.sample((n,))