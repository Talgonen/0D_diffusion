import torch

class SimpleDenoiser(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, t_dim=32, max_timesteps=1000):
        super(SimpleDenoiser, self).__init__()
        self.max_timesteps = max_timesteps  # Used to normalize t to [0, 1]
        # self.t_embed = torch.nn.Sequential(
        #     torch.nn.Linear(1, t_dim),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(t_dim, t_dim)
        # )
        self.t_embed = torch.nn.Sequential(
            torch.nn.Linear(1, t_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(t_dim, t_dim)
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim + t_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        # t: (batch,) or (batch, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # Normalize t to [0, 1] range to be scale-invariant
        t = t / self.max_timesteps
        t_emb = self.t_embed(t)
        x_cat = torch.cat([x, t_emb], dim=1)
        return self.net(x_cat)