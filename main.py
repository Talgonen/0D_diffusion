from torch.utils.data import DataLoader, TensorDataset
from dataset.gaussian_mixture import GaussianMixture
from model import Model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np

from pathlib import Path
import yaml

class RunExp:
    def __init__(self, **config):
        self.k = config.get("k")
        self.dim = config.get("dim")
        self.epochs = config.get("epochs")
        self.train_size = config.get("train_size")
        self.test_size = config.get("test_size")
        self.batch_size = config.get("batch_size")
        self.diff_timesteps = config.get("diff_timesteps")
        self.results_dir = Path(f"./results/{config.get('name')}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.gaussian_mixture = GaussianMixture(k=self.k, dim=self.dim, amp=10.0, means=config.get("means"))
        self.sample_mixture = GaussianMixture(means=[[0]], dim=1
                                              )
        self.model = Model(self.diff_timesteps)
        if config.get("load_model"):
            self.model.load(Path(config.get("load_model")))

    def __call__(self, test_epochs):
        trainset = self.gaussian_mixture.sample(self.train_size)
        trainloader = DataLoader(TensorDataset(trainset), batch_size=self.batch_size, shuffle=True)

        # Train
        for (train_loss, epoch) in self.model.train(trainloader, self.epochs, test_epochs):
            self.model.save(self.results_dir / "model.pt")
            print(f"Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.4f}")
            origin = self.sample_mixture.sample(self.test_size)
            samples = self.model.DDIM_sample(origin).cpu().numpy()

            # Plot histogram of original data and generated samples
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram subplot
            ax1.hist(trainset, bins=1000, density=True, alpha=0.5, label='Original Data') 
            ax1.hist(samples, bins=1000, density=True, alpha=0.5, label='Generated Samples')
            ax1.legend()
            ax1.set_title(f"Comparison of Original Data and Samples (Epoch {epoch})")
            ax1.set_xlabel("Value")
            ax1.set_ylabel("Frequency")
            
            # Scatterplot subplot
            ax2.scatter(origin.cpu().numpy(), samples, s=1)
            ax2.set_title(f"Samples vs Origin (Epoch {epoch})")
            ax2.set_xlabel("Origin (Noise)")
            ax2.set_ylabel("Data (Cleaned Sample)")
            
            fig.tight_layout()
            fig.savefig(self.results_dir / f"epoch_{epoch}.png")
            plt.close(fig)


def main():
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    defult_config = yaml.load(Path("./exps_defaults.yaml").read_text(), Loader=yaml.FullLoader)

    # Set up experiments parameters
    exp_configs = yaml.load(Path("./exps.yaml").read_text(), Loader=yaml.FullLoader)
    assert isinstance(exp_configs, list), "Experiment configurations should be a list."
    for exp in exp_configs:
        assert isinstance(exp, dict), "Each experiment configuration must be a dictionary."
        assert 'name' in exp, "Each experiment must have a name ('name' field)."
    for _config in exp_configs:
        config = defult_config.copy()
        for k,v in _config.items():
            config[k] = v
        RunExp(**config)(config.get("test_epochs", 1000))

if __name__ == "__main__":
    main()
