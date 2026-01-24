# Diffusion Models Research

A hands-on research project exploring **Denoising Diffusion Probabilistic Models (DDPM)** and **Denoising Diffusion Implicit Models (DDIM)** from the ground up.

## 🎯 Purpose

This project is a personal learning and research initiative to deeply understand diffusion-based generative models by implementing them from scratch. Rather than using high-level libraries, the goal is to build each component manually to gain intuition about:

- The forward noising process and reverse denoising
- Noise scheduling strategies
- The relationship between DDPM and DDIM sampling
- How diffusion models learn to approximate complex distributions

## 🔬 Approach

The project uses **0D Gaussian Mixture Models** as synthetic data, allowing for:
- Easy visualization of learned distributions
- Fast iteration and experimentation
- Clear comparison between original and generated samples
- Study of mode coverage and sample quality

## 📁 Project Structure

```
├── main.py                 # Experiment runner
├── model.py                # Model wrapper (training & sampling)
├── exps.yaml               # Experiment configurations
├── exps_defaults.yaml      # Default hyperparameters
├── dataset/
│   └── gaussian_mixture.py # Synthetic data generator
├── networks/
│   └── SimpleDenoiser.py   # MLP-based noise prediction network
├── Processes/
│   └── Diffusion/
│       ├── diffusion.py    # Core diffusion process (loss, DDPM/DDIM sampling)
│       └── schedulers.py   # Linear & Cosine noise schedules
├── results/                # Saved models & experiment outputs
└── images/                 # Generated visualizations
```

## 🧠 Key Components

### Diffusion Process
- **Forward process**: Gradually adds Gaussian noise over `T` timesteps
- **Reverse process**: Learns to denoise step-by-step using a neural network
- **Loss function**: MSE between predicted and actual noise

### Sampling Methods
| Method | Steps | Stochasticity | Speed |
|--------|-------|---------------|-------|
| DDPM | 4000 | Stochastic | Slow |
| DDIM | 50 | Deterministic | Fast |

### Neural Network
A simple MLP architecture with:
- Learned time embedding (normalized to [0,1])
- 4 hidden layers with GELU activation
- Direct noise prediction

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- PyTorch
- matplotlib, numpy, geomloss

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/diffusion-research.git
cd diffusion-research

# Install dependencies
pip install -e .
```

### Running Experiments
```bash
python main.py
```

Configure experiments in `exps.yaml`:
```yaml
- name: my_experiment
  means: [[-10], [0], [10]]  # Gaussian mixture means
  epochs: 10000
  test_epochs: 100           # Visualization frequency
```

## 📊 Example Results

The model learns to generate samples from a mixture of Gaussians:

- **Left**: Histogram comparing original data (blue) vs. generated samples (orange)
- **Right**: Scatter plot showing the mapping from noise to generated samples

## 🔧 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diff_timesteps` | 4000 | Number of diffusion steps |
| `epochs` | 10000 | Training epochs |
| `train_size` | 4096 | Training samples |
| `batch_size` | 32 | Batch size |
| `lr` | 1e-4 | Learning rate |
| `beta_start` | 2.5e-5 | Initial noise level |
| `beta_end` | 0.005 | Final noise level |

## 📚 References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2020)

## 📝 License

This project is for educational and research purposes.

---

*Built as part of my journey to understand generative AI from first principles.*
