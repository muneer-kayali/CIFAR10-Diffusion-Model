# CIFAR-10 Diffusion Model

**Denoising Diffusion Probabilistic Model (DDPM) for Image Generation**

## Overview

This project implements a Denoising Diffusion Probabilistic Model (DDPM) for generating 32×32 RGB images trained on the CIFAR-10 dataset. The implementation follows the methodology from the seminal paper ["Denoising Diffusion Probabilistic Models" (Ho et al., 2020)](https://arxiv.org/abs/2006.11239).

🔗 **Repository:** https://github.com/muneer-kayali/CIFAR10-Diffusion-Model

## Features

- 🧠 UNet architecture with attention mechanisms
- ⏱️ Sinusoidal time embeddings for diffusion timestep conditioning
- 📈 Linear beta noise schedule (β₁ = 10⁻⁴ to βₜ = 0.02)
- 🔄 Exponential Moving Average (EMA) for stable generation
- 💾 Checkpoint saving/loading for training resumption
- 🎨 Sample visualization across training epochs
- 📊 FID score evaluation using pytorch-fid
- ⚡ Mixed precision training with gradient accumulation

## Architecture

### UNet Model (~32.6M parameters)

| Component | Channels | Resolution |
|-----------|----------|------------|
| Input | 3 | 32×32 |
| Encoder Block 1 | 96 | 32×32 |
| Encoder Block 2 | 192 | 16×16 |
| Encoder Block 3 | 384 | 8×8 |
| Encoder Block 4 | 384 | 4×4 |
| Bottleneck | 384 | 4×4 |
| Decoder Block 1 | 384 | 8×8 |
| Decoder Block 2 (+ Attention) | 192 | 16×16 |
| Decoder Block 3 | 96 | 32×32 |
| Output | 3 | 32×32 |

### Key Components

- **TimeEmbedding**: Sinusoidal positional encoding for timestep t
- **UNetBlock**: Conv → GroupNorm → SiLU → Dropout with time conditioning
- **Multi-head Attention**: 8-head attention at 16×16 resolution
- **Skip Connections**: Encoder features concatenated in decoder

## Diffusion Process

### Forward Diffusion (Training)
Gradually adds Gaussian noise to images over T=1000 timesteps:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

### Reverse Diffusion (Sampling)
Iteratively denoises from pure noise:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Timesteps (T) | 1000 |
| β_start | 1e-4 |
| β_end | 0.02 |
| Batch Size | 32 |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 128 |
| Learning Rate | 2e-4 |
| Epochs | 600 |
| EMA Decay | 0.9999 |
| Dropout | 0.1 |
| Optimizer | Adam |

## Requirements

**Python 3.10+**

### Core Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `torchvision` | CIFAR-10 dataset & transforms |
| `matplotlib` | Visualization |
| `tqdm` | Progress bars |
| `pynvml` | GPU memory monitoring |
| `pytorch-fid` | FID score calculation |

### Hardware

- **Recommended**: NVIDIA GPU with 6GB+ VRAM
- **Tested on**: GPU with ~5GB VRAM usage during training

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muneer-kayali/CIFAR10-Diffusion-Model
   cd CIFAR10-Diffusion-Model
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision matplotlib tqdm pynvml pytorch-fid
   ```

4. **Download CIFAR-10:**
   The dataset will be downloaded automatically on first run, or set `download=True` in the notebook.

## Usage

1. Open `project1.ipynb` in Jupyter Notebook/Lab

2. Run cells sequentially:
   - Import dependencies and configure device
   - Initialize noise schedule
   - Define UNet architecture
   - Create DiffusionModel wrapper
   - Load CIFAR-10 dataset
   - Train model (saves checkpoints every 2 epochs)
   - Generate samples (saved every 5 epochs)

3. Monitor training:
   - Loss plots saved to `samples_<timestamp>/loss_plot.png`
   - Sample grids saved to `samples_<timestamp>/epoch_<N>.png`

4. Evaluate FID score:
   ```python
   from pytorch_fid import fid_score
   fid_value = fid_score.calculate_fid_given_paths(
       paths=["./samples_<timestamp>/epoch_600_large", "./data/cifar10_png"],
       batch_size=64,
       device="cuda",
       dims=2048
   )
   ```

## File Structure

```
CIFAR10-Diffusion-Model/
├── project1.ipynb              # Main training notebook
├── last_epoch_samples.png      # Final generated samples
├── sample_grid_with_epochs.png # Samples across training
├── samples_<timestamp>/        # Generated samples directory
│   ├── epoch_5.png
│   ├── epoch_10.png
│   ├── ...
│   ├── epoch_600_large/        # Final 5000 samples for FID
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── loss_plot.png
├── checkpoints_<timestamp>/    # Model checkpoints
│   ├── epoch_2.pth
│   ├── epoch_4.pth
│   └── ...
└── data/
    ├── cifar-10-batches-py/    # CIFAR-10 dataset
    └── cifar10_png/            # Real images for FID
```

## Results

- **Training Duration**: ~600 epochs
- **FID Score**: **50.45** (after 600 epochs)
- **VRAM Usage**: ~5GB during training/sampling

### Sample Progression

The model learns to generate increasingly realistic CIFAR-10-like images:
- Early epochs: Noisy, unstructured patterns
- Mid training: Emerging shapes and colors
- Late training: Recognizable objects with CIFAR-10 characteristics

## Training Details

### Loss Function
Mean Squared Error (MSE) between predicted and actual noise:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### Optimizations

- **Mixed Precision Training**: `torch.amp.autocast` for faster computation
- **Gradient Accumulation**: 4 steps to simulate larger batch size
- **Gradient Clipping**: `max_norm=1.0` for stability
- **EMA Model**: Separate model with exponential moving average weights for sampling
- **Checkpointing**: `torch.utils.checkpoint` during sampling to reduce VRAM

## Limitations

- Training requires significant compute time (600 epochs)
- Limited to 32×32 resolution (CIFAR-10 native size)
- FID of ~50 is moderate; further training or architectural changes could improve quality
- Single attention layer at 16×16 resolution only

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

2. Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)

## License

See repository for license information.
