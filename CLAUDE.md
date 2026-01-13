# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**emnet** is a neuroscience-inspired computational model for eye movement-conditioned image reconstruction. It simulates how the visual system processes information during natural eye movements (fixations and saccades) over images.

The project is organized as a Python 3.13 package using modern tooling (uv package manager, PyTorch, NumPy).

## Technology Stack

- **PyTorch (>=2.9.1)**: Core deep learning framework
- **NumPy (>=2.4.0)**: Numerical computing for data processing
- **Matplotlib (>=3.10.8)**: Visualization
- **tqdm (>=4.67.1)**: Progress bars
- **Python 3.13+**: Required Python version

Build system: setuptools with uv package manager

## Architecture Overview

The project implements a three-stage neural processing pipeline:

```
Input (video frames at eye positions)
    ↓
RetinalEncoder (spatiotemporal RGC-like filtering)
    ↓
V1Decoder (spatial mixing + temporal convolution)
    ↓
FrameDecoder (linear projection to image space)
    ↓
Output (reconstructed image patches)
```

### Key Components

**Data Generation** (`src/data/`):
- **ReconDataset**: Generates synthetic eye movement sequences over pink noise images
  - Returns: (video_frames, target_img, eye_trace, visibility_mask, saccade_end_idx)
  - Parameters: img_size (256), roi_size (32), total_samples (128), sampling_frequency (360 Hz)
  - Generates 1 million virtual samples with caching of 500 pink noise base images
- **VideoDataset**: Loads pre-recorded video data from .npy files with memory mapping

**Model Layers** (`src/models/`):
- **RetinalEncoder**: Applies spatial then temporal convolution with ReLU
  - Input: (batch, roi_size, roi_size, time)
  - Output: (batch, n_channels, reduced_time)
- **V1Decoder**: Spatial mixing matrix + temporal convolution
  - Input: (batch, rgc_n_channels, time)
  - Output: (batch, v1_n_channels, reduced_time)
- **FrameDecoder**: Linear projection from V1 space to image space
  - Input: (batch, v1_n_channels, time)
  - Output: (batch, time, roi_size²)

**Utilities** (`src/utils/`):
- **eye_movement.py**: `generate_brownian_motion()`, `generate_saccade()` - eye trace simulation
- **image_generation.py**: `pink_noise_gray_image()` - generates 1/f noise images
- **reconstruction.py**: `stitch_frames_by_position()` - reconstructs images from frame positions

### Biological Constraints

Two constraints are implemented (partially - see TODO items):

1. **Filter Energy Constraint**: Temporal kernels normalized to unit L2 norm
   - Applied in main.py after training updates
   - Prevents weight magnitude scaling

2. **Firing Rate Constraint**: Augmented Lagrangian Method (ALM) for metabolic budget
   - Target: average firing rate = 1.0
   - Not yet fully implemented

## Common Development Commands

```bash
# Install dependencies
uv sync

# Run the main forward pass example
python main.py

# Generate a dataset sample
python -c "from src.data.recon_dataset import ReconDataset; ds = ReconDataset(); sample = ds[0]"

# Import and test a single model
python -c "from src.models.retinal_encoder import RetinalEncoder; import torch; enc = RetinalEncoder(100, 32, 12); x = torch.randn(8, 32, 32, 12); y = enc(x); print(y.shape)"

# Run a single utility function
python -c "from src.utils.eye_movement import generate_brownian_motion; import numpy as np; trace = generate_brownian_motion(20/3600, 360, 128); print(trace.shape)"
```

## Project Structure Details

```
src/
├── data/              # Data loading and synthesis
│   ├── recon_dataset.py    # Main dataset with eye movement simulation
│   └── video_dataset.py    # Pre-recorded video loader
├── models/            # Neural network layers
│   ├── retinal_encoder.py  # RGC-like spatiotemporal filtering
│   ├── v1_decoder.py       # V1 representation layer
│   └── frame_decoder.py    # Final reconstruction layer
├── training/          # Training infrastructure
│   └── trainer.py     # Training loop (currently empty)
├── utils/             # Standalone utility functions
│   ├── eye_movement.py      # Brownian motion, saccade generation
│   ├── image_generation.py  # Pink noise synthesis
│   └── reconstruction.py    # Image stitching
└── losses/            # (empty) Loss function definitions
```

## Main Entry Point

`main.py` demonstrates the complete forward pass:
1. Creates a ReconDataset with specific parameters (img_size=128, roi_w=10, etc.)
2. Instantiates the three model layers
3. Runs a forward pass on a batch
4. Applies unit norm constraint to temporal kernels
5. **TODO**: Implement image stitching to reconstruct the full image from patches

## Important Implementation Notes

### ReconDataset Parameters

- `img_size`: Size of the full image (default 256)
- `roi_size`: Size of region-of-interest patches extracted at eye positions (default 32)
- `total_samples`: Number of frames in each sequence (default 128)
- `pad_start`: Number of frames before mask activation (default 32)
- `diffusion_coefficient`: Brownian motion diffusion (default 20/3600 deg²/s)
- `sampling_frequency`: Hz (default 360)
- `pixels_per_degree`: Visual angle scaling (default 240)
- `saccade`: Bool to include saccadic movements (default False)
- `average`: Bool to average visited regions instead of returning original image (default False)

### Model Tensor Shapes

- **RetinalEncoder input**: (batch, roi_size, roi_size, time) → (batch, n_channels, reduced_time)
  - Temporal kernel length creates valid convolution reduction
- **V1Decoder input**: (batch, rgc_channels, time) → (batch, v1_channels, reduced_time)
- **FrameDecoder input**: (batch, v1_channels, time) → (batch, time, roi_size²)

### Import Paths

Use absolute imports from the project root:
```python
from src.data.recon_dataset import ReconDataset
from src.models.retinal_encoder import RetinalEncoder
from src.utils.eye_movement import generate_brownian_motion
```

Or from within src/:
```python
from utils.eye_movement import generate_brownian_motion
from data.recon_dataset import ReconDataset
```

## Active Development Areas

**TODO items found in codebase**:
- Image stitching reconstruction (main.py:43): Complete implementation of `stitch_frames_by_position()` integration
- Loss functions: Define objective functions in `src/losses/`
- Training loop: Implement full training in `src/training/trainer.py`
- Constraint implementation: Complete ALM for firing rate constraint
- Tests: No test suite exists - would be valuable to add

## Key Design Patterns

1. **Dataset Caching**: ReconDataset generates pink noise images on-the-fly and caches 500 of them for reuse across the 1M virtual samples

2. **Spatiotemporal Filtering**: Both encoder and decoder apply spatial convolution first, then temporal convolution for efficiency

3. **Valid Convolution**: Temporal convolutions use `valid` padding, reducing sequence length by (kernel_length - 1)

4. **Memory Efficiency**: VideoDataset uses memory mapping (`np.load(..., mmap_mode='r')`) for large video files

## Dependencies and Imports

All external dependencies are in `pyproject.toml`. Key modules:
- `torch`: All models inherit from `torch.nn.Module`
- `torch.utils.data.Dataset`: Base class for datasets
- `numpy`: Used for array operations and image generation
- `tqdm`: Progress bars (used in ReconDataset initialization)

## Configuration Files

- `pyproject.toml`: Project metadata, dependencies, Python version requirement
- `.python-version`: Specifies Python 3.13
- `.gitignore`: Standard Python ignores
- `uv.lock`: Dependency lock file (generated by uv package manager)

## Notes for Future Development

1. **Constraint Implementation**: The unit norm constraint is manually applied in main.py. Consider automating this (hooks, custom optimizer, etc.)

2. **Firing Rate Constraint**: The ALM algorithm needs to be implemented. This involves dual-ascent optimization with learnable Lagrange multipliers.

3. **Loss Functions**: No loss functions defined yet. Likely candidates:
   - Mean squared error (MSE) between reconstructed and target images
   - Perceptual loss for image similarity
   - Regularization on constraint violations

4. **Training Framework**: `trainer.py` is currently empty. Should implement:
   - Batched training loop
   - Constraint enforcement per iteration
   - Validation on held-out data

5. **Testing**: No automated tests. Would be valuable for:
   - Model output shape validation
   - Dataset generation correctness
   - Constraint verification
   - Utility function edge cases

6. **Documentation**: Consider adding docstrings to model forward methods documenting exact tensor shapes and transformations
