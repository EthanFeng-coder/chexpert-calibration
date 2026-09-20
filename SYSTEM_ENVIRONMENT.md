# Original Compute Environment

This file records the local environment used for the analysis code and GPU-backed model inference during the study. Exact paths and raw datasets are not required for public use and are not included in this repository.

## Analysis conda environment

The analysis scripts in `scripts/` were checked against the local conda environment named `medical_cnn`:

```text
Python 3.9.23
numpy 2.0.2
pandas 2.3.3
scipy 1.13.1
scikit-learn 1.6.1
matplotlib 3.9.4
```

Only these direct runtime dependencies are listed in `requirements.txt` and `environment.yml` because the public scripts currently perform data preparation and calibration/statistical analysis. GPU inference/training dependencies are not required for these public analysis scripts.

## Original Ubuntu/GPU host

The original Linux host used for the GPU-backed work was:

```text
Ubuntu 24.04.4 LTS (Noble Numbat)
Kernel used in the old root: 6.17.0-14-generic
NVIDIA driver package: cuda-drivers-580 580.159.04-1ubuntu1
CUDA toolkit packages installed: 12.8.2 and 13.0.3
CUDA reported by nvidia-smi during the original check: 13.0
GPU 0: NVIDIA GeForce RTX 5090, 32 GB VRAM
GPU 1: NVIDIA GeForce RTX 5090, 32 GB VRAM
```

The working convention for GPU-isolated runs was to reserve physical GPU 0 for display/interactive use and run ML jobs on physical GPU 1, typically with:

```bash
CUDA_VISIBLE_DEVICES=1 python <script>.py
```

## Notes

- The current public repository does not redistribute trained checkpoints, raw CheXpert images, CXR8 images, or local prediction NPZ files.
- If GPU inference code is added later, its separate PyTorch/timm/torchvision requirements should be recorded in an additional optional environment file instead of bloating the minimal analysis environment.
