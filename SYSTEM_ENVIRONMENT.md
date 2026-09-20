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

Only these direct runtime dependencies are listed in `requirements.txt` and `environment.yml` because the public scripts currently perform data preparation and calibration/statistical analysis. Use conda for the recommended setup. GPU inference/training dependencies are not required for these public analysis scripts.

## Original Ubuntu/GPU host

The original Linux host used for GPU-backed work was:

```text
Ubuntu 24.04.4 LTS (Noble Numbat)
Kernel used in the old root: 6.17.0-14-generic
NVIDIA driver package: cuda-drivers-580 580.159.04-1ubuntu1
GPU: NVIDIA GeForce RTX 5090, 32 GB VRAM
```

CUDA toolkit packages were installed system-wide on the original Ubuntu host, but the public analysis scripts should be run through the conda environment described above. The minimal public analysis environment does not require CUDA. GPU training/checkpoint code is documented separately in `TRAINING.md`, with optional dependencies in `environment-gpu.yml`.

For GPU-isolated local inference runs, the working convention was:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n medical_cnn python <script>.py
```

## Notes

- The current public repository does not redistribute trained checkpoints, raw CheXpert images, CXR8 images, or local prediction NPZ files.
- GPU training/inference dependencies are kept separate from the minimal analysis environment in `environment-gpu.yml` and `requirements-gpu.txt`.
