# CheXpert Calibration

Public reproducibility repository for the paper:

**Patient-Disjoint and External-Domain Calibration of CheXpert Probability Models: A Retrospective Evaluation of Post-Hoc Recalibration, Ensemble Performance, and Dataset Shift**

## Purpose

This repository provides the public code scaffold, environment files, data-access notes, and reproducibility documentation for the study. It is intended to support transparent review and later full reproduction of the reported calibration, ensemble, and dataset-shift analyses.

## Repository layout

```text
README.md
DATA_ACCESSIBILITY.md
REPRODUCIBILITY_PACKAGE_CHECKLIST.md
LICENSE
environment.yml
requirements.txt
configs/
scripts/
src/
results/
```

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate chexpert-calibration
```

Or use pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Raw data availability

The raw CheXpert chest radiograph images are third-party medical imaging data and are not redistributed in this repository. CheXpert can be requested from the official Stanford ML Group site:

<https://stanfordmlgroup.github.io/competitions/chexpert/>

If external-domain datasets are used, their official access links and restrictions should be documented in `DATA_ACCESSIBILITY.md`.

## Reproducibility target

A reviewer or reader with authorised access to the source datasets should be able to:

1. Install the environment.
2. Place datasets or derived permitted inputs in the expected folder layout.
3. Run the evaluation and calibration scripts once added.
4. Regenerate reported tables and figures from authorised data and permitted derived outputs.
5. Check random seeds, configuration files, and bootstrap confidence interval settings.

## Commands placeholder

Update this section after adding the real scripts.

```bash
conda env create -f environment.yml
conda activate chexpert-calibration
python scripts/run_calibration.py --config configs/main.yaml
```

## Citation

If this repository is archived on Zenodo, OSF, or Figshare, add the DOI here.
