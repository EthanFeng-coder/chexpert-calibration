# RSIF 2026-1060 Reproducibility Package

This repository is for the reproducibility materials for the manuscript:

**Patient-Disjoint and External-Domain Calibration of CheXpert Probability Models: A Retrospective Evaluation of Post-Hoc Recalibration, Ensemble Performance, and Dataset Shift**

Manuscript ID: `rsif-2026-1060`  
Journal: *Journal of the Royal Society Interface*

## Purpose

The journal requested that data and code be made freely available at submission. This repository is intended to hold the code, configuration files, derived outputs and documentation needed to verify and reproduce the manuscript's tables and figures.

## What should be included

Recommended repository layout:

```text
README.md
DATA_ACCESSIBILITY.md
REPRODUCIBILITY_PACKAGE_CHECKLIST.md
LICENSE
environment.yml or requirements.txt
configs/
scripts/
src/
results/
```

## Raw data availability

The raw CheXpert chest radiograph images are third-party medical imaging data and should not be redistributed here unless the licence explicitly permits redistribution. CheXpert can be requested from the official Stanford ML Group site:

<https://stanfordmlgroup.github.io/competitions/chexpert/>

If external-domain datasets are used, their official access links and restrictions should be documented in `DATA_ACCESSIBILITY.md`.

## Reproducibility target

A reviewer with authorised access to the source datasets should be able to:

1. Install the environment.
2. Place datasets in the expected folder layout.
3. Run the evaluation/calibration scripts.
4. Regenerate reported tables and figures.
5. Check random seeds, split definitions, and bootstrap confidence interval outputs.

## Suggested commands placeholder

Update this section after adding the real code.

```bash
# Example only — replace with real commands
conda env create -f environment.yml
conda activate chexpert-calibration
python scripts/run_calibration.py --config configs/main.yaml
python scripts/generate_tables.py --results results/ --out tables/
python scripts/generate_figures.py --results results/ --out figures/source_data/
```

## Citation

If this repository is archived on Zenodo/OSF/Figshare, add the DOI here.
