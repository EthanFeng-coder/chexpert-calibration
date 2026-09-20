# Reproducibility Package Checklist

Use this before resubmitting `rsif-2026-1060`.

## Repository basics

- [ ] `README.md` explains the manuscript, repository purpose, and reproduction workflow.
- [ ] `DATA_ACCESSIBILITY.md` contains ready-to-paste ScholarOne text.
- [ ] `LICENSE` is added for code reuse.
- [ ] `requirements.txt` or `environment.yml` is added.
- [ ] No passwords, SSH keys, tokens, or private paths are committed.

## Code

- [ ] Calibration scripts are included.
- [ ] Evaluation scripts are included.
- [ ] Ensemble evaluation scripts are included.
- [ ] Bootstrap confidence interval scripts are included.
- [ ] Table generation scripts are included.
- [ ] Figure generation scripts are included.
- [ ] Config files specify seeds, inputs, outputs, and model names.

## Data and derived outputs

- [ ] Patient-disjoint split definitions are included.
- [ ] Prediction CSVs are included where permitted.
- [ ] Calibration outputs are included.
- [ ] Aggregate result CSVs are included.
- [ ] Bootstrap CI outputs are included.
- [ ] Table source data are included.
- [ ] Figure source data are included.
- [ ] Logs/configs/seeds are included.

## Restricted raw datasets

- [ ] CheXpert raw images are not uploaded unless redistribution is explicitly permitted.
- [ ] CheXpert official access link is provided.
- [ ] External-domain dataset access links are provided.
- [ ] Any dataset restrictions are explained clearly.

## Submission

- [ ] Public GitHub repository is created, or private review link is prepared.
- [ ] Repository is archived to Zenodo/OSF/Figshare if possible.
- [ ] DOI or archive link is inserted into ScholarOne.
- [ ] Data Accessibility text is inserted into ScholarOne.
- [ ] Same statement is added to the manuscript Data Accessibility section.
