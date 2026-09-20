# Reproducibility Package Checklist

Use this checklist before public release or archival deposit.

## Repository basics

- [ ] `README.md` explains the paper, repository purpose, and reproduction workflow.
- [ ] `DATA_ACCESSIBILITY.md` explains which data can and cannot be redistributed.
- [ ] `LICENSE` is included for code reuse.
- [ ] `requirements.txt` is included.
- [ ] `environment.yml` is included.
- [ ] No passwords, SSH keys, tokens, private paths, or credentials are committed.

## Code

- [ ] Calibration scripts are included.
- [ ] Evaluation scripts are included.
- [ ] Ensemble evaluation scripts are included.
- [ ] Bootstrap confidence interval scripts are included.
- [ ] Table generation scripts are included when ready for release.
- [ ] Figure generation scripts are included when ready for release.
- [ ] Config files specify seeds, inputs, outputs, and model names.

## Data and derived outputs

- [ ] Calibration outputs are included if permitted.
- [ ] Aggregate result CSVs are included if permitted.
- [ ] Bootstrap CI outputs are included if permitted.
- [ ] Configs and seeds are documented.

## Restricted raw datasets

- [ ] CheXpert raw images are not uploaded unless redistribution is explicitly permitted.
- [ ] CheXpert official access link is provided.
- [ ] External-domain dataset access links are provided.
- [ ] Dataset restrictions are explained clearly.

## Public release

- [ ] GitHub repository is public.
- [ ] Repository is archived to Zenodo, OSF, or Figshare if a DOI is needed.
- [ ] DOI or archive link is added to `README.md` after archival.
