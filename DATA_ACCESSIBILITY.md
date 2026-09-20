# Data Accessibility Statement

Suggested text for ScholarOne and the manuscript Data Accessibility section.

## Ready-to-paste ScholarOne text

The code, configuration files, evaluation scripts, calibration scripts, split definitions, derived prediction outputs where permitted, bootstrap confidence interval outputs, experiment logs, and source data for tables and figures are available at: **[INSERT GITHUB/ZENODO/OSF LINK]**.

The raw CheXpert chest radiograph images are third-party medical imaging data and are not redistributed by the authors due to the dataset's access and licensing terms. CheXpert can be requested from the official Stanford ML Group repository: <https://stanfordmlgroup.github.io/competitions/chexpert/>

For reproducibility, the repository/archive provides the patient-disjoint split definitions, random seeds, model prediction files where permitted, calibration outputs, aggregate performance results, bootstrap confidence interval outputs, and scripts to regenerate the reported tables and figures from an authorised local copy of the dataset.

Any external-domain datasets used in this study are available from their original providers subject to their respective access terms, as described in the manuscript. Derived non-identifying aggregate results and analysis code are provided in the repository/archive above.

## What this repository should contain

- Code for post-hoc calibration and evaluation.
- Patient-disjoint split definitions.
- Prediction files where redistribution is permitted.
- Derived calibration outputs.
- Bootstrap confidence interval outputs.
- Table source CSVs.
- Figure source data.
- Run logs and random seeds.
- Environment files.

## What this repository should not contain unless permitted

- Raw CheXpert chest radiograph images.
- Raw third-party external-domain images that cannot be redistributed.
- Identifiable patient data.
- Credentials, API keys, tokens, or private paths.
