# Data Accessibility Statement

Suggested text for ScholarOne and the manuscript Data Accessibility section.

## Ready-to-paste ScholarOne text

The code, configuration files, evaluation scripts, calibration scripts, configuration files, evaluation scripts, calibration scripts, and derived outputs where permitted are available at: **[INSERT GITHUB/ZENODO/OSF LINK]**.

The raw CheXpert chest radiograph images are third-party medical imaging data and are not redistributed by the authors due to the dataset's access and licensing terms. CheXpert can be requested from the official Stanford ML Group repository: <https://stanfordmlgroup.github.io/competitions/chexpert/>

For reproducibility, the repository/archive provides the random seeds, calibration outputs where permitted, aggregate performance results where permitted, bootstrap confidence interval outputs where permitted, and scripts to regenerate the reported tables and figures from the manuscript data from an authorised local copy of the dataset.

Any external-domain datasets used in this study are available from their original providers subject to their respective access terms, as described in the manuscript. Derived non-identifying aggregate results and analysis code are provided in the repository/archive above.

## What this repository should contain

- Code for post-hoc calibration and evaluation.
- Derived calibration outputs where permitted before publication.
- Bootstrap confidence interval outputs where permitted before publication.
- Random seeds and run configuration notes.
- Environment files.

## What this repository should not contain unless permitted

- Raw CheXpert chest radiograph images.
- Raw third-party external-domain images that cannot be redistributed.
- Identifiable patient data.
- Credentials, API keys, tokens, or private paths.
