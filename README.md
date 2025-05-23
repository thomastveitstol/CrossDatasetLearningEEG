# CrossDatasetLearningEEG

![Tests](https://github.com/thomastveitstol/CrossDatasetLearningEEG/actions/workflows/tests.yml/badge.svg)

A project for cross-dataset deep learning in EEG. This repository contains the code used for our experiments. The corresponding paper is available as a preprint.

doi: https://doi.org/10.1101/2025.05.20.655022



## Repository Structure
The code is organised as follows:
- **`scripts/`** - All executable scripts. Key directories include:
  - `models/training/` - Includes experiment scripts and configuration files.
  - `visualisations/results/` - Used to generate figures for the paper.
- **`src/`** - Contains core code, including classes and functions for data processing and model implementation.
- **`tests/`** - Contains tests for code in `src/cdl_eeg/`.

## Notes
- The term `ilodo` (inverted leave-one-dataset-out) occasionally appears instead of `lodi` (leave-one-dataset-in). This terminology was updated in the paper, but for consistency and to avoid potential issues, we have retained `ilodo` in parts of the code.


## Citation
```bibtex
@article{Tveitstoel2025,
	author = {Tveitstøl, Thomas and Tveter, Mats and Hatlestad-Hall, Christoffer and Hammer, Hugo L. and Engemann, Denis and Haraldsen, Ira},
	title = {Assessing the robustness of deep learning based brain age prediction models across multiple EEG datasets},
	elocation-id = {2025.05.20.655022},
	year = {2025},
	doi = {10.1101/2025.05.20.655022},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/05/21/2025.05.20.655022},
	eprint = {https://www.biorxiv.org/content/early/2025/05/21/2025.05.20.655022.full.pdf},
	journal = {bioRxiv}
}
