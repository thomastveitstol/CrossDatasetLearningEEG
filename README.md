# CrossDatasetLearningEEG

A project for cross-dataset deep learning in EEG. This repository contains the code used for our experiments. The corresponding scientific paper will be referenced once it is published.

![Tests](https://github.com/thomastveitstol/CrossDatasetLearningEEG/actions/workflows/tests.yml/badge.svg)

## Repository Structure
The code is organised as follows:
- **`scripts/`** - All executable scripts. Key directories include:
  - `models/training/` - Includes experiment scripts and configuration files.
  - `visualisations/results/` - Used to generate figures for the paper.
- **`src/`** - Contains core code, including classes and functions for data processing and model implementation.
- **`tests/`** - Contains tests for code in `src/cdl_eeg/`.

## Notes
- The term `ilodo` (inverted leave-one-dataset-out) occasionally appears instead of `lodi` (leave-one-dataset-in). This terminology was updated in the paper, but for consistency and to avoid potential issues, we have retained `ilodo` in parts of the code.
