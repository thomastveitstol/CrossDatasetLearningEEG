"""
The first script for training!

Used for debugging purposes only, and for me to see how I want the pipeline to work from start until end (what works,
what does not)
"""
import os
import warnings

import yaml

from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel
from cdl_eeg.models.region_based_pooling.region_based_pooling import RBPDesign, RBPPoolType


def main():
    # -----------------
    # Load settings (.yml file)
    # -----------------
    config_file = "debug.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_file)
    with open(path, "r") as f:
        configs = yaml.safe_load(f)

    # -----------------
    # Read RBP designs
    # -----------------
    designs_config = configs["RBP Designs"]
    rbp_designs = []
    total_num_regions = 0
    for name, design in designs_config.items():
        rbp_designs.append(
            RBPDesign(pooling_type=RBPPoolType(design["pooling_type"]),
                      pooling_methods=design["pooling_methods"],
                      pooling_methods_kwargs=design["pooling_methods_kwargs"],
                      split_methods=design["split_methods"],
                      split_methods_kwargs=design["split_methods_kwargs"],
                      num_designs=design["num_designs"])
        )

        num_regions = design["pooling_methods_kwargs"]["num_regions"]  # todo: should be specified by split instead
        if isinstance(num_regions, int):
            total_num_regions += num_regions * design["num_designs"]
        else:
            total_num_regions += sum(num_regions) * design["num_designs"]

    # -----------------
    # Make model
    # -----------------
    # Read configuration file
    mts_design = configs["MTS Module"]
    mts_design["kwargs"]["in_channels"] = total_num_regions

    # Define model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        _ = MainRBPModel(mts_module=mts_design["model"], mts_module_kwargs=mts_design["kwargs"],
                         rbp_designs=tuple(rbp_designs))


if __name__ == "__main__":
    main()
