"""
Script for making plots which summarise the failed models
"""
import dataclasses
import enum
import os
from typing import Any, Optional, Tuple, Union, Dict, Set, List

import seaborn
from matplotlib import rcParams, pyplot

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.analysis.results_analysis import get_all_lodo_runs, get_all_ilodo_runs, get_config_file, PRETTY_NAME


# ----------------
# Small convenient classes
# ----------------
@dataclasses.dataclass(frozen=True)
class HParam:
    key_path: Union[str, Tuple[str, ...]]
    preprocessing: bool  # Indicating if the hyperparameter is in the preprocessing config file or not
    default: Any
    order: Optional[Union[Tuple[str, ...], Tuple[bool, bool]]]


class _ExperimentType(enum.Enum):
    LODO = "lodo"
    ILODO = "ilodo"


# ----------------
# Convenient functions
# ----------------
def _get_filtered_runs(results_dir, experiment_type, successful):
    """Function for getting all successful runs"""
    if successful:
        if experiment_type == _ExperimentType.LODO:
            return get_all_lodo_runs(results_dir=results_dir, successful_only=True)
        elif experiment_type == _ExperimentType.ILODO:
            return get_all_ilodo_runs(results_dir=results_dir, successful_only=True)
        else:
            raise ValueError(f"Unexpected experiment type: {experiment_type}")
    else:
        # Here, we rather summarise the unsuccessful ones
        if experiment_type == _ExperimentType.LODO:
            all_runs = get_all_lodo_runs(results_dir=results_dir, successful_only=False)
            successful_runs = get_all_lodo_runs(results_dir=results_dir, successful_only=True)
        elif experiment_type == _ExperimentType.ILODO:
            all_runs = get_all_ilodo_runs(results_dir=results_dir, successful_only=False)
            successful_runs = get_all_ilodo_runs(results_dir=results_dir, successful_only=True)
        else:
            raise ValueError(f"Unexpected experiment type: {experiment_type}")
        return tuple(run for run in all_runs if run not in set(successful_runs))


# --------------
# Functions for getting hyperparameter values
# --------------
def _get_band_pass_filter(preprocessing_config):
    return INV_FREQUENCY_BANDS[tuple(preprocessing_config["general"]["filtering"])]  # type: ignore


def _get_spatial_method(config):
    method = config["Varied Numbers of Channels"]["name"]
    if method == "RegionBasedPooling":
        return "RBP"
    elif method == "Interpolation":
        return config["Varied Numbers of Channels"]["kwargs"]["method"]
    else:
        raise ValueError(f"Unexpected method for handling varied numbers of channels: {method}")


def _get_cmmn(config):
    # How to extract CMMN depends on the method for handling the different electrode configurations
    if config["Varied Numbers of Channels"]["name"] == "RegionBasedPooling":
        use_cmmn: Set[bool] = set()
        for rbp_design in config["Varied Numbers of Channels"]["kwargs"]["RBPDesigns"].values():
            use_cmmn.add(rbp_design["use_cmmn_layer"])

        if len(use_cmmn) != 1:
            raise ValueError(f"Expected all or none of the RBP layers to use CMMN, but found {use_cmmn}")

        # Return the only element in the set
        return tuple(use_cmmn)[0]

    elif config["Varied Numbers of Channels"]["name"] == "Interpolation":
        return config["DL Architecture"]["CMMN"]["use_cmmn_layer"]
    else:
        raise ValueError(f"Unexpected method for handling different electrode configurations: "
                         f"{config['Varied Numbers of Channels']['name']}")


def _get_domain_adaptation(config):
    # Extracting CMMN
    use_cmmn = _get_cmmn(config)

    # Extract use of domain discriminator
    use_domain_discriminator = False if config["DomainDiscriminator"] is None else True

    # Combine outputs and return
    if use_cmmn and use_domain_discriminator:
        return "CMMN + DD"
    elif use_cmmn:
        return "CMMN"
    elif use_domain_discriminator:
        return "DD"
    else:
        return "Nothing"


def _get_hyperparameter(config, hparam: HParam):
    if hparam.key_path == "band_pass":
        return _get_band_pass_filter(config)
    elif hparam.key_path == "domain_adaptation":
        return _get_domain_adaptation(config)
    elif hparam.key_path == "sampling_freq_multiple":
        return f"{int(config['general']['resample'] // config['general']['filtering'][-1])} * f max"
    elif hparam.key_path == "num_seconds":
        return f"{int(config['general']['num_time_steps'] // config['general']['resample'])}s"
    elif hparam.key_path == "spatial_dimension":
        return _get_spatial_method(config)

    hyperparameter = config
    for key in hparam.key_path:
        try:
            hyperparameter = hyperparameter[key]

            if hyperparameter is None:
                return hparam.default
        except KeyError:
            return hparam.default

    return hyperparameter


def _get_model_hyperparameters(run, hyperparameters: Dict[str, HParam]):
    hyp_values = {}

    # Load the config files
    main_config = get_config_file(results_folder=run, preprocessing=False)
    preprocessing_config = get_config_file(results_folder=run, preprocessing=True)
    for name, param in hyperparameters.items():
        # Get the hyperparameter values
        config = preprocessing_config if param.preprocessing else main_config
        value = _get_hyperparameter(config=config, hparam=param)

        # Add the value
        hyp_values[name] = value

    return hyp_values


# --------------
# Main function for getting an overview
# --------------
def _get_summary(results_dir, runs, hyperparameters):
    configurations: Dict[str, List[Any]] = dict()
    num_runs = len(runs)
    for i, run in enumerate(runs):
        if i % 10 == 0:
            print(f"Run {i + 1}/{num_runs}")

        # Get the hyperparameter values
        hparams = _get_model_hyperparameters(run=os.path.join(results_dir, run), hyperparameters=hyperparameters)

        # Increment the counting dict for each hyperparameter
        for hparam_name, hparam_value in hparams.items():
            if hparam_name not in configurations:
                configurations[hparam_name] = []

            _pretty_value = PRETTY_NAME[hparam_value] if hparam_value in PRETTY_NAME else hparam_value
            configurations[hparam_name].append(_pretty_value)

    return configurations

# --------------
# Constants
# --------------
# Plot cosmetics
STAT = "count"
FIGSIZE = (7, 5)
FONTSIZE = 18
rcParams["legend.fontsize"] = FONTSIZE
rcParams["legend.title_fontsize"] = FONTSIZE


# Mapping to frequency band
INV_FREQUENCY_BANDS = {(1.0, 4.0): "Delta",
                       (4.0, 8.0): "Theta",
                       (8.0, 12.0): "Alpha",
                       (12.0, 30.0): "Beta",
                       (30.0, 45.0): "Gamma",
                       (1.0, 45.0): "All"}


# Hyperparameters to check
HYPERPARAMETERS = {
    "DL Architecture": HParam(key_path=("DL Architecture", "model"), default=NotImplemented, preprocessing=False,
                              order=("InceptionNetwork", "Deep4Net", "ShallowFBCSPNet")),
    "Spatial Dimension Handling": HParam(key_path="spatial_dimension", default=NotImplemented, preprocessing=False,
                                         order=("Spline", "MNE", "RBP")),
    "Band-pass filter": HParam(key_path="band_pass", preprocessing=True, default=NotImplemented,
                               order=("Delta", "Theta", "Alpha", "Beta", "Gamma", "All")),
    "Domain Adaptation": HParam(key_path="domain_adaptation", preprocessing=False, default=NotImplemented,
                                order=("Nothing", "CMMN", "DD", "CMMN + DD")),
    "Sampling frequency": HParam(key_path="sampling_freq_multiple", default=NotImplemented, preprocessing=True,
                                 order=(r"$2 \cdot f_{max}$", r"$4 \cdot f_{max}$", r"$8 \cdot f_{max}$")),
    "Time series length (s)": HParam(key_path="num_seconds", default=NotImplemented, preprocessing=True,
                                     order=("5s", "10s")),
}


def main():
    results_dir = get_results_dir()
    hyperparameters = HYPERPARAMETERS.copy()
    summarise_successful_runs = False

    # ---------------
    # Get summaries
    # ---------------
    # lodo
    lodo_runs = _get_filtered_runs(results_dir=results_dir, experiment_type=_ExperimentType.LODO,
                              successful=summarise_successful_runs)
    lodo_summary = _get_summary(results_dir=results_dir, runs=lodo_runs, hyperparameters=hyperparameters)

    # ilodo
    ilodo_runs = _get_filtered_runs(results_dir=results_dir, experiment_type=_ExperimentType.ILODO,
                                   successful=summarise_successful_runs)
    ilodo_summary = _get_summary(results_dir=results_dir, runs=ilodo_runs, hyperparameters=hyperparameters)

    print(f"LODO summary: {lodo_summary}")
    print(f"iLODO summary: {ilodo_summary}")

    # ---------------
    # Create combined summary
    # ---------------
    # Make summaries
    summary = lodo_summary.copy()
    lodo_run_lengths = set()
    ilodo_run_lengths = set()
    for param_name, param_history in ilodo_summary.items():
        lodo_run_lengths.add(len(lodo_summary[param_name]))
        ilodo_run_lengths.add(len(param_history))

        summary[param_name].extend(param_history)

    # Sanity check
    assert len(lodo_run_lengths) == 1, f"{len(lodo_run_lengths)}"
    assert len(ilodo_run_lengths) == 1, f"{len(ilodo_run_lengths)}"

    # Add experiment type
    num_lodo_runs = tuple(lodo_run_lengths)[0]
    num_ilodo_runs = tuple(ilodo_run_lengths)[0]
    summary["Experiment"] = ["LODO"] * num_lodo_runs + ["iLODO"] * num_ilodo_runs

    # ---------------
    # Plotting
    # ---------------
    for param_name, param_description in hyperparameters.items():
        pyplot.figure(figsize=FIGSIZE)
        seaborn.countplot(data=summary, x=param_name, hue="Experiment", order=param_description.order, stat=STAT,
                          hue_order=("LODO", "iLODO"))

        # Cosmetics
        pyplot.grid(axis="y")
        pyplot.xlabel(param_name, fontsize=FONTSIZE)
        pyplot.ylabel(STAT.capitalize(), fontsize=FONTSIZE)
        pyplot.ylim(0, max(num_lodo_runs, num_ilodo_runs))
        pyplot.tick_params(labelsize=FONTSIZE)
        pyplot.tight_layout()

    pyplot.show()


if __name__ == "__main__":
    main()
