"""
Script for generating distributions of top configurations. It is not used in the paper, but I used it for finding the
best models.

Best models:
TDBrain: age_inverted_cv_experiments_2024-06-21_082009/leave_one_dataset_out/Fold_1
HatlestadHall: age_inverted_cv_experiments_2024-06-24_224405/leave_one_dataset_out/Fold_4
MPILemon: age_inverted_cv_experiments_2024-06-30_192130/leave_one_dataset_out/Fold_2
"""
import dataclasses
import os
import warnings
from typing import Any, Tuple, Union, Dict, List, Set, Optional

import numpy
import pandas
import seaborn
from matplotlib import pyplot, rcParams
from matplotlib.ticker import MultipleLocator

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.analysis.results_analysis import SkipFold, is_better, higher_is_better, get_ilodo_val_dataset_name, \
    get_all_ilodo_runs, get_config_file, PRETTY_NAME


# ----------------
# Small convenient classes
# ----------------
@dataclasses.dataclass(frozen=True)
class HParam:
    key_path: Union[str, Tuple[str, ...]]
    preprocessing: bool  # Indicating if the hyperparameter is in the preprocessing config file or not
    default: Any
    order: Optional[Union[Tuple[str, ...], Tuple[bool, bool]]]


@dataclasses.dataclass(frozen=True)
class _Run:
    run: str  # The folder name of the run
    test_performance: float  # The test set performance
    metric: str

    def __lt__(self, other):
        # Input checks
        if not isinstance(other, type(self)):
            raise TypeError(f"Less than operator only implemented for rhs and lhs of type {type(self)}")
        if other.metric != self.metric:
            raise ValueError("Can only compare two runs with the same metric")

        # Evaluate if the other is better or worse
        _metric = self.metric
        return not is_better(metric=self.metric, old_metrics={_metric: self.test_performance},
                             new_metrics={_metric: other.test_performance})


# --------------
# Functions for getting hyperparameter values
# --------------
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

    # Domain discriminator should not be used for iLODO experiments
    if config["DomainDiscriminator"] is not None:
        raise ValueError("Found domain discriminator in an iLODO experiment")

    # Combine outputs and return
    if use_cmmn:
        return "CMMN"
    else:
        return "Nothing"


def _get_band_pass_filter(preprocessing_config):
    return INV_FREQUENCY_BANDS[tuple(preprocessing_config["general"]["filtering"])]  # type: ignore


def _get_normalisation(config):
    if config["Varied Numbers of Channels"]["name"] == "Interpolation":
        return str(config["DL Architecture"]["normalise"])
    elif config["Varied Numbers of Channels"]["name"] == "RegionBasedPooling":
        return str(config["Varied Numbers of Channels"]["kwargs"]["normalise_region_representations"])
    else:
        raise ValueError


def _get_spatial_method(config):
    method = config["Varied Numbers of Channels"]["name"]
    if method == "RegionBasedPooling":
        return "RBP"
    elif method == "Interpolation":
        return config["Varied Numbers of Channels"]["kwargs"]["method"]
    else:
        raise ValueError(f"Unexpected method for handling varied numbers of channels: {method}")


def _get_hyperparameter(config, hparam: HParam):
    if hparam.key_path == "band_pass":
        return _get_band_pass_filter(config)
    elif hparam.key_path == "domain_adaptation":
        return _get_domain_adaptation(config)
    elif hparam.key_path == "sampling_freq_multiple":
        return f"{int(config['general']['resample'] // config['general']['filtering'][-1])} * f max"
    elif hparam.key_path == "num_seconds":
        return f"{int(config['general']['num_time_steps'] // config['general']['resample'])}s"
    elif hparam.key_path == "normalisation":
        return str(_get_normalisation(config))
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


# ----------------
# Functions for getting the best models
# ----------------
def _get_model_performance(path, *, main_metric, datasets):
    dataset_name = get_ilodo_val_dataset_name(path=path)
    if datasets is not None and dataset_name not in datasets:
        raise SkipFold

    # --------------
    # Get the best epoch, as evaluated on the validation set
    # --------------
    # Load the dataframe of the validation performances
    val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

    # Get the best epoch
    if higher_is_better(metric=main_metric):
        best_epoch = numpy.argmax(val_df[main_metric])
    else:
        best_epoch = numpy.argmin(val_df[main_metric])

    # --------------
    # Return the test performance from the same epoch, as well as the name of the validation dataset
    # --------------
    # Using the test performance on the pooled dataset
    test_performance = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))[main_metric][best_epoch]
    return test_performance, dataset_name


def _get_all_model_performances(*, results_dir, runs, main_metric, datasets):
    all_performances: Dict[str, List[_Run]] = {}

    num_runs = len(runs)
    for i, run in enumerate(runs):
        if i % 10 == 0:
            print(f"Run {i + 1}/{num_runs}")

        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Loop through the folds (i-th fold has the i-th dataset as test holdout)
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            fold_path = os.path.join(run_path, fold)  # type: ignore
            try:
                test_performance, dataset = _get_model_performance(
                    path=fold_path, main_metric=main_metric, datasets=datasets,
                )
            except (KeyError, SkipFold):
                continue

            # Add performance to dict
            if dataset not in all_performances:
                all_performances[dataset] = []
            all_performances[dataset].append(
                _Run(run=fold_path, test_performance=test_performance, metric=main_metric)  # type: ignore
            )

    return all_performances


def _get_top_k_models(*, k, results_dir, main_metric, datasets):
    # Get all runs for LODO
    runs = get_all_ilodo_runs(results_dir)

    # --------------
    # Get a list of all model performances for each dataset
    # --------------
    all_performances = _get_all_model_performances(
        results_dir=results_dir, runs=runs, main_metric=main_metric, datasets=datasets
    )

    # --------------
    # Filter runs, to only get the best ones
    # --------------
    # Will have dataset keys and tuple of runs as values
    top_k_models: Dict[str, Tuple[str, ...]] = {}
    for dataset, run_performances in all_performances.items():
        sorted_run_performance = sorted(run_performances)

        idx = int(len(runs) * k)
        top_k_models[dataset] = tuple(run.run for run in sorted_run_performance[:idx])

        _top_performances = tuple(round(run.test_performance, 2) for run in sorted_run_performance[:idx])
        print(f"{dataset} ({main_metric}): {_top_performances}")

    # Print the very best results
    print("\nBest models:")
    for dataset, top_run in top_k_models.items():
        print(f"{dataset}: {top_run[0]}")

    return top_k_models


def _get_model_hyperparameters(run, hyperparameters: Dict[str, HParam]):
    hyp_values = {}
    for name, param in hyperparameters.items():
        # Get the hyperparameter values
        config = get_config_file(results_folder=os.path.dirname(os.path.dirname(run)),
                                 preprocessing=param.preprocessing)
        value = _get_hyperparameter(config=config, hparam=param)

        # Add the value
        hyp_values[name] = value

    return hyp_values


def _plot_top_configurations(*, k, results_dir, hyperparameters, main_metric, datasets):
    # -------------
    # Get top performing models
    # -------------
    top_k_models = _get_top_k_models(k=k, results_dir=results_dir, main_metric=main_metric, datasets=datasets)

    # Get the number of top runs
    _all_num_top_runs = set(len(r) for r in top_k_models.values())
    assert len(_all_num_top_runs) == 1
    num_top_runs = tuple(_all_num_top_runs)[0]

    # -------------
    # Get the hyperparameters
    # -------------
    configurations = {"Source dataset": [], **{param: [] for param in hyperparameters}}
    for dataset, runs in top_k_models.items():
        for run in runs:
            run_configurations = _get_model_hyperparameters(run, hyperparameters=hyperparameters)

            # Add configurations
            configurations["Source dataset"].append(PRETTY_NAME[dataset])
            for param_name, param_value in run_configurations.items():
                _pretty_value = PRETTY_NAME[param_value] if param_value in PRETTY_NAME else param_value
                configurations[param_name].append(_pretty_value)

    # -------------
    # Make plots
    # -------------
    for param_name, param_description in hyperparameters.items():
        pyplot.figure(figsize=FIGSIZE)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            _hue_order = tuple(PRETTY_NAME[d] for d in DATASET_ORDER)
            seaborn.countplot(data=configurations, x=param_name, hue="Source dataset", hue_order=_hue_order,
                              order=param_description.order, stat=STAT)

        # Cosmetics
        pyplot.xlabel(param_name, fontsize=FONTSIZE)
        pyplot.ylabel(STAT.capitalize(), fontsize=FONTSIZE)
        pyplot.ylim(0, num_top_runs)
        pyplot.tick_params(labelsize=FONTSIZE)
        pyplot.gca().yaxis.set_major_locator(MultipleLocator(GRID_SPACING))
        pyplot.grid(axis="y")
        pyplot.tight_layout()

    pyplot.show()


# --------------
# Globals
# --------------
# Plot cosmetics
STAT = "count"
FIGSIZE = (7, 5)
FONTSIZE = 18
GRID_SPACING = 5
rcParams["legend.fontsize"] = FONTSIZE
rcParams["legend.title_fontsize"] = FONTSIZE
DATASET_ORDER = ("TDBrain", "MPILemon", "HatlestadHall")

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
    "Loss": HParam(key_path=("Training", "Loss", "loss"), default=NotImplemented, preprocessing=False,
                   order=("MAE", "MSE")),
    "Band-pass filter": HParam(key_path="band_pass", preprocessing=True, default=NotImplemented,
                               order=("Delta", "Theta", "Alpha", "Beta", "Gamma", "All")),
    "Domain Adaptation": HParam(key_path="domain_adaptation", preprocessing=False, default=NotImplemented,
                                order=("Nothing", "CMMN")),
    "Sampling frequency": HParam(key_path="sampling_freq_multiple", default=NotImplemented, preprocessing=True,
                                 order=(r"$2 \cdot f_{max}$", r"$4 \cdot f_{max}$", r"$8 \cdot f_{max}$")),
    "Time series length (s)": HParam(key_path="num_seconds", default=NotImplemented, preprocessing=True,
                                     order=("5s", "10s")),
    "Normalisation": HParam(key_path="normalisation", preprocessing=False, default=NotImplemented,
                            order=(True, False)),
    "Autoreject": HParam(key_path=("general", "autoreject"), default=NotImplemented, preprocessing=True,
                         order=(True, False)),
}


def main():
    # -------------
    # Hyperparameters
    # -------------
    k = 0.05
    main_metric = "pearson_r"
    hyperparameters = HYPERPARAMETERS
    datasets = ("HatlestadHall", "MPILemon", "TDBrain")
    results_dir = get_results_dir()

    # -------------
    # Make plots
    # -------------
    _plot_top_configurations(k=k, results_dir=results_dir, hyperparameters=hyperparameters, main_metric=main_metric,
                             datasets=datasets)


if __name__ == "__main__":
    main()
