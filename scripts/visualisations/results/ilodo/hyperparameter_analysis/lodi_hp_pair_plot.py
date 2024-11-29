"""
This script also contains copied code from fANOVA

(Sorry about the copypaste from LODO, but I just want this work done atm...)
"""
import dataclasses
import os
import random
from typing import Dict, Tuple, Union, Optional, Set, Any, List

import matplotlib
import numpy
import pandas
import yaml
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, ConfigurationSpace, \
    OrdinalHyperparameter, Constant
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Hyperparameter, NumericalHyperparameter
from fanova import fANOVA
from fanova.visualizer import Visualizer
from matplotlib import pyplot, rcParams
from matplotlib.ticker import FuncFormatter

from cdl_eeg.data.analysis.results_analysis import get_config_file, SkipFold, higher_is_better, PRETTY_NAME, \
    get_ilodo_val_dataset_name, get_all_ilodo_runs
from cdl_eeg.data.paths import get_results_dir


class UpdatedVisualizer(Visualizer):
    """
    3D plot didn't work, so I'll need to make a few changes to the 'Visualizer' class. I can't see that the original
    code has a license file, but here is the GitHub link: https://github.com/automl/fanova/tree/master
    """

    def generate_pairwise_marginal(self, param_list, resolution=20):
        """
        Creates a plot of pairwise marginal of a selected parameters

        Parameters
        ----------
        param_list: list of ints or strings
            Contains the selected parameters
        resolution: int
            Number of samples to generate from the parameter range as
            values to predict

        Returns
        -------
        grid_orig: List[List[...]]
            nested list of two lists with the values that were evaluated for parameters 1 and 2 respectively
        zz: np.array
            array that contains the predicted importance (shape is len(categoricals) or resolution depending on
            parameter, so e.g. zz = np.array(3, 20) if first parameter has three categories and resolution is set to 20
        """
        if len(set(param_list)) != 2:
            raise ValueError("You have to specify 2 (different) parameters")

        params: List[Hyperparameter]
        param_names: List[str]
        param_indices: Union[List[int], numpy.ndarray]
        params, param_names, param_indices = self._get_parameter(param_list)

        # The grid_fanova contains only numeric values (even for categorical choices) while grid_orig contains the
        #   actual value (also for categorical choices!)
        grid_fanova, grid_orig = [], []

        for p in params:
            if isinstance(p, CategoricalHyperparameter):
                grid_orig.append(p.choices)
                grid_fanova.append(numpy.arange(len(p.choices)))
            elif isinstance(p, OrdinalHyperparameter):
                grid_orig.append(p.sequence)
                grid_fanova.append(numpy.arange(len(p.sequence)))
            elif isinstance(p, Constant):
                grid_orig.append((p.value,))
                grid_fanova.append(numpy.arange(1))
            elif isinstance(p, NumericalHyperparameter):
                if p.log:
                    base = numpy.e  # assuming ConfigSpace uses the natural logarithm
                    log_lower = numpy.log(p.lower) / numpy.log(base)
                    log_upper = numpy.log(p.upper) / numpy.log(base)
                    grid = numpy.logspace(log_lower, log_upper, resolution, endpoint=True, base=base)
                else:
                    grid = numpy.linspace(p.lower, p.upper, resolution)
                grid_orig.append(grid)
                grid_fanova.append(grid)
            else:
                raise ValueError("Hyperparameter %s of type %s not supported." % (p.name, type(p)))

        # Turn into arrays, squeeze all but the first two dimensions (avoid squeezing away the dimension for Constants)
        param_indices = numpy.array(param_indices)

        # TT: Don't understand all this... param_indices was list[int], now it is a numpy array with shape=(2,)
        # (the length of param list is 2). 'i' is therefore always in [0, 1]. s != 1 is always true because
        # param_indices.shape = (2,). Hence, we reshape the numpy array to have share=(2,)...
        _copy_param_indices = param_indices.copy()
        param_indices = param_indices.reshape([s for i, s in enumerate(param_indices.shape) if i in [0, 1] or s != 1])
        assert numpy.array_equal(_copy_param_indices, param_indices), f"TT: This makes no sense at all..."

        # TT: Can't apply numpy.array() when grid_fanova is a list with two arrays of unequal shapes. But it is clear to
        # me that the reshaping would have no effect, and since we're only iterating through the two elements in
        # grid_fanova, we're fine with commenting this out

        # grid_fanova = numpy.array(grid_fanova)
        # grid_fanova = grid_fanova.reshape([s for i, s in enumerate(grid_fanova.shape) if i in [0, 1] or s != 1])

        # Populating the result
        zz = numpy.zeros((len(grid_fanova[0]), len(grid_fanova[1])))
        for i, x_value in enumerate(grid_fanova[0]):
            for j, y_value in enumerate(grid_fanova[1]):
                zz[i][j] = self.fanova.marginal_mean_variance_for_values(param_indices, [x_value, y_value])[0]

        return grid_orig, zz

    # noinspection PyMethodOverriding
    def plot_pairwise_marginal(self, param_list, dataset_name, percentile, supplementary, resolution=20, show=False,
                               three_d=True, colormap=matplotlib.cm.jet, add_colorbar=True, title=None):  # type: ignore
        """
        Creates a plot of pairwise marginal of a selected parameters

        Parameters
        ----------
        param_list: list of ints or strings
            Contains the selected parameters
        resolution: int
            Number of samples to generate from the parameter range as
            values to predict
        dataset_name : str
            Name of dataset. Needed for saving the figure
        show: boolean
            whether to call plt.show() to show plot directly as interactive matplotlib-plot
        percentile : float
        three_d: boolean
            whether or not to plot pairwise marginals in 3D-plot
        colormap: matplotlib.Colormap
            which colormap to use for the 3D plots
        add_colorbar: bool
            whether to add the colorbar for 3d plots
        title : str
        supplementary : bool
        """
        if len(set(param_list)) != 2:
            raise ValueError("You have to specify 2 (different) parameters")

        params, param_names, param_indices = self._get_parameter(param_list)

        first_is_numerical = isinstance(params[0], NumericalHyperparameter)
        second_is_numerical = isinstance(params[1], NumericalHyperparameter)

        pyplot.close()
        fig = pyplot.figure(figsize=FIGSIZE["SUPP" if supplementary else "RES"])

        title = '%s and %s' % (param_names[0], param_names[1]) if title is None else title
        pyplot.title(title, fontsize=TITLE_FONTSIZE)

        if first_is_numerical and second_is_numerical:
            # No categoricals -> create heatmap / 3D-plot
            grid_list, zz = self.generate_pairwise_marginal(param_indices, resolution)

            z_min, z_max = zz.min(), zz.max()
            display_xx, display_yy = numpy.meshgrid(grid_list[0], grid_list[1])

            if three_d:
                # This is where I had to change from fanova package
                ax = fig.add_subplot(111, projection='3d')
                surface = ax.plot_surface(display_xx, display_yy, zz.T,
                                          rstride=1, cstride=1, cmap=colormap, linewidth=0, antialiased=False)
                ax.set_xlabel(param_names[0])
                ax.set_ylabel(param_names[1])
                ax.set_zlabel(self._y_label)
                if add_colorbar:
                    fig.colorbar(surface, shrink=0.5, aspect=5)

            else:
                pyplot.pcolor(display_xx, display_yy, zz.T, cmap=colormap, vmin=z_min, vmax=z_max)
                pyplot.xlabel(param_names[0])

                if self.cs_params[param_indices[0]].log:
                    pyplot.xscale('log')
                if self.cs_params[param_indices[1]].log:
                    pyplot.yscale('log')

                pyplot.ylabel(param_names[1])
                pyplot.colorbar()
        else:
            # At least one of the two parameters is non-numerical (categorical, ordinal or constant)
            if first_is_numerical or second_is_numerical:
                # Only one of them is non-numerical -> create multi-line-plot
                # Make sure categorical is first in indices (for iteration below)
                numerical_idx = 0 if first_is_numerical else 1
                categorical_idx = 1 - numerical_idx
                grid_labels, zz = self.generate_pairwise_marginal(param_indices, resolution)

                if first_is_numerical:
                    zz = zz.T

                for i, cat in enumerate(grid_labels[categorical_idx]):
                    if params[numerical_idx].log:
                        pyplot.semilogx(grid_labels[numerical_idx], zz[i], label='%s' % str(cat))
                    else:
                        pyplot.plot(grid_labels[numerical_idx], zz[i], label='%s' % str(cat))

                pyplot.ylabel(self._y_label)
                pyplot.xlabel(param_names[numerical_idx])  # x-axis displays numerical
                pyplot.legend()
                pyplot.tight_layout()

            else:
                # Both parameters are categorical -> create hotmap
                choices, zz = self.generate_pairwise_marginal(param_indices, resolution)
                pyplot.imshow(zz.T, cmap=COLORMAP, interpolation='nearest')

                # Cosmetics
                _x_choices = tuple(LABEL_NAMES.get(choice, choice) for choice in choices[0])
                _y_choices = tuple(LABEL_NAMES.get(choice, choice) for choice in choices[1])
                pyplot.xticks(numpy.arange(0, len(choices[0])), _x_choices, fontsize=FONTSIZE)
                pyplot.yticks(numpy.arange(0, len(choices[1])), _y_choices, fontsize=FONTSIZE)
                # Better to fix this afterward
                # if PRETTY_NAME[dataset_name] == "SRM":
                pyplot.xlabel(param_names[0])

                if percentile == 0 or not supplementary:  # This is a quick fix
                    pyplot.ylabel(param_names[1])

                cbar = pyplot.colorbar()
                if percentile == 95 or not supplementary:  # This is a quick fix
                    cbar.set_label(self._y_label, fontsize=FONTSIZE)
                cbar.ax.tick_params(labelsize=FONTSIZE)
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
                pyplot.tight_layout()

                # Save figure
                results_dir = os.path.join(os.path.dirname(__file__), "pairplots")
                if not os.path.isdir(results_dir):
                    os.mkdir(results_dir)

                pyplot.savefig(os.path.join(
                    results_dir, f"{param_names[0]}_{param_names[1]}_{PRETTY_NAME[dataset_name].lower()}_percentile_"
                                 f"{percentile}_supp_{supplementary}.png"
                ), dpi=DPI)

        if show:
            pyplot.show()

        return pyplot

# ---------------
# Convenient dataclasses
# ---------------
@dataclasses.dataclass(frozen=True)
class _HP:
    key_path: Union[Tuple[str, ...], str]
    preprocessing: bool
    dist_path: Optional[Union[Tuple[str, ...], str]] = None
    in_dist_config: bool = True  # If the distribution is to be found in the config file. If False, special case must be
    # implemented
    is_conditional: bool = False  # If this is True, we will interpret KeyError raised when the HP is not found. But I
    # don't think I can use it... :(


@dataclasses.dataclass(frozen=True)
class _HPCResult:
    """Hyperparameter configurations and the corresponding score"""
    configurations: Dict[str, Any]
    score: float


# ---------------
# Getting configurations. fANOVA requires numerical encoding
# ---------------
_ENCODING = {"spatial_method": {"RBP": 0, "spline": 1, "MNE": 2},
             "band_pass": {"Delta": 0, "Theta": 1, "Alpha": 2, "Beta": 3, "Gamma": 4, "All": 5},
             "dl_architecture": {"InceptionNetwork": 0, "ShallowFBCSPNetMTS": 1, "Deep4NetMTS": 2},
             "sampling_freq": {"2 * f max": 0, "4 * f max": 1, "8 * f max": 2},
             "normalisation": {False: 0, True: 1},
             "input_length": {"5s": 0, "10s": 1},
             "loss": {"L1Loss": 0, "MSELoss": 1},
             "autoreject": {False: 0, True: 1}}


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
        return 0  # "CMMN + DD"
    elif use_cmmn:
        return 1  # "CMMN"
    elif use_domain_discriminator:
        return 2  # "DD"
    else:
        return 3  # "Nothing"


def _get_spatial_method(config):
    method = config["Varied Numbers of Channels"]["name"]
    if method == "RegionBasedPooling":
        return _ENCODING["spatial_method"]["RBP"]
    elif method == "Interpolation":
        return _ENCODING["spatial_method"][config["Varied Numbers of Channels"]["kwargs"]["method"]]
    else:
        raise ValueError(f"Unexpected method for handling varied numbers of channels: {method}")


def _get_band_pass_filter(preprocessing_config):
    band_pass = INV_FREQUENCY_BANDS[tuple(preprocessing_config["general"]["filtering"])]  # type: ignore
    return _ENCODING["band_pass"][band_pass]


def _get_normalisation(config):
    if config["Varied Numbers of Channels"]["name"] == "Interpolation":
        return _ENCODING["normalisation"][config["DL Architecture"]["normalise"]]
    elif config["Varied Numbers of Channels"]["name"] == "RegionBasedPooling":
        return _ENCODING["normalisation"][config["Varied Numbers of Channels"]["kwargs"][("normalise_region_"
                                                                                          "representations")]]
    else:
        raise ValueError


def _get_weighted_loss(config):
    loss_config = config["Training"]["Loss"]  # tau in the paper (as per now, at least)

    # Not using weighted loss is equivalent to tau = 0
    return 0 if loss_config["weighter"] is None else loss_config["weighter_kwargs"]["weight_power"]


def _get_dl_architecture(config):
    return _ENCODING["dl_architecture"][config["DL Architecture"]["model"]]


def _get_loss(config):
    return _ENCODING["loss"][config["Training"]["Loss"]["loss"]]


def _get_autoreject(config):
    return _ENCODING["autoreject"][config["general"]["autoreject"]]


def _get_hyperparameter(config, hparam: _HP, dist):
    if hparam.key_path == "domain_adaptation" and not dist:
        return _get_domain_adaptation(config)
    elif hparam.key_path == "spatial_dimension" and not dist:
        return _get_spatial_method(config)
    elif hparam.key_path == "band_pass":
        return _get_band_pass_filter(config)
    elif hparam.key_path == "sampling_freq_multiple":
        _enc = _ENCODING["sampling_freq"]
        return _enc[f"{int(config['general']['resample'] // config['general']['filtering'][-1])} * f max"]
    elif hparam.key_path == "normalisation":
        return _get_normalisation(config)
    elif hparam.key_path == "num_seconds":
        _enc = _ENCODING["input_length"]
        return _enc[f"{int(config['general']['num_time_steps'] // config['general']['resample'])}s"]
    elif hparam.key_path == "weighted_loss":
        return _get_weighted_loss(config)
    elif hparam.key_path == "dl_architecture" and not dist:
        return _get_dl_architecture(config)
    elif hparam.key_path == "loss" and not dist:
        return _get_loss(config)
    elif hparam.key_path == "autoreject" and not dist:
        return _get_autoreject(config)

    hyperparameter = config.copy()
    keys = hparam.dist_path if dist and hparam.dist_path is not None else hparam.key_path
    for key in keys:
        hyperparameter = hyperparameter[key]
    return hyperparameter


def _get_params(path, hyperparameters):
    """Generates parameters to be passed to the Optuna create_trial"""
    config = get_config_file(results_folder=path, preprocessing=False)  # todo
    preprocessing_config = get_config_file(results_folder=path, preprocessing=True)

    hparams = dict()
    for hp_name, hp_value in hyperparameters.items():
        try:
            _config = preprocessing_config if hp_value.preprocessing else config
            hparams[hp_name] = _get_hyperparameter(config=_config, hparam=hp_value, dist=False)
        except KeyError as e:
            if not hp_value.is_conditional:
                raise KeyError(f"Could not find the HP {hp_name}. If this is expected (conditional HP), "
                               f"set is_conditional to True") from e

    return hparams


# ----------------
# Functions for getting the results
# ----------------
def _get_validation_performance(path, *, main_metric):
    # Load the dataframe of the validation performances
    val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

    # Get the best performance and its epoch
    if higher_is_better(metric=main_metric):
        val_idx = numpy.argmax(val_df[main_metric])
    else:
        val_idx = numpy.argmin(val_df[main_metric])
    val_metric = val_df[main_metric][val_idx]

    return val_metric, val_idx


def _get_pooled_test_performance(path,  *, target_metric, selection_metric, datasets):
    """Function for getting the test score"""
    # --------------
    # Get train/val dataset name
    # --------------
    dataset_name = get_ilodo_val_dataset_name(path)
    if dataset_name not in datasets:
        # If we are not interested, we will not waste time loading data for then to discard the results
        raise SkipFold

    # --------------
    # Get validation performance and optimal epoch
    # --------------
    val_performance, epoch = _get_validation_performance(path=path, main_metric=selection_metric)

    # --------------
    # Get test performance
    # --------------
    test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
    test_performance = test_df[target_metric][epoch]

    return test_performance, dataset_name


def _get_mts_module_dist(hp_name, *, distribution):
    return CategoricalHyperparameter(name=hp_name, choices=distribution.keys())


def _get_domain_adaptation_dist(hp_name):
    return CategoricalHyperparameter(name=hp_name, choices=("CMMN + DD", "CMMN", "DD", "Nothing"))


def _get_spatial_dim_choices(hp_name):
    # I don't think I need to specify the weights
    return CategoricalHyperparameter(name=hp_name, choices=("spline", "MNE", "RBP"), weights=(0.25, 0.25, 0.5))


def _get_band_pass_dist(hp_name):
    return CategoricalHyperparameter(name=hp_name, choices=list(INV_FREQUENCY_BANDS.values()))


def _get_sampling_freq_dist(hp_name):
    return OrdinalHyperparameter(name=hp_name, sequence=("2 * f max", "4 * f max", "8 * f max"))


def _get_normalisation_dist(hp_name):
    return CategoricalHyperparameter(name=hp_name, choices=(True, False))


def _get_input_length_dist(hp_name):
    return OrdinalHyperparameter(name=hp_name, sequence=("5s", "10s"))


def _get_loss_dist(hp_name):
    return CategoricalHyperparameter(name=hp_name, choices=("L1Loss", "MSELoss"))


def _get_weighted_loss_dist(hp_name):
    # Looks to me like only the upper and lower bounds are used anyway (all instances of type NumericalHyperparameter
    # seem to be treated equally). Changing to a normal distribution doesn't change the outcome neither
    return UniformFloatHyperparameter(name=hp_name, lower=0, upper=1)


def _get_autoreject_dist(hp_name):
    return CategoricalHyperparameter(name=hp_name, choices=(True, False))


def _config_dist_to_optuna_dist(distribution, hp_name):
    """Convert from how the distributions are specified in the sampling config file to Optuna compatible instances"""
    if hp_name == "DL Architecture":
        return _get_mts_module_dist(hp_name=hp_name, distribution=distribution)
    elif hp_name == "Domain Adaptation":
        return _get_domain_adaptation_dist(hp_name=hp_name)
    elif hp_name == "Spatial Dimension Handling":
        return _get_spatial_dim_choices(hp_name=hp_name)
    elif hp_name == "Band-pass filter":
        return _get_band_pass_dist(hp_name=hp_name)
    elif hp_name == "Sampling frequency":
        return _get_sampling_freq_dist(hp_name=hp_name)
    elif hp_name == "Normalisation":
        return _get_normalisation_dist(hp_name=hp_name)
    elif hp_name == "Input length":
        return _get_input_length_dist(hp_name=hp_name)
    elif hp_name == "Loss":
        return _get_loss_dist(hp_name=hp_name)
    elif hp_name == r"Weighted loss ($\tau$)":
        return _get_weighted_loss_dist(hp_name=hp_name)
    elif hp_name == "Autoreject":
        return _get_autoreject_dist(hp_name=hp_name)

    # Not a very elegant function... should preferably use the sampling distribution functions
    dist_name = distribution["dist"]
    if dist_name == "uniform":
        low = distribution["kwargs"]["a"]
        high = distribution["kwargs"]["b"]
        return UniformFloatHyperparameter(name=hp_name, lower=low, upper=high, log=False)
    elif dist_name == "log_uniform":
        base = distribution["kwargs"]["base"]
        low = base ** distribution["kwargs"]["a"]
        high = base ** distribution["kwargs"]["b"]
        return UniformFloatHyperparameter(name=hp_name, lower=low, upper=high, log=True)
    if dist_name == "uniform_int":
        low = distribution["kwargs"]["a"]
        high = distribution["kwargs"]["b"]
        return UniformIntegerHyperparameter(name=hp_name, lower=low, upper=high, log=False)
    elif dist_name == "log_uniform_int":
        base = distribution["kwargs"]["base"]
        low = base ** distribution["kwargs"]["a"]
        high = base ** distribution["kwargs"]["b"]
        return UniformIntegerHyperparameter(name=hp_name, lower=low, upper=high, log=True)
    elif dist_name == "n_log_uniform_int":
        # I'll register it as a log scale
        n = distribution["kwargs"]["n"]
        base = distribution["kwargs"]["base"]
        low = n * round(base ** distribution["kwargs"]["a"])
        high = n * round(base ** distribution["kwargs"]["b"])
        return UniformIntegerHyperparameter(name=hp_name, lower=low, upper=high, log=True)
    else:
        raise ValueError(f"Unrecognised distribution: {dist_name}")


def _get_distributions(hyperparameters, dist_config) -> Dict[str, Hyperparameter]:
    """Function for getting the distributions spaces of the HPs, compatible with Optuna"""
    # --------------
    # Loop through the desired hyperparameters
    # --------------
    hp_distributions: Dict[str, Hyperparameter] = dict()
    for hp_name, hp_info in hyperparameters.items():
        if hp_info.in_dist_config:
            hp_dist = _get_hyperparameter(dist_config, hparam=hp_info, dist=True)
        else:
            hp_dist = None
        hp_distributions[hp_name] = _config_dist_to_optuna_dist(distribution=hp_dist, hp_name=hp_name)

    return hp_distributions


def create_fanova(*, datasets, runs, results_dir, target_metric, selection_metric, dist_config, hyperparameters=None,
                  fanova_kwargs=None, verbose):
    """
    Function for creating a fANOVA object per dataset. Once these are obtained, hyperparameter importance can be
    assessed config_space

    Parameters
    ----------
    datasets : tuple[str, ...]
        Dataset names we want to create studies of
    runs : tuple[str, ...]
        Runs we want to add to the studies
    results_dir
        Path to where the results are stored
    hyperparameters
        The hyperparameters to register as part of the trials
    target_metric
        Metric to optimise
    selection_metric
        Metric for model selection (epoch selection in this case)
    dist_config
        configurations file containing distributions used
    fanova_kwargs : dict[str, typing.Any], optional
        kwargs to pass to fANOVA
    verbose : bool
        To print the dataframes or not

    Returns
    -------
    tuple[dict[str, fANOVA], dict[str, pandas.DataFrame]]
    """
    fanova_kwargs = dict() if fanova_kwargs is None else fanova_kwargs

    # -------------
    # Extract the HP distributions
    # -------------
    hyperparameters = _HYPERPARAMETERS.copy() if hyperparameters is None else hyperparameters
    distributions: Dict[str, Hyperparameter] = _get_distributions(
        hyperparameters=hyperparameters, dist_config=dist_config
    )

    # Create the configuration space
    config_space = ConfigurationSpace(distributions)

    # -------------
    # Collate HPCs and results
    # -------------
    studies: Dict[str, List[_HPCResult]] = {dataset_name: [] for dataset_name in datasets}
    for run in runs:
        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the hyperparameters of the current run/experiment
        parameters = _get_params(path=os.path.dirname(run_path), hyperparameters=hyperparameters)

        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            # Get the performance
            try:
                test_performance, dataset_name = _get_pooled_test_performance(
                    path=os.path.join(run_path, fold), target_metric=target_metric,  # type: ignore
                    selection_metric=selection_metric, datasets=datasets,
                )
            except SkipFold:
                continue
            except KeyError:
                # todo: how to register seriously bad runs? This error should only occur for correlations, so could set
                #  it to zero?
                continue

            # Add the configurations and score study object of the current dataset
            studies[dataset_name].append(_HPCResult(configurations=parameters, score=test_performance))

    # -------------
    # Convert to dataframes as recommended in the fANOVA documentation
    # -------------
    dfs: Dict[str, pandas.DataFrame] = dict()
    for dataset_name, hp_results in studies.items():
        hp_history = {hp_name: [] for hp_name in hyperparameters}

        # Quick sanity check
        assert target_metric not in hp_history, (f"A hyperparameter had the same name as the target metric "
                                                 f"({target_metric})")
        hp_history[target_metric] = []

        for hp_result in hp_results:
            # Add all configuration values
            for hp_name, hp_value in hp_result.configurations.items():
                hp_history[hp_name].append(hp_value)

            # Add the score
            hp_history[target_metric].append(hp_result.score)

        # Add the dataframe
        df = pandas.DataFrame(hp_history)
        if target_metric in ("pearson_r", "spearman_rho"):
            df.fillna(0.0, inplace=True)
        dfs[dataset_name] = df


    # -------------
    # Make fANOVA objects
    # -------------
    if verbose:
        for dataset_name, df in dfs.items():
            print(dataset_name)
            print(df)

    fanovas: Dict[str, fANOVA] = {
        dataset_name: fANOVA(X=df.drop(target_metric, axis="columns"), Y=df[target_metric], config_space=config_space,
                             **fanova_kwargs)
        for dataset_name, df in dfs.items()
    }
    performance_df: Dict[str, pandas.DataFrame] = {dataset_name: df[target_metric] for dataset_name, df in dfs.items()}

    return fanovas, performance_df


# --------------
# Plot functions
# --------------
def _plot_hp_interactions(studies, *, plot_3d, performance_df, percentiles, resolution, hps_to_plot, show,
                          supplementary):
    for dataset_name, fanova in studies.items():
        for percentile in percentiles:
            # Compute cutoff (assuming correlation coefficient)
            lower_cutoff = numpy.percentile(performance_df[dataset_name], percentile)
            fanova.set_cutoffs(cutoffs=(lower_cutoff, numpy.inf))

            # Select visualiser (had to make some changes for 3D)
            if plot_3d:
                visualiser = UpdatedVisualizer(fanova, fanova.cs, directory=os.path.dirname(__file__))
            else:
                visualiser = UpdatedVisualizer(fanova, fanova.cs, directory=os.path.dirname(__file__))

            # Plot the hyperparameter pairs
            for hp_pair in hps_to_plot:  # fanova.get_most_important_pairwise_marginals(n=num_pairwise_marginals):
                if supplementary:
                    title = f"Percentile {percentile}"
                else:
                    title = f"{PRETTY_NAME[dataset_name]} (Percentile {percentile})"

                visualiser.plot_pairwise_marginal(
                    hp_pair, show=show, resolution=resolution, three_d=plot_3d, title=title, dataset_name=dataset_name,
                    percentile=percentile, supplementary=supplementary
                )


INV_FREQUENCY_BANDS = {(1.0, 4.0): "Delta",
                       (4.0, 8.0): "Theta",
                       (8.0, 12.0): "Alpha",
                       (12.0, 30.0): "Beta",
                       (30.0, 45.0): "Gamma",
                       (1.0, 45.0): "All"}


_HYPERPARAMETERS = {
    # Training HPs
    "Learning rate": _HP(key_path=("Training", "learning_rate"), preprocessing=False),
    r"$\beta_1$": _HP(key_path=("Training", "beta_1"), preprocessing=False),
    r"$\beta_2$": _HP(key_path=("Training", "beta_2"), preprocessing=False),
    r"$\epsilon$": _HP(key_path=("Training", "eps"), preprocessing=False),
    # DL architecture
    "DL Architecture": _HP(key_path="dl_architecture", dist_path=("MTS Module",), preprocessing=False),
    # Spatial dimension mismatch
    "Spatial Dimension Handling": _HP(key_path="spatial_dimension", preprocessing=False, in_dist_config=False),
    # Domain discriminator
    "Domain Adaptation": _HP(key_path="domain_adaptation", preprocessing=False, in_dist_config=False),
    # Pre-processing / feature extraction
    "Band-pass filter": _HP(key_path="band_pass", preprocessing=True, in_dist_config=False),
    "Sampling frequency": _HP(key_path="sampling_freq_multiple", preprocessing=True, in_dist_config=False),
    "Normalisation": _HP(key_path="normalisation", preprocessing=False, in_dist_config=False),
    "Input length": _HP(key_path="num_seconds", preprocessing=True, in_dist_config=False),
    "Autoreject": _HP(key_path="autoreject", preprocessing=True, in_dist_config=False),
    # Loss
    "Loss": _HP(key_path="loss", preprocessing=False, in_dist_config=False),
}


FONTSIZE = 20
TITLE_FONTSIZE = FONTSIZE + 4
FIGSIZE = {"SUPP": (7.1, 9), "RES": (7.1, 9)}
DPI = 300
Y_ROTATION = None
COLORMAP = "viridis"
ERRORBAR = "sd"
rcParams["legend.fontsize"] = FONTSIZE
rcParams["legend.title_fontsize"] = FONTSIZE
rcParams["axes.labelsize"] = FONTSIZE
_GREEK = {"Delta": r"$\delta$", "Theta": r"$\theta$", "Alpha": r"$\alpha$", "Beta": r"$\beta$", "Gamma": r"$\gamma$",
          "All": "All"}
LABEL_NAMES = {"InceptionNetwork": "IN", "ShallowFBCSPNetMTS": "SN", "Deep4NetMTS": "DN", **_GREEK}


def main():
    meaning_of_life = 42

    random.seed(meaning_of_life)
    numpy.random.seed(meaning_of_life)

    # ----------------
    # A few design choices for the analysis
    # ----------------
    datasets = ("TDBrain", "MPILemon", "HatlestadHall")
    hps_to_plot = (("DL Architecture", "Band-pass filter"), ("Normalisation", "Band-pass filter"))  # Determined after
    # running 'hp_analysis' script
    investigated_hps = tuple(_HYPERPARAMETERS)
    percentiles = (0, 50, 75, 90, 95)
    plot_3d = False
    fanova_kwargs = {"n_trees": 64, "max_depth": 64}
    resolution = 50
    verbose = False
    show_plots = False
    results_dir = get_results_dir()
    selection_metric = "mae"
    target_metric = "pearson_r"
    hyperparameters = _HYPERPARAMETERS.copy()
    hyperparameters = {hp: hyperparameters[hp] for hp in investigated_hps}
    runs = get_all_ilodo_runs(results_dir=results_dir, successful_only=True)  # [:20]
    config_dist_path = os.path.join(  # not very elegant...
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "models",
        "training", "config_files", "hyperparameter_random_search.yml"
    )
    with open(config_dist_path) as f:
        config_dist = yaml.safe_load(f)

    # ----------------
    # Create the fANOVA objects
    # ----------------
    studies: Dict[str, fANOVA]
    studies, performance_df = create_fanova(
        datasets=datasets, runs=runs, results_dir=results_dir, target_metric=target_metric,
        selection_metric=selection_metric, hyperparameters=hyperparameters, dist_config=config_dist, verbose=verbose,
        fanova_kwargs=fanova_kwargs
    )

    # ----------------
    # Interaction plot
    # ----------------
    # The fanova package uses numpy.float, which is deprecated. The error message says that replacing with 'float' is
    # safe
    numpy.float = float

    # HP interaction analysis
    for supplementary in (True, False):  # Quick fix
        _plot_hp_interactions(
            studies, plot_3d=plot_3d, performance_df=performance_df, resolution=resolution, hps_to_plot=hps_to_plot,
            percentiles=percentiles, show=show_plots, supplementary=supplementary
        )


if __name__ == "__main__":
    main()
