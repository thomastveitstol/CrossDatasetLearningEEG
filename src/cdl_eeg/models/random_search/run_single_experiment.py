import copy
import itertools
import os
import random
import warnings
from typing import Any, Dict, Optional, List

import matplotlib
import numpy
import pandas
import seaborn
import torch
from matplotlib import pyplot
from torch import optim
from torch.utils.data import DataLoader
from umap import UMAP

from cdl_eeg.data.combined_datasets import CombinedDatasets
from cdl_eeg.data.data_generators.data_generator import DownstreamDataGenerator, strip_tensors
from cdl_eeg.data.subject_split import get_data_split, leave_1_fold_out
from cdl_eeg.data.scalers.target_scalers import get_target_scaler
from cdl_eeg.models.losses import CustomWeightedLoss, get_activation_function
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel
from cdl_eeg.models.metrics import save_discriminator_histories_plots, save_histories_plots, Histories, \
    save_test_histories_plots, compute_distribution_distance
from cdl_eeg.models.utils import tensor_dict_to_device


class Experiment:
    """
    Class for running a single cross validation experiment
    """

    __slots__ = "_config", "_pre_processing_config", "_results_path", "_device"

    def __init__(self, config, pre_processing_config, results_path, device=None):
        """
        Initialise

        Parameters
        ----------
        config : dict[str, Any]
            The main config file for the experiment
        pre_processing_config : dict[str, Any]
            The pre-processing config file
        results_path : str
            The path of where to store the results
        """
        self._config = config
        self._pre_processing_config = pre_processing_config
        self._results_path = results_path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # -------------
    # Methods for preparing for cross validation
    # -------------
    def _load_data(self):
        """Method for loading data"""
        return CombinedDatasets.from_config(config=self.datasets_config, target=self.train_config["target"])

    @staticmethod
    def _extract_dataset_details(combined_dataset: CombinedDatasets):
        subjects = combined_dataset.dataset_subjects
        datasets = combined_dataset.datasets
        channel_systems = {dataset.name: dataset.channel_system for dataset in datasets}
        channel_name_to_index = {dataset.name: dataset.channel_name_to_index() for dataset in datasets}
        return {"subjects": subjects, "channel_systems": channel_systems,
                "channel_name_to_index": channel_name_to_index}

    def _make_subject_split(self, subjects):
        """Method for splitting subjects into multiple folds"""
        data_split = get_data_split(split=self.subject_split_config["name"], dataset_subjects=subjects,
                                    **self.subject_split_config["kwargs"])
        return data_split.folds

    # -------------
    # Methods to be used inside cross validation
    # -------------
    def run_inverted_cross_validation(self, *, folds, channel_systems, channel_name_to_index, combined_dataset):
        test_histories: Dict[str, Histories] = dict()

        # Loop through all folds
        for i, train_val_subjects in enumerate(folds):
            print(f"\nFold {i + 1}/{len(folds)}")

            # -----------------
            # Make folder for the current fold
            # -----------------
            fold_path = os.path.join(self._results_path, f"Fold_{i}")
            os.mkdir(fold_path)

            # -----------------
            # Split into train and validation
            # -----------------
            # Shuffle the non-test subjects randomly
            non_test_subjects = list(train_val_subjects)
            random.shuffle(non_test_subjects)

            # Split into train and validation
            num_subjects = len(non_test_subjects)
            split_idx = int(num_subjects * (1 - self.train_config["val_split"]))
            train_subjects = tuple(non_test_subjects[:split_idx])
            val_subjects = tuple(non_test_subjects[split_idx:])

            # Get the test subjects
            test_subjects = leave_1_fold_out(i, folds=folds)

            # -----------------
            # Run the current fold
            # -----------------
            histories = self._run_single_fold(
                train_subjects=train_subjects, val_subjects=val_subjects, test_subjects=test_subjects,
                results_path=fold_path, channel_systems=channel_systems, channel_name_to_index=channel_name_to_index,
                combined_dataset=combined_dataset
            )
            # -----------------
            # Save test history
            # -----------------
            # Extract the test history (should only be one)
            _test_histories = tuple(history for history in histories if history.name[:4] == "test")

            if not _test_histories:
                continue
            if len(_test_histories) != 1:
                raise RuntimeError(f"Expected only one test history per fold, but found {len(_test_histories)}")

            test_history = tuple(_test_histories)[0]

            # If there is only one dataset in the test fold, name it as the dataset name, otherwise just use fold number
            test_datasets = set(subject.dataset_name for subject in test_subjects)
            test_name = tuple(test_datasets)[0] if len(test_datasets) == 1 else f"Fold {i}"

            # Add histories object to dict
            if test_name in test_histories:
                raise RuntimeError  # todo: add message
            test_histories[test_name] = test_history

        save_test_histories_plots(path=self._results_path, histories=test_histories)

    def run_cross_validation(self, *, folds, channel_systems, channel_name_to_index, combined_dataset):
        test_histories: Dict[str, Histories] = dict()

        # Loop through all folds
        for i, test_subjects in enumerate(folds):
            print(f"\nFold {i + 1}/{len(folds)}")

            # -----------------
            # Make folder for the current fold
            # -----------------
            fold_path = os.path.join(self._results_path, f"Fold_{i}")
            os.mkdir(fold_path)

            # -----------------
            # Split into train and validation
            # -----------------
            train_subjects, val_subjects = self._train_val_split(folds=folds, left_out_fold=i)

            # -----------------
            # Run the current fold
            # -----------------
            histories = self._run_single_fold(
                train_subjects=train_subjects, val_subjects=val_subjects, test_subjects=test_subjects,
                results_path=fold_path, channel_systems=channel_systems, channel_name_to_index=channel_name_to_index,
                combined_dataset=combined_dataset
            )
            # -----------------
            # Save test history
            # -----------------
            # Extract the test history (should only be one)
            _test_histories = tuple(history for history in histories if history.name[:4] == "test")

            if not _test_histories:
                continue
            if len(_test_histories) != 1:
                raise RuntimeError(f"Expected only one test history per fold, but found {len(_test_histories)}")

            test_history = tuple(_test_histories)[0]

            # If there is only one dataset in the test fold, name it as the dataset name, otherwise just use fold number
            test_datasets = set(subject.dataset_name for subject in test_subjects)
            test_name = tuple(test_datasets)[0] if len(test_datasets) == 1 else f"Fold {i}"

            # Add histories object to dict
            if test_name in test_histories:
                raise RuntimeError  # todo: add message
            test_histories[test_name] = test_history

        save_test_histories_plots(path=self._results_path, histories=test_histories)

    def _train_val_split(self, folds, left_out_fold):
        # Exclude the test fold
        non_test_subjects = list(leave_1_fold_out(i=left_out_fold, folds=folds))

        # Shuffle the non-test subjects randomly
        random.shuffle(non_test_subjects)

        # Split into train and validation
        num_subjects = len(non_test_subjects)
        split_idx = int(num_subjects * (1 - self.train_config["val_split"]))
        train_subjects = non_test_subjects[:split_idx]
        val_subjects = non_test_subjects[split_idx:]

        return train_subjects, val_subjects

    def _run_single_fold(self, *, train_subjects, val_subjects, test_subjects, channel_systems, channel_name_to_index,
                         combined_dataset: CombinedDatasets, results_path):
        # -----------------
        # Define model
        # -----------------
        print("Defining model...")
        # Filter some warnings from Voronoi split
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            model = self._make_model()

        # Fit channel systems
        self._fit_channel_systems(model=model, channel_systems=channel_systems)

        # (Maybe) fit the CMMN layers of RBP
        if model.any_rbp_cmmn_layers:
            self._fit_cmmn_layers(model=model, train_data=combined_dataset.get_data(train_subjects),
                                  channel_systems=channel_systems)

        # -----------------
        # Create data loaders (and target scaler)
        # -----------------
        print("Creating data loaders...")
        # Create loaders for training and validation
        train_loader, val_loader, target_scaler = self._load_train_val_data_loaders(
            model=model, train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset
        )

        # Maybe create loaders for test data
        test_loader: Optional[DataLoader[Any]]
        if self.train_config["continuous_testing"]:
            test_loader = self._load_test_data_loader(model=model, test_subjects=test_subjects,
                                                      combined_dataset=combined_dataset, target_scaler=target_scaler)

            # Also, maybe fit the CMMN monge filters
            if model.any_rbp_cmmn_layers:
                self._fit_cmmn_layers_test_data(model=model, test_data=combined_dataset.get_data(test_subjects),
                                                channel_systems=channel_systems)
        else:
            test_loader = None

        # Some type checks
        if not isinstance(train_loader.dataset, DownstreamDataGenerator):
            raise TypeError(f"Expected training Pytorch datasets to inherit from {DownstreamDataGenerator.__name__}, "
                            f"but found {type(train_loader.dataset)}")
        if not isinstance(val_loader.dataset, DownstreamDataGenerator):
            raise TypeError(f"Expected validation Pytorch datasets to inherit from {DownstreamDataGenerator.__name__}, "
                            f"but found {type(val_loader.dataset)}")

        # -----------------
        # Create loss and optimiser
        # -----------------
        dataset_sizes = train_loader.dataset.dataset_sizes

        # For the downstream model
        optimiser, criterion = self._create_loss_and_optimiser(model=model, dataset_sizes=dataset_sizes)

        # Maybe for a domain discriminator
        discriminator_criterion: Optional[CustomWeightedLoss]
        if self.domain_discriminator_config is None:
            discriminator_criterion = None
            discriminator_weight = None
            discriminator_metrics = None
        else:
            (discriminator_criterion, discriminator_weight,
             discriminator_metrics) = self._get_domain_discriminator_details(dataset_sizes=dataset_sizes)

        # -----------------
        # Train model
        # -----------------
        print(f"{' Training ':-^20}")

        histories = model.train_model(
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            metrics=self.train_config["metrics"], main_metric=self.train_config["main_metric"],
            classifier_criterion=criterion, optimiser=optimiser, discriminator_criterion=discriminator_criterion,
            discriminator_weight=discriminator_weight, discriminator_metrics=discriminator_metrics,
            num_epochs=self.train_config["num_epochs"], verbose=self.train_config["verbose"],
            channel_name_to_index=channel_name_to_index, device=self._device, target_scaler=target_scaler,
            prediction_activation_function=get_activation_function(self.train_config["prediction_activation_function"]),
            sub_group_splits=self.sub_groups_config["sub_groups"], sub_groups_verbose=self.sub_groups_config["verbose"]
        )

        # -----------------
        # Test model (but only if continuous testing was not used)
        # -----------------
        test_estimate: Optional[Histories]
        if not self.train_config["continuous_testing"]:
            print(f"\n{' Testing ':-^20}")

            # Get test loader
            test_loader = self._load_test_data_loader(model=model, test_subjects=test_subjects,
                                                      combined_dataset=combined_dataset, target_scaler=target_scaler)

            # Also, maybe fit the CMMN monge filters
            if model.any_rbp_cmmn_layers:
                self._fit_cmmn_layers_test_data(model=model, test_data=combined_dataset.get_data(test_subjects),
                                                channel_systems=channel_systems)

            # Test model on test data
            test_estimate = model.test_model(
                data_loader=test_loader, metrics=self.train_config["metrics"], verbose=self.train_config["verbose"],
                channel_name_to_index=channel_name_to_index, device=self._device, target_scaler=target_scaler,
                sub_group_splits=self.sub_groups_config["sub_groups"],
                sub_groups_verbose=self.sub_groups_config["verbose"]
            )
        else:
            test_estimate = None

        # -----------------
        # Save results
        # -----------------
        self._save_results(histories=histories, test_estimate=test_estimate, results_path=results_path)

        return histories

    # -------------
    # Methods for saving results
    # -------------
    def _save_results(self, *, histories, test_estimate, results_path):
        # Save prediction histories
        if self.domain_discriminator_config is None:
            train_history, val_history, test_history = histories

            train_history.save_prediction_history(history_name="train_history", path=results_path)
            val_history.save_prediction_history(history_name="val_history", path=results_path)
            if test_history is not None:
                test_history.save_prediction_history(history_name="test_history", path=results_path)
        else:
            domain_discriminator_path = os.path.join(results_path, "domain_discriminator")
            os.mkdir(domain_discriminator_path)

            train_history, val_history, test_history, dd_train_history, dd_val_history = histories

            train_history.save_prediction_history(history_name="train_history", path=results_path)
            val_history.save_prediction_history(history_name="val_history", path=results_path)
            if test_history is not None:
                test_history.save_prediction_history(history_name="test_history", path=results_path)

            dd_train_history.save_prediction_history(history_name="dd_train_history",
                                                     path=domain_discriminator_path)
            dd_val_history.save_prediction_history(history_name="dd_val_history",
                                                   path=domain_discriminator_path)

            # Save domain discriminator metrics plots
            save_discriminator_histories_plots(path=domain_discriminator_path,
                                               histories=(dd_train_history, dd_val_history))

        # Save subgroup plots
        sub_group_path = os.path.join(results_path, "sub_groups_plots")
        os.mkdir(sub_group_path)

        train_history.save_subgroup_metrics_plots(history_name="train", path=sub_group_path)
        val_history.save_subgroup_metrics_plots(history_name="val", path=sub_group_path)
        if test_history is not None:
            test_history.save_subgroup_metrics_plots(history_name="test", path=sub_group_path)

        # Save plots
        save_histories_plots(path=results_path, train_history=train_history, val_history=val_history,
                             test_estimate=test_estimate, test_history=test_history)

    # -------------
    # Method for creating optimisers and loss
    # -------------
    def _get_domain_discriminator_details(self, dataset_sizes):
        # Initialise domain discriminator kwargs
        dd_kwargs = copy.deepcopy(self.domain_discriminator_config)

        # Maybe add sample weighting
        if dd_kwargs["training"]["Loss"]["weighter"] is not None:
            dd_kwargs["training"]["Loss"]["weighter_kwargs"]["dataset_sizes"] = dataset_sizes

        # Set criterion, weight of domain discriminator loss, and the metrics to be used
        discriminator_criterion = CustomWeightedLoss(**dd_kwargs["training"]["Loss"])
        discriminator_weight = dd_kwargs["training"]["lambda"]
        discriminator_metrics = dd_kwargs["training"]["metrics"]

        return discriminator_criterion, discriminator_weight, discriminator_metrics

    def _create_loss_and_optimiser(self, model, dataset_sizes):
        # Create optimiser
        optimiser = optim.Adam(model.parameters(), lr=self.train_config["learning_rate"],
                               betas=(self.train_config["beta_1"], self.train_config["beta_2"]),
                               eps=self.train_config["eps"])

        # Create loss
        if self.train_config["Loss"]["weighter"] is not None:
            self.train_config["Loss"]["weighter_kwargs"]["dataset_sizes"] = dataset_sizes
        criterion = CustomWeightedLoss(**self.train_config["Loss"])

        return optimiser, criterion

    # -------------
    # Method for creating Pytorch data loaders
    # -------------
    def _load_train_val_data_loaders(self, *, model, train_subjects, val_subjects, combined_dataset):
        # Extract input data
        train_data = combined_dataset.get_data(subjects=train_subjects)
        val_data = combined_dataset.get_data(subjects=val_subjects)

        # Extract scaled target data and the scaler itself
        train_targets, val_targets, target_scaler = self._get_targets_and_scaler(
            train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset
        )

        # Compute the pre-computed features
        train_pre_computed, val_pre_computed = self._get_pre_computed_features(model=model, train_data=train_data,
                                                                               val_data=val_data)

        # Create data generators
        train_gen = DownstreamDataGenerator(data=train_data, targets=train_targets, pre_computed=train_pre_computed,
                                            subjects=combined_dataset.get_subjects_dict(train_subjects))
        val_gen = DownstreamDataGenerator(data=val_data, targets=val_targets, pre_computed=val_pre_computed,
                                          subjects=combined_dataset.get_subjects_dict(val_subjects))

        # Create data loaders
        train_loader = DataLoader(dataset=train_gen, batch_size=self.train_config["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return train_loader, val_loader, target_scaler

    def _load_test_data_loader(self, *, model, test_subjects, combined_dataset, target_scaler):
        # Extract input data
        test_data = combined_dataset.get_data(subjects=test_subjects)

        # Extract scaled targets
        test_targets = combined_dataset.get_targets(subjects=test_subjects)
        test_targets = target_scaler.transform(test_targets)

        # Compute the pre-computed features
        test_pre_computed = model.pre_compute(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(self._device)
                           for dataset_name, data in test_data.items()})
        test_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                  for pre_comp in test_pre_computed)

        # Create data generators
        test_gen = DownstreamDataGenerator(data=test_data, targets=test_targets, pre_computed=test_pre_computed,
                                           subjects=combined_dataset.get_subjects_dict(test_subjects))

        # Create data loader
        test_loader = DataLoader(dataset=test_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return test_loader

    def _get_pre_computed_features(self, *, model, train_data, val_data):
        # Perform pre-computing of features
        train_pre_computed = model.pre_compute(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(self._device)
                           for dataset_name, data in train_data.items()})
        val_pre_computed = model.pre_compute(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(self._device)
                           for dataset_name, data in val_data.items()})

        # Send pre-computed features to cpu
        train_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                   for pre_comp in train_pre_computed)
        val_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                 for pre_comp in val_pre_computed)

        return train_pre_computed, val_pre_computed

    def _get_targets_and_scaler(self, *, train_subjects, val_subjects, combined_dataset):
        # Extract target data
        train_targets = combined_dataset.get_targets(subjects=train_subjects)
        val_targets = combined_dataset.get_targets(subjects=val_subjects)

        # Fit scaler and scale
        target_scaler = get_target_scaler(self.scaler_config["target"]["name"],
                                          **self.scaler_config["target"]["kwargs"])
        target_scaler.fit(train_targets)

        train_targets = target_scaler.transform(train_targets)
        val_targets = target_scaler.transform(val_targets)

        return train_targets, val_targets, target_scaler

    # -------------
    # Method for making and preparing model
    # -------------
    def _make_model(self):
        """Method for defining a model with RBP as the first layer"""
        # Maybe add number of time steps
        mts_config = copy.deepcopy(self.dl_architecture_config)
        if "num_time_steps" in mts_config["kwargs"] and mts_config["kwargs"]["num_time_steps"] is None:
            mts_config["kwargs"]["num_time_steps"] = self.shared_pre_processing_config["num_time_steps"]

        # Define model
        return MainRBPModel.from_config(
            rbp_config=self.rbp_config,
            mts_config=mts_config,
            discriminator_config=None if self.domain_discriminator_config is None
            else self.domain_discriminator_config["discriminator"]
        ).to(self._device)

    @staticmethod
    def _fit_channel_systems(model, channel_systems):
        model.fit_channel_systems(tuple(channel_systems.values()))

    def _fit_cmmn_layers(self, *, model, train_data, channel_systems):
        model.fit_psd_barycenters(data=train_data, channel_systems=channel_systems,
                                  sampling_freq=self.shared_pre_processing_config["resample"])
        model.fit_monge_filters(data=train_data, channel_systems=channel_systems)

    @staticmethod
    def _fit_cmmn_layers_test_data(*, model, test_data, channel_systems):
        for name, eeg_data in test_data.items():
            if name not in model.cmmn_fitted_channel_systems:
                # As long as the channel systems for the test data are present in 'channel_systems', this works
                # fine. Redundant channel systems is not a problem
                model.fit_monge_filters(data={name: eeg_data}, channel_systems=channel_systems)

    # -------------
    # Main method for running the cross validation experiment
    # -------------
    def run_experiment(self):
        print(f"Running on device: {self._device}")

        # -----------------
        # Load data and extract some details
        # -----------------
        combined_dataset = self._load_data()

        # Get some dataset details
        dataset_details = self._extract_dataset_details(combined_dataset)

        subjects = dataset_details["subjects"]
        channel_systems = dataset_details["channel_systems"]
        channel_name_to_index = dataset_details["channel_name_to_index"]

        # -----------------
        # Create folder for storing results, config file etc.
        # -----------------
        os.mkdir(self._results_path)

        # -----------------
        # Make subject split
        # -----------------
        folds = self._make_subject_split(subjects)

        # -----------------
        # Run cross validation
        # -----------------
        if self._config["cv_method"] == "normal":
            self.run_cross_validation(
                folds=folds, channel_systems=channel_systems, channel_name_to_index=channel_name_to_index,
                combined_dataset=combined_dataset
            )
        elif self._config["cv_method"] == "inverted":
            self.run_inverted_cross_validation(
                folds=folds, channel_systems=channel_systems, channel_name_to_index=channel_name_to_index,
                combined_dataset=combined_dataset
            )
        else:
            raise ValueError

    # -------------
    # Methods for exploration of why cross dataset DL is difficult
    # -------------
    def initial_hidden_layer_distributions(self):
        """Method which investigates the distribution of a hidden layer of the various EEG datasets"""
        print(f"Running on device: {self._device}")

        # -----------------
        # Load data and extract some details
        # -----------------
        combined_dataset = self._load_data()

        # Get some dataset details
        dataset_details = self._extract_dataset_details(combined_dataset)

        dataset_subjects = dataset_details["subjects"]
        channel_systems = dataset_details["channel_systems"]
        channel_name_to_index = dataset_details["channel_name_to_index"]

        # -----------------
        # Define model
        # -----------------
        print("Defining model...")
        # Filter some warnings from Voronoi split
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            model = self._make_model()

        # Fit channel systems
        self._fit_channel_systems(model=model, channel_systems=channel_systems)

        # (Maybe) fit the CMMN layers of RBP  todo: must check with and without CMMN
        # if model.any_rbp_cmmn_layers:
        #     self._fit_cmmn_layers(model=model, train_data=combined_dataset.get_data(train_subjects),
        #                           channel_systems=channel_systems)
        # -----------------
        # Split on dataset level
        # -----------------
        folds = get_data_split(split="SplitOnDataset", dataset_subjects=dataset_subjects).folds

        # Get the data loaders (the targets are not important, may change a little here in a future refactoring)
        data_loaders = {fold[0].dataset_name:  # todo
                        self._load_test_data_loader(
                            model=model, test_subjects=fold, combined_dataset=combined_dataset,
                            target_scaler=get_target_scaler(scaler="NoScaler")
                        ) for fold in folds}

        # -----------------
        # Compute features
        # -----------------
        latent_features = {
            dataset_name: self._extract_all_latent_features(model=model, data_loader=data_loader, device=self._device,
                                                            channel_name_to_index=channel_name_to_index)
            for dataset_name, data_loader in data_loaders.items()
        }

        # Compute distances between the latent distributions
        distances = self._compute_distribution_distances(data=latent_features,
                                                         distance_measures=self._config["distance_measures"])

        # Convert to dataframe
        features_df = self._latent_dict_features_to_dataframe(latent_features)

        # -----------------
        # Compute UMAP plots
        # -----------------
        umap = UMAP(n_components=2)
        umap_data = umap.fit_transform(features_df[[col_name for col_name in features_df.columns
                                                    if col_name != "dataset_name"]]).T

        # Selecting colors
        colormap = matplotlib.colormaps.get_cmap(self._config["colormap"])
        colors = colormap(numpy.linspace(start=0, stop=1, num=len(latent_features)))

        # Loop through all datasets
        for col, dataset_name in zip(colors, latent_features):
            indices = features_df["dataset_name"] == dataset_name
            pyplot.scatter(umap_data[0][indices], umap_data[1][indices], marker='o', label=dataset_name,
                           color=col)

        # Plot cosmetics
        fontsize = 15
        pyplot.xlabel("Dimension 1", fontsize=fontsize)
        pyplot.ylabel("Dimension 2", fontsize=fontsize)
        pyplot.title("UMAP")
        pyplot.legend()

        # -----------------
        # Plot distances
        # -----------------
        for metric, distance_matrix in distances.items():
            pyplot.figure()
            seaborn.heatmap(distance_matrix, annot=True, fmt=".3f")
            pyplot.title(metric)

        pyplot.show()

    @staticmethod
    def _compute_distribution_distances(data, distance_measures):
        """
        Examples
        --------
        >>> my_x1 = torch.tensor([[0, 1], [2, 1], [2, -1], [0, -1]], dtype=torch.float)
        >>> my_x2 = torch.tensor([[1, 0], [3, 0], [3, -2], [1, -2], [2, -1]], dtype=torch.float)
        >>> my_distances = Experiment._compute_distribution_distances({"d1": my_x1, "d2": my_x2},
        ...                                                           ("centroid_l2", "average_l2_to_centroid"))
        >>> my_distances["centroid_l2"]  # doctest: +NORMALIZE_WHITESPACE
                       d1        d2
        dataset
        d1       0.000000  1.414214
        d2       1.414214  0.000000
        >>> my_distances["average_l2_to_centroid"]  # doctest: +NORMALIZE_WHITESPACE
                       d1        d2
        dataset
        d1       1.414214  1.707107
        d2       1.648528  1.131371
        """
        distances = dict()
        for distance_measure in distance_measures:
            distances_single_metric = {dataset: [] for dataset in data}
            for features_j in data.values():
                for dataset_i, features_i in data.items():
                    distances_single_metric[dataset_i].append(
                        compute_distribution_distance(metric=distance_measure, x1=features_i, x2=features_j)
                    )

            df = pandas.DataFrame.from_dict(distances_single_metric)
            df["dataset"] = tuple(data.keys())
            df.set_index("dataset", inplace=True)
            distances[distance_measure] = df
        return distances

    @staticmethod
    def _latent_dict_features_to_dataframe(latent_features):
        """
        Examples
        --------
        >>> _ = torch.manual_seed(2)
        >>> my_features = {"d1": torch.rand((2, 64)), "d3": torch.rand((3, 64)), "d2": torch.rand((1, 64))}
        >>> Experiment._latent_dict_features_to_dataframe(my_features)  # doctest: +NORMALIZE_WHITESPACE
            dataset_name        v0        v1  ...       v61       v62       v63
        0           d1  0.614695  0.381013  ...  0.072670  0.646266  0.980437
        1           d1  0.944122  0.492143  ...  0.141569  0.321714  0.840315
        2           d3  0.013947  0.061767  ...  0.722807  0.688880  0.075720
        3           d3  0.823534  0.679069  ...  0.031638  0.116090  0.781444
        4           d3  0.564110  0.829796  ...  0.206841  0.694941  0.774273
        5           d2  0.324212  0.210616  ...  0.162639  0.147315  0.668140
        <BLANKLINE>
        [6 rows x 65 columns]
        """

        features = numpy.array(torch.cat(tuple(latent_features.values()), dim=0))
        _dataset_names = tuple([dataset_name]*feature.shape[0] for dataset_name, feature in latent_features.items())
        dataset_names = tuple(itertools.chain(*_dataset_names))

        # Create dataframe
        features_dict: Dict[str, Any] = {f"v{i}": feature for i, feature in enumerate(numpy.transpose(features))}
        return pandas.DataFrame.from_dict({"dataset_name": dataset_names, **features_dict})

    @staticmethod
    def _extract_all_latent_features(model, data_loader, device, channel_name_to_index):
        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for x, pre_computed, _, _ in data_loader:
                # Strip the dictionaries for 'ghost tensors'
                x = strip_tensors(x)
                pre_computed = [strip_tensors(pre_comp) for pre_comp in pre_computed]

                # Send data to correct device
                x = tensor_dict_to_device(x, device=device)
                pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device) for pre_comp in pre_computed)

                # Feature extraction
                outputs.append(
                    model.extract_latent_features(x, pre_computed=pre_computed,
                                                  channel_name_to_index=channel_name_to_index).to("cpu")
                )

        return torch.cat(outputs, dim=0)

    # -------------
    # Properties
    # -------------
    # Shortcuts to sub-part of the config file
    @property
    def train_config(self):
        return self._config["Training"]

    @property
    def datasets_config(self):
        return self._config["Datasets"]

    @property
    def subject_split_config(self):
        return self._config["SubjectSplit"]

    @property
    def sub_groups_config(self):
        return self._config["SubGroups"]

    @property
    def rbp_config(self):
        return self._config["Varied Numbers of Channels"]["RegionBasedPooling"]

    @property
    def dl_architecture_config(self):
        return self._config["DL Architecture"]

    @property
    def domain_discriminator_config(self):
        return self._config["DomainDiscriminator"]

    @property
    def scaler_config(self):
        return self._config["Scalers"]

    @property
    def shared_pre_processing_config(self):
        """Get the dict of the pre-processing config file which contains all shared pre-processing configurations"""
        return self._pre_processing_config["general"]
