import copy
import os
import random
import warnings
from typing import Any, Optional

import torch
from torch import optim
from torch.utils.data import DataLoader

from cdl_eeg.data.combined_datasets import CombinedDatasets
from cdl_eeg.data.data_generators.data_generator import DownstreamDataGenerator
from cdl_eeg.data.data_split import get_data_split, leave_1_fold_out
from cdl_eeg.data.scalers.target_scalers import get_target_scaler
from cdl_eeg.models.losses import CustomWeightedLoss, get_activation_function
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel
from cdl_eeg.models.metrics import save_discriminator_histories_plots, save_histories_plots, Histories
from cdl_eeg.models.utils import tensor_dict_to_device


class Experiment:
    """
    Class for running a single cross validation experiment
    """

    __slots__ = "_config", "_pre_processing_config", "_results_path", "_device"

    def __init__(self, config, pre_processing_config, results_path):
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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def run_cross_validation(self, *, folds, channel_systems, channel_name_to_index, combined_dataset):
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
            self._run_single_fold(
                train_subjects=train_subjects, val_subjects=val_subjects, test_subjects=test_subjects,
                results_path=fold_path, channel_systems=channel_systems, channel_name_to_index=channel_name_to_index,
                combined_dataset=combined_dataset
            )

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
            prediction_activation_function=get_activation_function(self.train_config["prediction_activation_function"])
        )

        # -----------------
        # Test model (but only if continuous testing was not used)
        # -----------------
        test_estimate: Optional[Histories]
        if self.train_config["continuous_testing"]:
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
                channel_name_to_index=channel_name_to_index, device=self._device, target_scaler=target_scaler
            )
        else:
            test_estimate = None

        # -----------------
        # Save results
        # -----------------
        self._save_results(histories=histories, test_estimate=test_estimate, results_path=results_path)

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
            train_history, val_history, test_history, dd_train_history, dd_val_history = histories

            train_history.save_prediction_history(history_name="train_history", path=results_path)
            val_history.save_prediction_history(history_name="val_history", path=results_path)
            if test_history is not None:
                test_history.save_prediction_history(history_name="test_history", path=results_path)

            dd_train_history.save_prediction_history(history_name="dd_train_history", path=results_path)
            dd_val_history.save_prediction_history(history_name="dd_val_history", path=results_path)

            # Save domain discriminator metrics plots
            save_discriminator_histories_plots(path=results_path, histories=(dd_train_history, dd_val_history))

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
        return MainRBPModel.from_config(
            rbp_config=self.rbp_config,
            mts_config=self.dl_architecture_config,
            discriminator_config=None if self.domain_discriminator_config["DomainDiscriminator"] is None
            else self.domain_discriminator_config["discriminator"]
        ).to(self._device)

    @staticmethod
    def _fit_channel_systems(model, channel_systems):
        model.fit_channel_systems(channel_systems)

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
                model.fit_monge_filters(data=test_data, channel_systems=channel_systems)

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
        self.run_cross_validation(
            folds=folds, channel_systems=channel_systems, channel_name_to_index=channel_name_to_index,
            combined_dataset=combined_dataset
        )

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


def run_experiment(config, pre_processing_config, results_path):
    """
    Function for running an experiment

    Parameters
    ----------
    config : dict[str, typing.Any]
    pre_processing_config : dict[str, typing.Any]
        The config file used for pre-processing
    results_path : str
        Where to store the results

    Returns
    -------
    None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -----------------
    # Load data
    # -----------------
    train_config = config["Training"]

    print("Loading data...")
    combined_dataset = CombinedDatasets.from_config(config=config["Datasets"], target=train_config["target"])

    # -----------------
    # Create folder for storing results, config file etc.
    # -----------------
    os.mkdir(results_path)

    # -----------------
    # Extract dataset details and perform data split
    # -----------------
    # Extract some details of the datasets
    subjects = combined_dataset.dataset_subjects
    datasets = combined_dataset.datasets
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index() for dataset in datasets}

    # Split data into training and validation
    data_split = get_data_split(split=config["SubjectSplit"]["name"], dataset_subjects=subjects,
                                **config["SubjectSplit"]["kwargs"])
    folds = data_split.folds

    # -----------------
    # Perform (e.g.) cross validation
    # -----------------
    for i, test_subjects in enumerate(folds):
        print(f"\nFold {i + 1}/{len(folds)}")
        print(f"{' Training ':-^20}")

        # -----------------
        # Make folder for the current fold
        # -----------------
        fold_path = os.path.join(results_path, f"Fold_{i}")
        os.mkdir(fold_path)

        # -----------------
        # Define model
        # -----------------
        # Filter some warnings from Voronoi split
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            model = MainRBPModel.from_config(
                rbp_config=config["Varied Numbers of Channels"]["RegionBasedPooling"],
                mts_config=config["DL Architecture"],
                discriminator_config=None if config["DomainDiscriminator"] is None
                else config["DomainDiscriminator"]["discriminator"]
            ).to(device)

        # -----------------
        # Prepare everything
        # -----------------
        # Fit channel systems
        model.fit_channel_systems(tuple(data.channel_system for data in datasets))

        # Split into train and validation
        non_test_subjects = list(leave_1_fold_out(i=i, folds=folds))
        random.shuffle(non_test_subjects)
        num_subjects = len(non_test_subjects)

        split_idx = int(num_subjects * (1 - train_config["val_split"]))
        train_subjects = non_test_subjects[:split_idx]
        val_subjects = non_test_subjects[split_idx:]

        # Extract input data
        train_data = combined_dataset.get_data(subjects=train_subjects)
        val_data = combined_dataset.get_data(subjects=val_subjects)

        print({d_name: d.shape for d_name, d in train_data.items()})

        # Extract target data
        train_targets = combined_dataset.get_targets(subjects=train_subjects)
        val_targets = combined_dataset.get_targets(subjects=val_subjects)

        # Fit scaler and scale
        target_scaler = get_target_scaler(config["Scalers"]["target"]["name"], **config["Scalers"]["target"]["kwargs"])
        target_scaler.fit(train_targets)

        train_targets = target_scaler.transform(train_targets)
        val_targets = target_scaler.transform(val_targets)
        # todo: why are we scaling anything but training targets???

        # Perform pre-computing
        print("Pre-computing...")
        train_pre_computed = model.pre_compute(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                           for dataset_name, data in train_data.items()})
        val_pre_computed = model.pre_compute(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                           for dataset_name, data in val_data.items()})

        # Send pre-computed features to cpu
        train_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                   for pre_comp in train_pre_computed)
        val_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                 for pre_comp in val_pre_computed)

        # Create data generators
        train_gen = DownstreamDataGenerator(data=train_data, targets=train_targets, pre_computed=train_pre_computed,
                                            subjects=combined_dataset.get_subjects_dict(train_subjects))
        val_gen = DownstreamDataGenerator(data=val_data, targets=val_targets, pre_computed=val_pre_computed,
                                          subjects=combined_dataset.get_subjects_dict(val_subjects))

        # Create data loaders
        train_loader = DataLoader(dataset=train_gen, batch_size=train_config["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_gen, batch_size=train_config["batch_size"], shuffle=True)

        # (Maybe) fit the CMMN layers of RBP
        channel_systems = {dataset.name: dataset.channel_system for dataset in datasets}
        if model.any_rbp_cmmn_layers:
            model.fit_psd_barycenters(data=train_data, channel_systems=channel_systems,
                                      sampling_freq=pre_processing_config["general"]["resample"])
            model.fit_monge_filters(data=train_data, channel_systems=channel_systems)

        # Maybe repeat the above steps for the test data as well
        test_loader: Optional[DataLoader[Any]]
        if train_config["continuous_testing"]:
            test_data = combined_dataset.get_data(subjects=test_subjects)
            test_targets = combined_dataset.get_targets(subjects=test_subjects)
            test_targets = target_scaler.transform(test_targets)
            test_pre_computed = model.pre_compute(
                input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                               for dataset_name, data in test_data.items()})
            test_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                      for pre_comp in test_pre_computed)
            test_gen = DownstreamDataGenerator(data=test_data, targets=test_targets, pre_computed=test_pre_computed,
                                               subjects=combined_dataset.get_subjects_dict(test_subjects))
            test_loader = DataLoader(dataset=test_gen, batch_size=train_config["batch_size"], shuffle=True)

            # If there are unseen datasets by the CMMN layer, add them as well
            if model.any_rbp_cmmn_layers:
                for name, eeg_data in test_data.items():
                    if name not in train_data:
                        model.fit_monge_filters(data=test_data, channel_systems=channel_systems)
        else:
            test_loader = None

        # Create optimiser
        optimiser = optim.Adam(model.parameters(), lr=train_config["learning_rate"],
                               betas=(train_config["beta_1"], train_config["beta_2"]), eps=train_config["eps"])

        # Create loss
        if train_config["Loss"]["weighter"] is not None:
            train_config["Loss"]["weighter_kwargs"]["dataset_sizes"] = train_gen.dataset_sizes
        criterion = CustomWeightedLoss(**train_config["Loss"])

        # (Maybe) create optimiser and loss for domain discriminator
        discriminator_criterion: Optional[CustomWeightedLoss]
        if config["DomainDiscriminator"] is not None:
            if config["DomainDiscriminator"]["training"]["Loss"]["weighter"] is not None:
                config["DomainDiscriminator"]["training"]["Loss"]["weighter_kwargs"]["dataset_sizes"] = (
                    train_gen.dataset_sizes)
            discriminator_criterion = CustomWeightedLoss(**config["DomainDiscriminator"]["training"]["Loss"])
            discriminator_weight = config["DomainDiscriminator"]["training"]["lambda"]
            discriminator_metrics = config["DomainDiscriminator"]["training"]["metrics"]
        else:
            discriminator_criterion = None
            discriminator_weight = None
            discriminator_metrics = None

        # -----------------
        # Train model
        # -----------------
        histories = model.train_model(
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, metrics=train_config["metrics"],
            main_metric=train_config["main_metric"], classifier_criterion=criterion, optimiser=optimiser,
            discriminator_criterion=discriminator_criterion, discriminator_weight=discriminator_weight,
            discriminator_metrics=discriminator_metrics, num_epochs=train_config["num_epochs"],
            verbose=train_config["verbose"], channel_name_to_index=channel_name_to_index, device=device,
            target_scaler=target_scaler,
            prediction_activation_function=get_activation_function(train_config["prediction_activation_function"])
        )

        # -----------------
        # Save prediction histories
        # -----------------
        if config["DomainDiscriminator"] is None:
            train_history, val_history, test_history = histories

            train_history.save_prediction_history(history_name="train_history", path=fold_path)
            val_history.save_prediction_history(history_name="val_history", path=fold_path)
            if test_history is not None:
                test_history.save_prediction_history(history_name="test_history", path=fold_path)
        else:
            train_history, val_history, test_history, dd_train_history, dd_val_history = histories

            train_history.save_prediction_history(history_name="train_history", path=fold_path)
            val_history.save_prediction_history(history_name="val_history", path=fold_path)
            if test_history is not None:
                test_history.save_prediction_history(history_name="test_history", path=fold_path)

            dd_train_history.save_prediction_history(history_name="dd_train_history", path=fold_path)
            dd_val_history.save_prediction_history(history_name="dd_val_history", path=fold_path)

            save_discriminator_histories_plots(path=fold_path, histories=(dd_train_history, dd_val_history))

        # -----------------
        # Test model on test fold
        # -----------------
        print(f"\n{' Testing ':-^20}")
        model.eval()

        # Extract input and target data
        test_data = combined_dataset.get_data(subjects=test_subjects)
        test_targets = combined_dataset.get_targets(subjects=test_subjects)

        # Scale targets
        test_targets = target_scaler.transform(test_targets)

        with torch.no_grad():
            # Perform pre-computing
            test_pre_computed = model.pre_compute(
                input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                               for dataset_name, data in test_data.items()})

            # Send to cpu
            test_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                      for pre_comp in test_pre_computed)

            # Create data generator and loader
            test_gen = DownstreamDataGenerator(data=test_data, targets=test_targets, pre_computed=test_pre_computed,
                                               subjects=combined_dataset.get_subjects_dict(test_subjects))
            test_loader = DataLoader(dataset=test_gen, batch_size=train_config["batch_size"], shuffle=True)

            # Test model on test data
            test_estimate = model.test_model(
                data_loader=test_loader, metrics=train_config["metrics"], verbose=train_config["verbose"],
                channel_name_to_index=channel_name_to_index, device=device, target_scaler=target_scaler
            )

            # Save predictions
            test_estimate.save_prediction_history(history_name="test_estimate", path=fold_path)

        # -----------------
        # Save plots
        # -----------------
        save_histories_plots(path=fold_path, train_history=train_history, val_history=val_history,
                             test_estimate=test_estimate, test_history=test_history)
