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
from cdl_eeg.models.losses import CustomWeightedLoss
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel
from cdl_eeg.models.metrics import save_discriminator_histories_plots, save_histories_plots
from cdl_eeg.models.utils import tensor_dict_to_device


def run_experiment(config, results_path):
    """
    Function for running an experiment

    Parameters
    ----------
    config : dict[str, typing.Any]
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
        else:
            test_loader = None

        # Create optimiser
        optimiser = optim.Adam(model.parameters(), lr=train_config["learning_rate"],
                               betas=(train_config["beta_1"], train_config["beta_2"]), eps=train_config["eps"])

        # Create loss
        if train_config["Loss"]["weighter"] is not None:
            train_config["Loss"]["weighter"]["weighter_kwargs"]["dataset_sizes"] = train_gen.dataset_sizes
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
            target_scaler=target_scaler
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
