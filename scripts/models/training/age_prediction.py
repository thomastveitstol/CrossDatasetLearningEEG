"""
Script primarily made for age prediction using a single DL model only. No pre-training employed. Using RBP.

    - Remember to use t-SNE and color-coding the domains, as in "Unsupervised Domain Adaptation by Backpropagation" by
    Ganin and Lempitsky 2015
    - Remember to use unsupervised domain adaptation which do not require access to the target data during training
    (e.g. 'de-learning' as part of the learning is not accepted)
"""
import os
import random
import warnings

import numpy
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

from cdl_eeg.data.combined_datasets import CombinedDatasets
from cdl_eeg.data.data_generators.data_generator import DownstreamDataGenerator
from cdl_eeg.data.data_split import get_data_split, leave_1_fold_out
from cdl_eeg.data.scalers.target_scalers import get_target_scaler
from cdl_eeg.models.losses import get_loss_function
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel
from cdl_eeg.models.metrics import Histories
from cdl_eeg.models.utils import tensor_dict_to_device, flatten_targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -----------------
    # Load settings (.yml file)
    # -----------------
    config_file = "age_prediction.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_file)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # -----------------
    # Load data
    # -----------------
    train_config = config["Training"]

    print("Loading data...")
    combined_dataset = CombinedDatasets.from_config(config=train_config["Datasets"], target=train_config["target"])

    # Extract some details of the datasets
    subjects = combined_dataset.dataset_subjects
    datasets = combined_dataset.datasets
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index() for dataset in datasets}

    # Split data into training and validation
    data_split = get_data_split(split=train_config["Data Split"]["name"], dataset_subjects=subjects,
                                **train_config["Data Split"]["kwargs"])
    folds = data_split.folds

    # -----------------
    # Perform (e.g.) cross validation
    # -----------------
    test_history = Histories(metrics=train_config["metrics"], name="test")
    for i, test_subjects in enumerate(folds):
        print(f"\nFold {i+1}/{len(folds)}")
        print(f"{' Training ':-^20}")
        # -----------------
        # Define model
        # -----------------
        # Filter some warnings from Voronoi split
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            model = MainRBPModel.from_config(config["Model"]).to(device)

        # -----------------
        # Train model
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
        target_scaler = get_target_scaler(train_config["Scalers"]["target"]["name"],
                                          **train_config["Scalers"]["target"]["kwargs"])
        target_scaler.fit(train_targets)

        train_targets = target_scaler.transform(train_targets)
        val_targets = target_scaler.transform(val_targets)

        # Perform pre-computing
        print("Pre-computing...")
        train_pre_computed = model.pre_compute(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                           for dataset_name, data in train_data.items()})
        val_pre_computed = model.pre_compute(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                           for dataset_name, data in val_data.items()})

        # Send to cpu
        train_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                   for pre_comp in train_pre_computed)
        val_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                 for pre_comp in val_pre_computed)

        # Create data generators
        train_gen = DownstreamDataGenerator(data=train_data, targets=train_targets, pre_computed=train_pre_computed)
        val_gen = DownstreamDataGenerator(data=val_data, targets=val_targets, pre_computed=val_pre_computed)

        # Create data loaders
        train_loader = DataLoader(dataset=train_gen, batch_size=train_config["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_gen, batch_size=train_config["batch_size"], shuffle=True)

        # Create optimiser and loss
        optimiser = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
        criterion = get_loss_function(loss=train_config["loss"])

        # Train model
        model.pre_train(train_loader=train_loader, val_loader=val_loader, metrics=train_config["metrics"],
                        criterion=criterion, optimiser=optimiser, num_epochs=train_config["num_epochs"],
                        verbose=train_config["verbose"], channel_name_to_index=channel_name_to_index,
                        device=device, target_scaler=target_scaler)

        # -----------------
        # Test model on test fold
        # -----------------
        print(f"\n{' Testing ':-^20}")
        model.eval()

        test_data = {name: torch.tensor(data, dtype=torch.float).to(device)
                     for name, data in combined_dataset.get_data(subjects=test_subjects).items()}
        test_targets = flatten_targets(combined_dataset.get_targets(subjects=test_subjects)).to(device)
        with torch.no_grad():
            # Perform pre-computing
            test_pre_computed = model.pre_compute(input_tensors=test_data)

            # Forward pass
            predictions = model(test_data, pre_computed=test_pre_computed, channel_name_to_index=channel_name_to_index)
            predictions = target_scaler.inv_transform(scaled_data=predictions)

            # Update test history
            test_history.store_batch_evaluation(y_pred=predictions, y_true=test_targets)
            test_history.on_epoch_end(verbose=train_config["verbose"])

    # -----------------
    # Print summary
    # -----------------
    for metric, performance in test_history.history.items():
        print(f"\n----- Metric: {metric.capitalize()} -----")

        print(f"\tMean: {numpy.mean(performance):.2f}")
        print(f"\tSTD: {numpy.std(performance):.2f}")

        print(f"\tAll folds: {tuple(round(p, 2) for p in performance)}")


if __name__ == "__main__":
    main()
