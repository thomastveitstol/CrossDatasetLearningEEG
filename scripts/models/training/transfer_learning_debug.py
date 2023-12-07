"""
Script for (non-SSL) transfer learning without freezing any weights

Used for debugging purposes only, and for me to see how I want the pipeline to work from start until end (what works,
what does not)
"""
import os
import warnings

import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

from cdl_eeg.data.combined_datasets import CombinedDatasets
from cdl_eeg.data.data_generators.data_generator import DownstreamDataGenerator
from cdl_eeg.data.data_split import leave_1_fold_out, get_data_split
from cdl_eeg.data.scalers.target_scalers import get_target_scaler
from cdl_eeg.models.losses import get_loss_function
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel, tensor_dict_to_device


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -----------------
    # Load settings (.yml file)
    # -----------------
    config_file = "tf_debug.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_file)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # -----------------
    # Define model
    # -----------------
    # Filter some warnings from Voronoi split
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        model = MainRBPModel.from_config(config["Model"]).to(device)

    # -----------------
    # Load data for pre-training
    # -----------------
    pre_train_config = config["Pre-Training"]

    # Load all data
    combined_dataset = CombinedDatasets.from_config(config=pre_train_config["Datasets"],
                                                    target=pre_train_config["target"])

    # Extract some details of the datasets (needed later)
    subjects = combined_dataset.dataset_subjects
    datasets = combined_dataset.datasets
    channel_name_to_index = combined_dataset.channel_name_to_index

    # Split data into training and validation
    data_split = get_data_split(split=pre_train_config["Data Split"]["name"],  dataset_subjects=subjects,
                                **pre_train_config["Data Split"]["kwargs"])
    folds = data_split.folds

    train_subjects = leave_1_fold_out(i=-1, folds=folds)
    val_subjects = folds[-1]

    # -----------------
    # Perform pre-training  todo: too long, make it a function or method instead
    # -----------------
    # Fit channel systems
    model.fit_channel_systems(tuple(data.channel_system for data in datasets))

    # Extract input data
    train_data = combined_dataset.get_data(subjects=train_subjects)
    val_data = combined_dataset.get_data(subjects=val_subjects)

    # Extract target data
    train_targets = combined_dataset.get_targets(subjects=train_subjects)
    val_targets = combined_dataset.get_targets(subjects=val_subjects)

    # Fit scaler and scale
    target_scaler = get_target_scaler(pre_train_config["Scalers"]["target"]["name"],
                                      **pre_train_config["Scalers"]["target"]["kwargs"])
    target_scaler.fit(train_targets)

    train_targets = target_scaler.transform(train_targets)
    val_targets = target_scaler.transform(val_targets)

    # Perform pre-computing
    print("Pre-computing...")
    train_pre_computed = model.pre_compute(input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                                                          for dataset_name, data in train_data.items()})
    val_pre_computed = model.pre_compute(input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
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
    train_loader = DataLoader(dataset=train_gen, batch_size=pre_train_config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_gen, batch_size=pre_train_config["batch_size"], shuffle=True)

    # Create optimiser and loss
    optimiser = optim.Adam(model.parameters(), lr=pre_train_config["learning_rate"])
    criterion = get_loss_function(loss=pre_train_config["loss"])

    # Pre-train model
    print("Pre-training...")
    model.train_model(train_loader=train_loader, val_loader=val_loader, metrics=pre_train_config["metrics"],
                      criterion=criterion, optimiser=optimiser, num_epochs=pre_train_config["num_epochs"],
                      verbose=pre_train_config["verbose"], channel_name_to_index=channel_name_to_index, device=device)

    # -----------------
    # Load data for downstream training and split data
    # -----------------
    downstream_config = config["Downstream Training"]

    # Load all data
    combined_dataset = CombinedDatasets.from_config(config=downstream_config["Datasets"],
                                                    target=downstream_config["target"])

    # Extract some details of the datasets (needed later)
    subjects = combined_dataset.dataset_subjects
    datasets = combined_dataset.datasets
    channel_name_to_index = combined_dataset.channel_name_to_index

    # Split data into training and validation
    data_split = get_data_split(split=downstream_config["Data Split"]["name"], dataset_subjects=subjects,
                                **downstream_config["Data Split"]["kwargs"])
    folds = data_split.folds

    train_subjects = leave_1_fold_out(i=-1, folds=folds)
    val_subjects = folds[-1]

    # -----------------
    # Perform downstream training
    # -----------------
    # Fit channel systems
    model.fit_channel_systems(tuple(data.channel_system for data in datasets))

    # Extract input data
    train_data = combined_dataset.get_data(subjects=train_subjects)
    val_data = combined_dataset.get_data(subjects=val_subjects)

    # Extract target data
    train_targets = combined_dataset.get_targets(subjects=train_subjects)
    val_targets = combined_dataset.get_targets(subjects=val_subjects)

    # Fit scaler and scale
    target_scaler = get_target_scaler(downstream_config["Scalers"]["target"]["name"],
                                      **downstream_config["Scalers"]["target"]["kwargs"])
    target_scaler.fit(train_targets)

    train_targets = target_scaler.transform(train_targets)
    val_targets = target_scaler.transform(val_targets)

    # Perform pre-computing
    print("Pre-computing...")
    train_pre_computed = model.pre_compute(input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                                                          for dataset_name, data in train_data.items()})
    val_pre_computed = model.pre_compute(input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
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
    train_loader = DataLoader(dataset=train_gen, batch_size=downstream_config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_gen, batch_size=downstream_config["batch_size"], shuffle=True)

    # Create optimiser and loss
    optimiser = optim.Adam(model.parameters(), lr=downstream_config["learning_rate"])
    criterion = get_loss_function(downstream_config["loss"])

    # Pre-train model
    print("Downstream training...")
    model.train_model(train_loader=train_loader, val_loader=val_loader, metrics=downstream_config["metrics"],
                      criterion=criterion, optimiser=optimiser, num_epochs=downstream_config["num_epochs"],
                      verbose=downstream_config["verbose"], channel_name_to_index=channel_name_to_index, device=device)


if __name__ == "__main__":
    main()
