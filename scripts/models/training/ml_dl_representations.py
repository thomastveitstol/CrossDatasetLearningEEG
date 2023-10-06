"""
Script for supervised learning using non-DL machine learning models on learned DL representations (from pretext tasks)

Used for debugging purposes only, and for me to see how I want the pipeline to work from start until end (what works,
what does not)
"""
import os
import warnings

import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import optim
from torch.utils.data import DataLoader

from cdl_eeg.data.combined_datasets import CombinedDatasets
from cdl_eeg.data.data_generators.data_generator import SelfSupervisedDataGenerator
from cdl_eeg.data.data_split import leave_1_fold_out, get_data_split
from cdl_eeg.data.scalers.target_scalers import get_target_scaler
from cdl_eeg.models.losses import get_loss_function
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel, tensor_dict_to_device, flatten_targets
from cdl_eeg.models.main_models.ml_models import get_ml_model
from cdl_eeg.models.transformations.frequency_slowing import FrequencySlowing
from cdl_eeg.models.transformations.utils import get_random_distribution


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -----------------
    # Load settings (.yml file)
    # -----------------
    config_file = "ml_debug.yml"
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
    # Load data for SSL pre-training
    # -----------------
    pre_train_config = config["Pre-Training"]

    # Load all data (no targets to load, as SSL)
    combined_dataset = CombinedDatasets.from_config(config=pre_train_config["Datasets"], target=None)

    # Extract some details of the datasets (needed later)
    subjects = combined_dataset.dataset_subjects
    datasets = combined_dataset.datasets
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index() for dataset in datasets}

    # Split data into training and validation
    data_split = get_data_split(split=pre_train_config["Data Split"]["name"], dataset_subjects=subjects,
                                **pre_train_config["Data Split"]["kwargs"])
    folds = data_split.folds

    train_subjects = leave_1_fold_out(i=-1, folds=folds)
    val_subjects = folds[-1]

    # -----------------
    # Define pretext task
    # -----------------
    config_pretext = pre_train_config["Pretext Task"]
    slowing_distribution = get_random_distribution(distribution=config_pretext["Distribution"]["distribution"],
                                                   **config_pretext["Distribution"]["kwargs"])
    transformation = FrequencySlowing(slowing_distribution=slowing_distribution,
                                      scale_details=config_pretext["Distribution"]["scale_details"])
    pretext_task = config_pretext["task"]

    # -----------------
    # Perform pre-training
    # -----------------
    print("======================================")
    print("============ Pre-Training ============")
    # Fit channel systems
    model.fit_channel_systems(tuple(data.channel_system for data in datasets))

    # Extract input data
    train_data = combined_dataset.get_data(subjects=train_subjects)
    val_data = combined_dataset.get_data(subjects=val_subjects)

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
    train_gen = SelfSupervisedDataGenerator(data=train_data, pre_computed=train_pre_computed,
                                            transformation=transformation, pretext_task=pretext_task)
    val_gen = SelfSupervisedDataGenerator(data=val_data, pre_computed=val_pre_computed, transformation=transformation,
                                          pretext_task=pretext_task)

    # Create data loaders
    train_loader = DataLoader(dataset=train_gen, batch_size=pre_train_config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_gen, batch_size=pre_train_config["batch_size"], shuffle=True)

    # Create optimiser and loss
    optimiser = optim.Adam(model.parameters(), lr=pre_train_config["learning_rate"])
    criterion = get_loss_function(loss=pre_train_config["loss"])

    # Pre-train
    print("Pre-training...")
    model.pre_train(train_loader=train_loader, val_loader=val_loader, metrics="regression", criterion=criterion,
                    optimiser=optimiser, num_epochs=pre_train_config["num_epochs"], verbose=pre_train_config["verbose"],
                    channel_name_to_index=channel_name_to_index, device=device)

    print("========= Pre-Training Done ==========")
    print("======================================")

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

    # Get latent features
    with torch.no_grad():
        train_features = model.extract_latent_features(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                           for dataset_name, data in train_data.items()},
            channel_name_to_index=channel_name_to_index,
            pre_computed=train_pre_computed
        ).cpu().numpy()

    # Train the classifier
    ml_model = get_ml_model(downstream_config["ML Model"]["name"], **downstream_config["ML Model"]["kwargs"])
    ml_model.fit(train_features, flatten_targets(train_targets).numpy())

    # Predict
    with torch.no_grad():
        val_features = model.extract_latent_features(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                           for dataset_name, data in val_data.items()},
            channel_name_to_index=channel_name_to_index,
            pre_computed=val_pre_computed
        ).cpu().numpy()
    predictions = ml_model.predict(val_features)

    print("----------------------")
    print(f"AUC: {roc_auc_score(flatten_targets(val_targets).numpy(), predictions)}")
    print(f"Accuracy: {accuracy_score(flatten_targets(val_targets).numpy(), predictions)}")


if __name__ == "__main__":
    main()
