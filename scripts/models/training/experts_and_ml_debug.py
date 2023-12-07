"""
Script for supervised learning using non-DL machine learning models on multiple learned DL representations (from pretext
tasks)

Used for debugging purposes only, and for me to see how I want the pipeline to work from start until end (what works,
what does not)
"""
import os
import warnings

import numpy
import torch
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import optim
from torch.utils.data import DataLoader

from cdl_eeg.data.combined_datasets import CombinedDatasets
from cdl_eeg.data.data_generators.data_generator import SelfSupervisedDataGenerator
from cdl_eeg.data.data_split import get_data_split, leave_1_fold_out
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
    config_file = "ex_debug.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_file)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # -----------------
    # Train all experts
    # -----------------
    expert_models = []
    for expert_name, expert_config in config["Experts"].items():
        print(f"Training Expert model: {expert_name}")

        # -----------------
        # Define model
        # -----------------
        # Filter some warnings from Voronoi split
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            model = MainRBPModel.from_config(expert_config["Model"]).to(device)

        # -----------------
        # Load data for training the expert
        # -----------------
        train_config = expert_config["Training"]

        # Load all data (no targets to load, as SSL)  todo: should not only be SSL
        # todo: The data loaded here may already be in memory. Find a way to make this more efficient
        print("Loading data...")
        combined_dataset = CombinedDatasets.from_config(config=train_config["Datasets"], target=None)

        # Extract some details of the datasets
        subjects = combined_dataset.dataset_subjects
        datasets = combined_dataset.datasets
        channel_name_to_index = {dataset.name: dataset.channel_name_to_index() for dataset in datasets}

        # Split data into training and validation
        data_split = get_data_split(split=train_config["Data Split"]["name"], dataset_subjects=subjects,
                                    **train_config["Data Split"]["kwargs"])
        folds = data_split.folds

        train_subjects = leave_1_fold_out(i=-1, folds=folds)
        val_subjects = folds[-1]

        # -----------------
        # Define pretext task
        # -----------------
        print("Defining pretext task...")
        config_pretext = train_config["Pretext Task"]
        slowing_distribution = get_random_distribution(distribution=config_pretext["Distribution"]["distribution"],
                                                       **config_pretext["Distribution"]["kwargs"])
        transformation = FrequencySlowing(slowing_distribution=slowing_distribution,
                                          scale_details=config_pretext["Distribution"]["scale_details"])
        pretext_task = config_pretext["task"]

        # -----------------
        # Train the expert
        # -----------------
        # Fit channel systems
        model.fit_channel_systems(tuple(data.channel_system for data in datasets))

        # Extract input data
        train_data = combined_dataset.get_data(subjects=train_subjects)
        val_data = combined_dataset.get_data(subjects=val_subjects)

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
        train_gen = SelfSupervisedDataGenerator(data=train_data, pre_computed=train_pre_computed,
                                                transformation=transformation, pretext_task=pretext_task)
        val_gen = SelfSupervisedDataGenerator(data=val_data, pre_computed=val_pre_computed,
                                              transformation=transformation,
                                              pretext_task=pretext_task)

        # Create data loaders
        train_loader = DataLoader(dataset=train_gen, batch_size=train_config["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_gen, batch_size=train_config["batch_size"], shuffle=True)

        # Create optimiser and loss
        optimiser = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
        criterion = get_loss_function(loss=train_config["loss"])

        print("Training expert...")
        model.train_model(train_loader=train_loader, val_loader=val_loader, metrics="regression", criterion=criterion,
                          optimiser=optimiser, num_epochs=train_config["num_epochs"], verbose=train_config["verbose"],
                          channel_name_to_index=channel_name_to_index, device=device)

        # Freeze parameters and add to experts
        for params in model.parameters():
            params.requires_grad = False  # todo: this should be a method of the class instead
        expert_models.append(model)

    # -----------------
    # Train downstream model
    # -----------------
    downstream_config = config["Downstream Training"]
    print(f"Training Downstream Model: {downstream_config['ML Model']['name']}")

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

    # Fit channel systems
    for model in expert_models:
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

    # Get latent features
    train_features = []
    for model in expert_models:
        with torch.no_grad():
            # Pre-compute
            train_pre_computed = model.pre_compute(
                input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                               for dataset_name, data in train_data.items()})

            # Compute latent features
            train_features.append(model.extract_latent_features(
                input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                               for dataset_name, data in train_data.items()},
                channel_name_to_index=channel_name_to_index,
                pre_computed=train_pre_computed
            ).cpu().numpy())

    # Train classifier
    print(f"Features shape: {numpy.concatenate(train_features, axis=1).shape}")
    ml_model = get_ml_model(downstream_config["ML Model"]["name"], **downstream_config["ML Model"]["kwargs"])
    ml_model.fit(numpy.concatenate(train_features, axis=1), flatten_targets(train_targets).numpy())

    # Predict
    val_features = []
    for model in expert_models:
        with torch.no_grad():
            # Pre-compute
            val_pre_computed = model.pre_compute(
                input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                               for dataset_name, data in val_data.items()})

            # Compute latent features
            val_features.append(model.extract_latent_features(
                input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                               for dataset_name, data in val_data.items()},
                channel_name_to_index=channel_name_to_index,
                pre_computed=val_pre_computed
            ).cpu().numpy())

    print(f"Val features shape: {numpy.concatenate(val_features, axis=1).shape}")
    predictions = ml_model.predict(numpy.concatenate(val_features, axis=1))

    print("----------------------")

    print(f"AUC: {roc_auc_score(flatten_targets(val_targets).numpy(), predictions)}")
    print(f"Accuracy: {accuracy_score(flatten_targets(val_targets).numpy(), predictions)}")


if __name__ == "__main__":
    main()
