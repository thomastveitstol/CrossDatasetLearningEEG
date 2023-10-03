"""
Script for plotting t-SNE after SSL pre-training

Used for debugging purposes only, and for me to see how I want the pipeline to work from start until end (what works,
what does not)
"""
import os
import warnings

import torch
import torch.nn as nn
from torch import optim
import yaml
from torch.utils.data import DataLoader

from cdl_eeg.data.combined_datasets import LoadDetails, CombinedDatasets
from cdl_eeg.data.data_generators.data_generator import SelfSupervisedDataGenerator
from cdl_eeg.data.data_split import KFoldDataSplit
from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel, tensor_dict_to_device
from cdl_eeg.models.region_based_pooling.region_based_pooling import RBPDesign, RBPPoolType
from cdl_eeg.models.transformations.frequency_slowing import FrequencySlowing
from cdl_eeg.models.transformations.utils import UnivariateUniform


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -----------------
    # Load settings (.yml file)
    # -----------------
    config_file = "tsne_debug.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_file)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # -----------------
    # Read RBP designs
    # -----------------
    designs_config = config["RBP Designs"]
    rbp_designs = []
    total_num_regions = 0
    for name, design in designs_config.items():
        rbp_designs.append(
            RBPDesign(pooling_type=RBPPoolType(design["pooling_type"]),
                      pooling_methods=design["pooling_methods"],
                      pooling_methods_kwargs=design["pooling_methods_kwargs"],
                      split_methods=design["split_methods"],
                      split_methods_kwargs=design["split_methods_kwargs"],
                      num_designs=design["num_designs"])
        )

        num_regions = design["pooling_methods_kwargs"]["num_regions"]  # todo: should be specified by split instead
        if isinstance(num_regions, int):
            total_num_regions += num_regions * design["num_designs"]
        else:
            total_num_regions += sum(num_regions) * design["num_designs"]

    # -----------------
    # Make model
    # -----------------
    # Read configuration file
    mts_design = config["MTS Module"]
    mts_design["kwargs"]["in_channels"] = total_num_regions

    # Define model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        model = MainRBPModel(mts_module=mts_design["model"], mts_module_kwargs=mts_design["kwargs"],
                             rbp_designs=tuple(rbp_designs)).to(device)

    # -----------------
    # Load data
    # -----------------
    load_details = []
    datasets = []
    subjects = dict()
    channel_name_to_index = dict()
    for dataset_name, dataset_details in config["Datasets"].items():
        # Get dataset
        dataset = get_dataset(dataset_name)
        datasets.append(dataset)
        dataset_subjects = dataset.get_subject_ids()[:dataset_details["num_subjects"]]
        subjects[dataset_name] = dataset_subjects
        channel_name_to_index[dataset_name] = dataset.channel_name_to_index()

        # Construct loading details
        load_details.append(
            LoadDetails(subject_ids=dataset_subjects, time_series_start=dataset_details["time_series_start"],
                        num_time_steps=dataset_details["num_time_steps"])
        )

    # Load all data
    combined_dataset = CombinedDatasets(tuple(datasets), load_details=tuple(load_details))

    # -----------------
    # Split data
    # -----------------
    # Split the data
    conf_data_split = config["Data Split"]
    data_split = KFoldDataSplit(num_folds=conf_data_split["num_folds"], dataset_subjects=subjects,
                                seed=conf_data_split["seed"]).folds

    # Split into training and validation splits
    train_subjects = data_split[0]  # todo
    val_subjects = data_split[-1]

    # -----------------
    # Define pretext task
    # -----------------
    config_pretext = config["Pretext Task"]
    slowing_distribution = UnivariateUniform(lower=config_pretext["lower"], upper=config_pretext["upper"])
    transformation = FrequencySlowing(slowing_distribution=slowing_distribution)
    pretext_task = config_pretext["task"]

    # -----------------
    # Perform pre-training
    # -----------------
    # Fit channel systems
    model.fit_channel_systems(tuple(data.channel_system for data in datasets))

    # Extract numpy array data
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
    config_training = config["Pre-Training"]
    train_loader = DataLoader(dataset=train_gen, batch_size=config_training["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_gen, batch_size=config_training["batch_size"], shuffle=True)

    # Create optimiser and loss
    optimiser = optim.Adam(model.parameters(), lr=config_training["learning_rate"])
    criterion = nn.MSELoss(reduction="mean")

    # Pre-train
    print("Pre-training...")
    model.pre_train(train_loader=train_loader, val_loader=val_loader, metrics="regression", criterion=criterion,
                    optimiser=optimiser, num_epochs=config_training["num_epochs"], verbose=config_training["verbose"],
                    channel_name_to_index=channel_name_to_index, device=device)

    # -----------------
    # t-SNE
    # -----------------
    load_details = []  # TODO: reloading
    datasets = []
    subjects = dict()
    channel_name_to_index = dict()
    for dataset_name, dataset_details in config["t-SNE Data"].items():
        # Get dataset
        dataset = get_dataset(dataset_name)
        datasets.append(dataset)
        dataset_subjects = dataset.get_subject_ids()[:dataset_details["num_subjects"]]
        subjects[dataset_name] = dataset_subjects
        channel_name_to_index[dataset_name] = dataset.channel_name_to_index()

        # Construct loading details
        load_details.append(
            LoadDetails(subject_ids=dataset_subjects, time_series_start=dataset_details["time_series_start"],
                        num_time_steps=dataset_details["num_time_steps"])
        )

    # Load all data
    combined_dataset = CombinedDatasets(tuple(datasets), load_details=tuple(load_details),
                                        target=config["t-SNE Details"]["target"])

    # 'Split' the data  todo
    data_split = KFoldDataSplit(num_folds=1, dataset_subjects=subjects, seed=config["t-SNE Details"]["seed"]).folds

    # Split into training and validation splits
    subjects = data_split[0]

    # Extract input data
    data = combined_dataset.get_data(subjects=subjects)
    targets = combined_dataset.get_targets(subjects=subjects)

    # Perform pre-computing
    print("Pre-computing...")
    pre_computed = model.pre_compute(input_tensors={dataset_name: torch.tensor(x, dtype=torch.float).to(device)
                                                    for dataset_name, x in data.items()})

    # Create t-SNE object
    print("Fitting t-SNE...")
    tsne_data = model.fit_tsne(input_tensors={dataset_name: torch.tensor(x, dtype=torch.float).to(device)
                                              for dataset_name, x in data.items()},
                               channel_name_to_index=channel_name_to_index,
                               pre_computed=pre_computed)

    # todo
    import numpy
    healthy = tsne_data[numpy.where(targets["rockhill"] < 0.5)].T
    parkinsons = tsne_data[numpy.where(targets["rockhill"] > 0.5)].T

    print("Plotting t-SNE...")
    print(tsne_data.shape)
    from matplotlib import pyplot
    pyplot.scatter(healthy[0], healthy[1], marker='o', s=50)
    pyplot.scatter(parkinsons[0], parkinsons[1], marker='o', s=50)
    pyplot.title('t-SNE Plot')
    pyplot.xlabel('Dimension 1')
    pyplot.ylabel('Dimension 2')

    pyplot.show()


if __name__ == "__main__":
    main()
