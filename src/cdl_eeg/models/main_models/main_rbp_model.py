import copy

import enlighten
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from cdl_eeg.data.data_generators.data_generator import strip_tensors
from cdl_eeg.models.metrics import Histories
from cdl_eeg.models.mts_modules.getter import get_mts_module
from cdl_eeg.models.region_based_pooling.region_based_pooling import RegionBasedPooling, RBPDesign, RBPPoolType
from cdl_eeg.models.utils import tensor_dict_to_device, flatten_targets


class MainRBPModel(nn.Module):
    """
    (In early stages of development)

    Main model supporting use of RBP. That is, this class uses RBP as a first layer, followed by an MTS module

    PS: Merges channel splits by concatenation
    """

    def __init__(self, *, mts_module, mts_module_kwargs, rbp_designs, normalise_region_representations=True):
        """
        Initialise

        Parameters
        ----------
        mts_module : str
        mts_module_kwargs : dict[str, typing.Any]
        rbp_designs : tuple[cdl_eeg.models.region_based_pooling.region_based_pooling.RBPDesign, ...]
        normalise_region_representations : bool
        """
        super().__init__()

        # -----------------
        # Create RBP layer
        # -----------------
        self._region_based_pooling = RegionBasedPooling(rbp_designs)
        self._normalise_region_representations = normalise_region_representations

        # ----------------
        # Create MTS module
        # ----------------
        self._mts_module = get_mts_module(
            mts_module_name=mts_module, **{"in_channels": self._region_based_pooling.num_regions, **mts_module_kwargs}
        )

    @classmethod
    def from_config(cls, rbp_config, mts_config):
        # -----------------
        # Read RBP designs
        # -----------------
        designs_config = copy.deepcopy(rbp_config["RBPDesigns"])
        rbp_designs = []
        for name, design in designs_config.items():
            rbp_designs.append(
                RBPDesign(pooling_type=RBPPoolType(design["pooling_type"]),
                          pooling_methods=design["pooling_methods"],
                          pooling_methods_kwargs=design["pooling_methods_kwargs"],
                          split_methods=design["split_methods"],
                          split_methods_kwargs=design["split_methods_kwargs"],
                          num_designs=design["num_designs"])
            )

        # -----------------
        # Read MTS design
        # -----------------
        # Read configuration file
        mts_design = copy.deepcopy(mts_config)

        # -----------------
        # Make model
        # -----------------
        return cls(mts_module=mts_design["model"], mts_module_kwargs=mts_design["kwargs"],
                   rbp_designs=tuple(rbp_designs),
                   normalise_region_representations=rbp_config["normalise_region_representations"])

    def pre_compute(self, input_tensors):
        """
        Pre-compute

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]

        Returns
        -------
        tuple[dict[str, torch.Tensor], ...]
        """
        return self._region_based_pooling.pre_compute(input_tensors)

    def forward(self, input_tensors, *, channel_name_to_index, pre_computed=None):
        """
        Forward method

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
        channel_name_to_index : dict[str, int]
        pre_computed : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
            Outputs of MTS module without applying a final activation function
        """
        # Pass through RBP layer
        x = self._region_based_pooling(input_tensors, channel_name_to_index=channel_name_to_index,
                                       pre_computed=pre_computed)

        # Merge by concatenation
        x = torch.cat(x, dim=1)

        # Maybe normalise region representations
        if self._normalise_region_representations:
            x = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-8)

        # Pass through MTS module and return
        return self._mts_module(x)

    def extract_latent_features(self, input_tensors, *, channel_name_to_index, pre_computed=None,
                                method="default_latent_feature_extraction"):
        """Method for extracting latent features"""
        # Input check
        if not self._mts_module.supports_latent_feature_extraction():
            raise ValueError(f"The MTS module {type(self._mts_module).__name__} does not support latent feature "
                             f"extraction")

        # Pass through RBP layer
        x = self._region_based_pooling(input_tensors, channel_name_to_index=channel_name_to_index,
                                       pre_computed=pre_computed)
        # Merge by concatenation
        x = torch.cat(x, dim=1)

        # Pass through MTS module and return
        return self._mts_module.extract_latent_features(x, method=method)

    # ----------------
    # Methods for fitting channel systems
    # ----------------
    def fit_channel_system(self, channel_system):
        self._region_based_pooling.fit_channel_system(channel_system)

    def fit_channel_systems(self, channel_systems):
        self._region_based_pooling.fit_channel_systems(channel_systems)

    # ----------------
    # Methods for training and testing
    # ----------------
    def train_model(self, *, train_loader, val_loader, metrics, num_epochs, criterion, optimiser, device,
                    channel_name_to_index, prediction_activation_function=None, verbose=True, target_scaler=None):
        """
        Method for training

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
        metrics : str | tuple[str, ...]
        num_epochs : int
        criterion : nn.modules.loss._Loss
        optimiser : torch.optim.Optimizer
        device : torch.device
        channel_name_to_index : dict[str, dict[str, int]]
        prediction_activation_function : typing.Callable | None
        verbose : bool
        target_scaler : cdl_eeg.data.scalers.target_scalers.TargetScalerBase, optional

        Returns
        -------
        tuple[Histories, Histories]
            Training and validation histories
        """
        # todo: may want to think more on memory usage

        # Defining histories objects
        train_history = Histories(metrics=metrics)
        val_history = Histories(metrics=metrics, name="val")

        # ------------------------
        # Fit model
        # ------------------------
        for epoch in range(num_epochs):
            # Start progress bar
            pbar = enlighten.Counter(total=int(len(train_loader) / train_loader.batch_size + 1),
                                     desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            # ----------------
            # Training
            # ----------------
            self.train()
            for x_train, train_pre_computed, y_train, subject_indices in train_loader:
                # todo: Only works for train loaders with this specific __getitem__ return
                # Strip the dictionaries for 'ghost tensors'
                x_train = strip_tensors(x_train)
                y_train = strip_tensors(y_train)
                train_pre_computed = [strip_tensors(pre_comp) for pre_comp in train_pre_computed]

                # Send data to correct device
                x_train = tensor_dict_to_device(x_train, device=device)
                y_train = flatten_targets(y_train).to(device)
                train_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device)
                                           for pre_comp in train_pre_computed)

                # Forward pass
                output = self(x_train, pre_computed=train_pre_computed, channel_name_to_index=channel_name_to_index)

                # Compute loss
                loss = criterion(output, y_train)
                loss.backward()
                optimiser.step()

                # Update train history
                # todo: see if you can optimise more here
                with torch.no_grad():
                    y_pred = torch.clone(output)
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)

                    # (Maybe) re-scale targets and predictions before computing metrics
                    if target_scaler is not None:
                        y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                        y_train = target_scaler.inv_transform(scaled_data=y_train)
                    train_history.store_batch_evaluation(
                        y_pred=y_pred, y_true=y_train,
                        subjects=train_loader.dataset.get_subjects_from_indices(subject_indices)
                    )

                # Update progress bar
                pbar.update()

            # Finalise epoch for train history object
            train_history.on_epoch_end(verbose=verbose)

            # ----------------
            # Validation
            # ----------------
            self.eval()
            with torch.no_grad():
                for x_val, val_pre_computed, y_val, val_subject_indices in val_loader:
                    # Strip the dictionaries for 'ghost tensors'
                    x_val = strip_tensors(x_val)
                    y_val = strip_tensors(y_val)
                    val_pre_computed = tuple(strip_tensors(pre_comp) for pre_comp in val_pre_computed)

                    # Send data to correct device
                    x_val = tensor_dict_to_device(x_val, device=device)
                    y_val = flatten_targets(y_val).to(device)
                    val_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device)
                                             for pre_comp in val_pre_computed)

                    # Forward pass  todo: why did I use .clone() in the PhD course tasks?
                    y_pred = self(x_val, pre_computed=val_pre_computed, channel_name_to_index=channel_name_to_index)

                    # Update validation history
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)

                    # (Maybe) re-scale targets and predictions before computing metrics
                    if target_scaler is not None:
                        y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                        y_val = target_scaler.inv_transform(scaled_data=y_val)
                    val_history.store_batch_evaluation(
                        y_pred=y_pred, y_true=y_val,
                        subjects=val_loader.dataset.get_subjects_from_indices(val_subject_indices)
                    )

                # Finalise epoch for validation history object
                val_history.on_epoch_end(verbose=verbose)

        return train_history, val_history

    def test_model(self, *, data_loader, metrics, device, channel_name_to_index, prediction_activation_function=None,
                   verbose=True, target_scaler=None):
        # Defining histories objects
        history = Histories(metrics=metrics, name="test")

        # No gradients needed
        self.eval()
        with torch.no_grad():
            for x, pre_computed, y, subject_indices in data_loader:
                # Strip the dictionaries for 'ghost tensors'
                x = strip_tensors(x)
                y = strip_tensors(y)
                pre_computed = tuple(strip_tensors(pre_comp) for pre_comp in pre_computed)

                # Send data to correct device
                x = tensor_dict_to_device(x, device=device)
                y = flatten_targets(y).to(device)
                pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device) for pre_comp in pre_computed)

                # Forward pass
                y_pred = self(x, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index)

                # Update validation history
                if prediction_activation_function is not None:
                    y_pred = prediction_activation_function(y_pred)

                # (Maybe) re-scale targets and predictions before computing metrics
                if target_scaler is not None:
                    y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                    y = target_scaler.inv_transform(scaled_data=y)
                history.store_batch_evaluation(
                    y_pred=y_pred, y_true=y,
                    subjects=data_loader.dataset.get_subjects_from_indices(subject_indices)
                )

            # Finalise epoch for validation history object
            history.on_epoch_end(verbose=verbose)

        return history

    # ----------------
    # Methods for t-SNE
    # ----------------
    def fit_tsne(self, input_tensors, *, channel_name_to_index, pre_computed=None, n_components=2):
        """
        Method for fitting t-SNE

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
        channel_name_to_index : dict[str, dict[str, int]]
        pre_computed : tuple[dict[str, typing.Any], ...], optional
        n_components : int
            Number of components for the t-SNE object

        Returns
        -------
        numpy.ndarray
        """
        with torch.no_grad():
            # ---------------
            # Forward pass
            # ---------------
            # Pass through RBP layer
            x = self._region_based_pooling(input_tensors, channel_name_to_index=channel_name_to_index,
                                           pre_computed=pre_computed)

            # Merge by concatenation
            x = torch.cat(x, dim=1)

            # Pass through MTS module, and extract features
            x = self._mts_module(x, return_features=True)

            # ---------------
            # Create and fit t-SNE
            # ---------------
            # Send features to cpu
            x = x.cpu()

        # Create and fit t-SNE
        tsne = TSNE(n_components=n_components)
        return tsne.fit_transform(x)
