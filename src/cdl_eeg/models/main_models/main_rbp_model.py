import copy
from typing import List, Dict

import enlighten
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from cdl_eeg.data.data_generators.data_generator import strip_tensors
from cdl_eeg.data.data_split import Subject
from cdl_eeg.models.domain_discriminators.getter import get_domain_discriminator
from cdl_eeg.models.metrics import Histories, is_improved_model
from cdl_eeg.models.mts_modules.getter import get_mts_module
from cdl_eeg.models.region_based_pooling.region_based_pooling import RegionBasedPooling, RBPDesign, RBPPoolType
from cdl_eeg.models.utils import tensor_dict_to_device, flatten_targets, ReverseLayerF


class MainRBPModel(nn.Module):
    """
    (In early stages of development)

    Main model supporting use of RBP. That is, this class uses RBP as a first layer, followed by an MTS module

    PS: Merges channel splits by concatenation
    """

    def __init__(self, *, mts_module, mts_module_kwargs, rbp_designs, normalise_region_representations=True,
                 domain_discriminator=None, domain_discriminator_kwargs=None):
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

        # ----------------
        # (Maybe) create domain discriminator
        # ----------------
        if domain_discriminator is None:
            self._domain_discriminator = None
        else:
            # Set kwargs to empty dict if none are passed
            domain_discriminator_kwargs = dict() if domain_discriminator_kwargs is None else domain_discriminator_kwargs

            # Need to get input features from MTS module
            domain_discriminator_kwargs["in_features"] = self._mts_module.latent_features_dim

            self._domain_discriminator = get_domain_discriminator(
                name=domain_discriminator, **domain_discriminator_kwargs
            )

    @classmethod
    def from_config(cls, rbp_config, mts_config, discriminator_config=None):
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
                   normalise_region_representations=rbp_config["normalise_region_representations"],
                   domain_discriminator=None if discriminator_config is None else discriminator_config["name"],
                   domain_discriminator_kwargs=None if discriminator_config is None else discriminator_config["kwargs"])

    # ----------------
    # Methods for forward pass and related
    # ----------------
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

    def _forward(self, input_tensors, *, channel_name_to_index, pre_computed=None):
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

    def extract_latent_features(self, input_tensors, *, channel_name_to_index, pre_computed=None):
        """Method for extracting latent features"""
        # Pass through RBP layer
        x = self._region_based_pooling(input_tensors, channel_name_to_index=channel_name_to_index,
                                       pre_computed=pre_computed)
        # Merge by concatenation
        x = torch.cat(x, dim=1)

        # Maybe normalise region representations
        if self._normalise_region_representations:
            x = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-8)

        # Pass through MTS module and return
        return self._mts_module.extract_latent_features(x)

    def forward(self, input_tensors, *, channel_name_to_index, pre_computed=None, use_domain_discriminator=False):
        # If no domain discriminator is used, just run the normal forward method
        if not use_domain_discriminator:
            return self._forward(input_tensors, channel_name_to_index=channel_name_to_index, pre_computed=pre_computed)

        # ----------------
        # Extract latent features
        # ----------------
        x = self.extract_latent_features(input_tensors, channel_name_to_index=channel_name_to_index,
                                         pre_computed=pre_computed)

        # ----------------
        # Pass through both the classifier and domain discriminator
        # ----------------
        # Adding a gradient reversal layer to the features passed to domain discriminator
        # todo: I think alpha can be set to 1 without loss of generality, as long as the weighing in the loss is varied
        gradient_reversed_x = ReverseLayerF.apply(x, 1.)

        return self._mts_module.classify_latent_features(x), self._domain_discriminator(gradient_reversed_x)

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
    def train_model(
            self, *, train_loader, val_loader, test_loader, metrics, main_metric, num_epochs, classifier_criterion,
            optimiser, discriminator_criterion=None, discriminator_weight=None, discriminator_metrics=None, device,
            channel_name_to_index, prediction_activation_function=None, verbose=True, target_scaler=None):
        if any(discriminator_arg is None for discriminator_arg
               in (discriminator_criterion, discriminator_weight)):
            return self._train_model(
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, metrics=metrics,
                main_metric=main_metric, num_epochs=num_epochs, criterion=classifier_criterion, optimiser=optimiser,
                device=device, channel_name_to_index=channel_name_to_index, verbose=verbose,
                target_scaler=target_scaler, prediction_activation_function=prediction_activation_function
            )
        else:
            return self._train_model_with_domain_adversarial_learning(
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, metrics=metrics,
                main_metric=main_metric, num_epochs=num_epochs, classifier_criterion=classifier_criterion,
                optimiser=optimiser, discriminator_criterion=discriminator_criterion,
                discriminator_weight=discriminator_weight, discriminator_metrics=discriminator_metrics, device=device,
                channel_name_to_index=channel_name_to_index,
                prediction_activation_function=prediction_activation_function, verbose=verbose,
                target_scaler=target_scaler
            )

    def _train_model_with_domain_adversarial_learning(
            self, *, train_loader, val_loader, test_loader=None, metrics, main_metric, num_epochs, classifier_criterion,
            optimiser, discriminator_criterion, discriminator_weight, discriminator_metrics, device,
            channel_name_to_index, prediction_activation_function=None, verbose=True, target_scaler=None):
        """Method for training with domain adversarial learning"""
        # Defining histories objects
        train_history = Histories(metrics=metrics)
        val_history = Histories(metrics=metrics, name="val")
        test_history = None if test_loader is None else Histories(metrics=metrics, name="test")

        dd_train_history = Histories(metrics=discriminator_metrics, name="dd")
        dd_val_history = Histories(metrics=discriminator_metrics, name="val_dd")

        # ------------------------
        # Fit model
        # ------------------------
        best_metrics = None
        best_model_state = {k: v.cpu() for k, v in self.state_dict().items()}
        for epoch in range(num_epochs):
            # Start progress bar
            pbar = enlighten.Counter(total=int(len(train_loader.dataset) / train_loader.batch_size + 1),
                                     desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            # ----------------
            # Training
            # ----------------
            self.train()
            for x_train, train_pre_computed, y_train, subject_indices in train_loader:
                # Strip the dictionaries for 'ghost tensors'
                x_train = strip_tensors(x_train)
                y_train = strip_tensors(y_train)
                train_pre_computed = [strip_tensors(pre_comp) for pre_comp in train_pre_computed]

                # Extract subjects and correct the ordering
                subjects = reorder_subjects(order=tuple(x_train.keys()),
                                            subjects=train_loader.dataset.get_subjects_from_indices(subject_indices))

                # Send data to correct device
                x_train = tensor_dict_to_device(x_train, device=device)
                y_train = flatten_targets(y_train).to(device)
                train_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device)
                                           for pre_comp in train_pre_computed)

                # Forward pass
                classifier_output, discriminator_output = self(x_train, pre_computed=train_pre_computed,
                                                               channel_name_to_index=channel_name_to_index,
                                                               use_domain_discriminator=True)

                # Compute dataset belonging (targets for discriminator)
                discriminator_targets = train_loader.dataset.get_dataset_indices_from_subjects(
                    subjects=subjects).to(device)

                # Compute loss
                loss = (classifier_criterion(classifier_output, y_train, subjects=subjects)
                        + discriminator_weight * discriminator_criterion(discriminator_output, discriminator_targets,
                                                                         subjects=subjects))

                # Optimise
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Update train histories
                # todo: see if you can optimise more here
                with torch.no_grad():
                    y_pred = torch.clone(classifier_output)
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)

                    # (Maybe) re-scale targets and predictions before computing metrics
                    if target_scaler is not None:
                        y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                        y_train = target_scaler.inv_transform(scaled_data=y_train)
                    train_history.store_batch_evaluation(y_pred=y_pred, y_true=y_train, subjects=subjects)

                    # Domain discriminator metrics
                    dd_train_history.store_batch_evaluation(y_pred=discriminator_output, y_true=discriminator_targets,
                                                            subjects=subjects)

                # Update progress bar
                pbar.update()

            # Finalise epoch for train history objects
            train_history.on_epoch_end(verbose=verbose)
            dd_train_history.on_epoch_end(verbose=verbose)

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

                    # Extract subjects and correct the ordering
                    val_subjects = reorder_subjects(
                        order=tuple(x_val.keys()),
                        subjects=val_loader.dataset.get_subjects_from_indices(val_subject_indices)
                    )

                    # Send data to correct device
                    x_val = tensor_dict_to_device(x_val, device=device)
                    y_val = flatten_targets(y_val).to(device)
                    val_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device)
                                             for pre_comp in val_pre_computed)

                    # Forward pass, getting both classifier and domain discriminator outputs
                    y_pred, discriminator_output = self(x_val, pre_computed=val_pre_computed,
                                                        channel_name_to_index=channel_name_to_index,
                                                        use_domain_discriminator=True)

                    # Compute dataset belonging (targets for discriminator)
                    discriminator_targets = val_loader.dataset.get_dataset_indices_from_subjects(
                        subjects=val_subjects).to(device)

                    # Update validation history
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)

                    # (Maybe) re-scale targets and predictions before computing metrics
                    if target_scaler is not None:
                        y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                        y_val = target_scaler.inv_transform(scaled_data=y_val)
                    val_history.store_batch_evaluation(y_pred=y_pred, y_true=y_val, subjects=val_subjects)

                    # Domain discriminator metrics
                    dd_val_history.store_batch_evaluation(y_pred=discriminator_output, y_true=discriminator_targets,
                                                          subjects=val_subjects)

                # Finalise epoch for validation history objects
                val_history.on_epoch_end(verbose=verbose)
                dd_val_history.on_epoch_end(verbose=verbose)

            # ----------------
            # (Maybe) testing
            # ----------------
            if test_loader is not None:
                with torch.no_grad():
                    for x_test, test_pre_computed, y_test, test_subject_indices in test_loader:
                        # Strip the dictionaries for 'ghost tensors'
                        x_test = strip_tensors(x_test)
                        y_test = strip_tensors(y_test)
                        test_pre_computed = tuple(strip_tensors(pre_comp) for pre_comp in test_pre_computed)

                        # Extract subjects and correct the ordering
                        test_subjects = reorder_subjects(
                            order=tuple(x_test.keys()),
                            subjects=test_loader.dataset.get_subjects_from_indices(test_subject_indices)
                        )

                        # Send data to correct device
                        x_test = tensor_dict_to_device(x_test, device=device)
                        y_test = flatten_targets(y_test).to(device)
                        test_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device)
                                                  for pre_comp in test_pre_computed)

                        # Forward pass
                        y_pred = self(x_test, pre_computed=test_pre_computed,
                                      channel_name_to_index=channel_name_to_index)

                        # Update test history
                        if prediction_activation_function is not None:
                            y_pred = prediction_activation_function(y_pred)

                        # (Maybe) re-scale targets and predictions before computing metrics
                        if target_scaler is not None:
                            y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                            y_test = target_scaler.inv_transform(scaled_data=y_test)
                        test_history.store_batch_evaluation(y_pred=y_pred, y_true=y_test, subjects=test_subjects)

                    # Finalise epoch for test history object
                    test_history.on_epoch_end(verbose=verbose)

            # ----------------
            # If this is the highest performing model, store it
            # ----------------
            if is_improved_model(old_metrics=best_metrics, new_metrics=val_history.newest_metrics,
                                 main_metric=main_metric):
                # Store the model on the cpu
                best_model_state = copy.deepcopy({k: v.cpu() for k, v in self.state_dict().items()})

                # Update the best metrics
                best_metrics = val_history.newest_metrics

            # Set the parameters back to those of the best model
        self.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})  # type: ignore[arg-type]

        # Return the histories
        return train_history, val_history, test_history, dd_train_history, dd_val_history

    def _train_model(self, *, train_loader, val_loader, test_loader=None, metrics, main_metric, num_epochs, criterion,
                     optimiser, device, channel_name_to_index, prediction_activation_function=None, verbose=True,
                     target_scaler=None):
        """
        Method for training

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
        test_loader : torch.utils.data.DataLoader
        metrics : str | tuple[str, ...]
        main_metric : str
            Main metric for model selection
        num_epochs : int
        criterion : cdl_eeg.models.losses.CustomWeightedLoss
        optimiser : torch.optim.Optimizer
        device : torch.device
        channel_name_to_index : dict[str, dict[str, int]]
        prediction_activation_function : typing.Callable | None
        verbose : bool
        target_scaler : cdl_eeg.data.scalers.target_scalers.TargetScalerBase, optional

        Returns
        -------
        tuple[Histories, Histories, Histories | None]
            Training and validation histories
        """
        # todo: may want to think more on memory usage

        # Defining histories objects
        train_history = Histories(metrics=metrics)
        val_history = Histories(metrics=metrics, name="val")
        test_history = None if test_loader is None else Histories(metrics=metrics, name="test")

        # ------------------------
        # Fit model
        # ------------------------
        best_metrics = None
        best_model_state = {k: v.cpu() for k, v in self.state_dict().items()}
        for epoch in range(num_epochs):
            # Start progress bar
            pbar = enlighten.Counter(total=int(len(train_loader.dataset) / train_loader.batch_size + 1),
                                     desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            # ----------------
            # Training
            # ----------------
            self.train()
            for x_train, train_pre_computed, y_train, subject_indices in train_loader:
                # Strip the dictionaries for 'ghost tensors'
                x_train = strip_tensors(x_train)
                y_train = strip_tensors(y_train)
                train_pre_computed = [strip_tensors(pre_comp) for pre_comp in train_pre_computed]

                # Extract subjects and correct the ordering
                subjects = reorder_subjects(order=tuple(x_train.keys()),
                                            subjects=train_loader.dataset.get_subjects_from_indices(subject_indices))

                # Send data to correct device
                x_train = tensor_dict_to_device(x_train, device=device)
                y_train = flatten_targets(y_train).to(device)
                train_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device)
                                           for pre_comp in train_pre_computed)

                # Forward pass
                output = self(x_train, pre_computed=train_pre_computed, channel_name_to_index=channel_name_to_index)

                # Compute loss
                optimiser.zero_grad()
                loss = criterion(output, y_train, subjects=subjects)
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
                    train_history.store_batch_evaluation(y_pred=y_pred, y_true=y_train, subjects=subjects)

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

                    # Extract subjects and correct the ordering
                    val_subjects = reorder_subjects(
                        order=tuple(x_val.keys()),
                        subjects=val_loader.dataset.get_subjects_from_indices(val_subject_indices)
                    )

                    # Send data to correct device
                    x_val = tensor_dict_to_device(x_val, device=device)
                    y_val = flatten_targets(y_val).to(device)
                    val_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device)
                                             for pre_comp in val_pre_computed)

                    # Forward pass
                    y_pred = self(x_val, pre_computed=val_pre_computed, channel_name_to_index=channel_name_to_index)

                    # Update validation history
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)

                    # (Maybe) re-scale targets and predictions before computing metrics
                    if target_scaler is not None:
                        y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                        y_val = target_scaler.inv_transform(scaled_data=y_val)
                    val_history.store_batch_evaluation(y_pred=y_pred, y_true=y_val, subjects=val_subjects)

                # Finalise epoch for validation history object
                val_history.on_epoch_end(verbose=verbose)

            # ----------------
            # (Maybe) testing
            # ----------------
            if test_loader is not None:
                with torch.no_grad():
                    for x_test, test_pre_computed, y_test, test_subject_indices in test_loader:
                        # Strip the dictionaries for 'ghost tensors'
                        x_test = strip_tensors(x_test)
                        y_test = strip_tensors(y_test)
                        test_pre_computed = tuple(strip_tensors(pre_comp) for pre_comp in test_pre_computed)

                        # Extract subjects and correct the ordering
                        test_subjects = reorder_subjects(
                            order=tuple(x_test.keys()),
                            subjects=test_loader.dataset.get_subjects_from_indices(test_subject_indices)
                        )

                        # Send data to correct device
                        x_test = tensor_dict_to_device(x_test, device=device)
                        y_test = flatten_targets(y_test).to(device)
                        test_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device)
                                                  for pre_comp in test_pre_computed)

                        # Forward pass
                        y_pred = self(x_test, pre_computed=test_pre_computed,
                                      channel_name_to_index=channel_name_to_index)

                        # Update validation history
                        if prediction_activation_function is not None:
                            y_pred = prediction_activation_function(y_pred)

                        # (Maybe) re-scale targets and predictions before computing metrics
                        if target_scaler is not None:
                            y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                            y_test = target_scaler.inv_transform(scaled_data=y_test)
                        test_history.store_batch_evaluation(y_pred=y_pred, y_true=y_test, subjects=test_subjects)

                    # Finalise epoch for test history object
                    test_history.on_epoch_end(verbose=verbose)

            # ----------------
            # If this is the highest performing model, as evaluated on the validation set, store it
            # ----------------
            if is_improved_model(old_metrics=best_metrics, new_metrics=val_history.newest_metrics,
                                 main_metric=main_metric):
                # Store the model on the cpu
                best_model_state = copy.deepcopy({k: v.cpu() for k, v in self.state_dict().items()})

                # Update the best metrics
                best_metrics = val_history.newest_metrics

        # Set the parameters back to those of the best model
        self.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})  # type: ignore[arg-type]

        # Return the histories
        return train_history, val_history, test_history

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


# ----------------
# Functions
# ----------------
def reorder_subjects(order, subjects):
    """
    Function for re-ordering subjects such that they align with how the input and target tensors are concatenated

    Parameters
    ----------
    order : tuple[str, ...]
        Ordering of the datasets
    subjects : tuple[Subject, ...]
        Subjects to re-order

    Returns
    -------
    tuple[Subject, ...]

    Examples
    --------
    >>> my_subjects = (Subject("P3", "D2"), Subject("P1", "D2"), Subject("P1", "D1"), Subject("P4", "D1"),
    ...                Subject("P2", "D2"))
    >>> reorder_subjects(order=("D1", "D2"), subjects=my_subjects)  # doctest: +NORMALIZE_WHITESPACE
    (Subject(subject_id='P1', dataset_name='D1', details=None),
     Subject(subject_id='P4', dataset_name='D1', details=None),
     Subject(subject_id='P3', dataset_name='D2', details=None),
     Subject(subject_id='P1', dataset_name='D2', details=None),
     Subject(subject_id='P2', dataset_name='D2', details=None))
    """
    subjects_dict: Dict[str, List[Subject]] = {dataset_name: [] for dataset_name in order}
    for subject in subjects:
        subjects_dict[subject.dataset_name].append(subject)

    # Convert to list
    corrected_subjects = []
    for subject_list in subjects_dict.values():
        corrected_subjects.extend(subject_list)

    # return as a tuple
    return tuple(corrected_subjects)
