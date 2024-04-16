import copy
from typing import List

import enlighten
import numpy
import torch
import torch.nn as nn

from cdl_eeg.data.data_generators.data_generator import strip_tensors
from cdl_eeg.models.domain_adaptation.cmmn import ConvMMN
from cdl_eeg.models.domain_adaptation.domain_discriminators.getter import get_domain_discriminator
from cdl_eeg.models.main_models.main_rbp_model import reorder_subjects
from cdl_eeg.models.metrics import Histories, is_improved_model
from cdl_eeg.models.mts_modules.getter import get_mts_module
from cdl_eeg.models.utils import ReverseLayerF, tensor_dict_to_device, flatten_targets


# ----------------
# Convenient decorators  todo: move to some base .py file
# ----------------
def train_method(func):
    setattr(func, "_is_train_method", True)
    return func


# ----------------
# Classes
# ----------------
class MainFixedChannelsModel(nn.Module):
    """
    Main model when the number of input channels is fixed

    Examples
    --------
    >>> my_mts_kwargs = {"in_channels": 19, "num_classes": 7, "depth": 3}
    >>> my_dd_kwargs = {"hidden_units": (8, 4), "num_classes": 3}
    >>> my_cmmn_kwargs = {"kernel_size": 128}
    >>> MainFixedChannelsModel("InceptionNetwork", mts_module_kwargs=my_mts_kwargs, domain_discriminator="FCModule",
    ...                        domain_discriminator_kwargs=my_dd_kwargs, use_cmmn_layer=True,
    ...                        cmmn_kwargs=my_cmmn_kwargs)
    MainFixedChannelsModel(
      (_mts_module): InceptionNetwork(
        (_inception_modules): ModuleList(
          (0): _InceptionModule(
            (_input_conv): Conv1d(19, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
            (_conv_list): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(40,), stride=(1,), padding=same, bias=False)
              (1): Conv1d(32, 32, kernel_size=(20,), stride=(1,), padding=same, bias=False)
              (2): Conv1d(32, 32, kernel_size=(10,), stride=(1,), padding=same, bias=False)
            )
            (_max_pool): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
            (_conv_after_max_pool): Conv1d(19, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
            (_batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1-2): 2 x _InceptionModule(
            (_input_conv): Conv1d(128, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
            (_conv_list): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(40,), stride=(1,), padding=same, bias=False)
              (1): Conv1d(32, 32, kernel_size=(20,), stride=(1,), padding=same, bias=False)
              (2): Conv1d(32, 32, kernel_size=(10,), stride=(1,), padding=same, bias=False)
            )
            (_max_pool): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
            (_conv_after_max_pool): Conv1d(128, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
            (_batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (_shortcut_layers): ModuleList(
          (0): _ShortcutLayer(
            (_conv): Conv1d(19, 128, kernel_size=(1,), stride=(1,), padding=same)
            (_batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (_fc_layer): Linear(in_features=128, out_features=7, bias=True)
      )
      (_domain_discriminator): FCModule(
        (_model): ModuleList(
          (0): Linear(in_features=128, out_features=8, bias=True)
          (1): Linear(in_features=8, out_features=4, bias=True)
          (2): Linear(in_features=4, out_features=3, bias=True)
        )
      )
    )
    """

    def __init__(self, mts_module, mts_module_kwargs, domain_discriminator, domain_discriminator_kwargs, use_cmmn_layer,
                 cmmn_kwargs):
        """
        Initialise

        Parameters
        ----------
        mts_module : str
        mts_module_kwargs : dict[str, typing.Any]
        domain_discriminator : str, optional
        domain_discriminator_kwargs: dict[str, typing.Any] | None
        use_cmmn_layer : bool
        cmmn_kwargs : dict[str, typing.Any] | None
        """
        super().__init__()

        # ----------------
        # (Maybe) create CMMN layer
        # ----------------
        self._cmmn_layer = None if not use_cmmn_layer else ConvMMN(**cmmn_kwargs)

        # ----------------
        # Create MTS module
        # ----------------
        self._mts_module = get_mts_module(mts_module_name=mts_module, **mts_module_kwargs)

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
    def from_config(cls, mts_config, discriminator_config, cmmn_config):
        """
        Initialise from config file

        Parameters
        ----------
        mts_config : dict[str, typing.Any]
        discriminator_config : dict[str, typing.Any] | None
        cmmn_config : dict[str, typing.Any]
            todo: this may be merged with the MTS config
        """
        use_cmmn_layer = cmmn_config["use_cmmn_layer"]
        return cls(mts_module=mts_config["model"],
                   mts_module_kwargs=mts_config["kwargs"],
                   domain_discriminator=None if discriminator_config is None else discriminator_config["name"],
                   domain_discriminator_kwargs=None if discriminator_config is None else discriminator_config["kwargs"],
                   use_cmmn_layer=cmmn_config["use_cmmn_layer"],
                   cmmn_kwargs=None if not use_cmmn_layer else cmmn_config["kwargs"])

    # ---------------
    # Methods for forward propagation
    # ---------------
    def forward(self, x, use_domain_discriminator=False):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
        use_domain_discriminator : bool
            Boolean to indicate if the domain disciminator should be used as well as the downstream model (True) or not
            (False)

        Returns
        -------
        torch.Tensor
            Outputs of MTS module without applying a final activation function

        Examples
        --------
        >>> my_mts_kwargs = {"in_channels": 23, "num_classes": 11, "depth": 3}
        >>> my_dd_kwargs = {"hidden_units": (8, 4), "num_classes": 3}
        >>> my_cmmn_kwargs = {"kernel_size": 64}
        >>> my_model = MainFixedChannelsModel("InceptionNetwork", mts_module_kwargs=my_mts_kwargs,
        ...                                   domain_discriminator="FCModule", domain_discriminator_kwargs=my_dd_kwargs,
        ...                                   use_cmmn_layer=True, cmmn_kwargs=my_cmmn_kwargs)
        >>> my_model(torch.rand(size=(10, 23, 300))).size()
        torch.Size([10, 11])

        If the domain discriminator is used, its output will be the last of two in a tuple of torch tensors

        >>> my_outs = my_model(torch.rand(size=(10, 23, 300)), use_domain_discriminator=True)
        >>> my_outs[0].shape, my_outs[1].shape
        (torch.Size([10, 11]), torch.Size([10, 3]))
        """
        # (Maybe) concatenate all tensors. This should be possible, as this class should ony be used with a fixed number
        # of input channels
        if isinstance(x, dict):
            x = torch.cat(tuple(x.values()), dim=0)

        # Normalise  todo: make optional
        x = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-8)

        # If no domain discriminator is used, just run the normal forward method
        if not use_domain_discriminator:
            return self._mts_module(x)

        # ----------------
        # Extract latent features
        # ----------------
        x = self.extract_latent_features(x)

        # ----------------
        # Pass through both the classifier and domain discriminator
        # ----------------
        # Adding a gradient reversal layer to the features passed to domain discriminator
        # todo: I think alpha can be set to 1 without loss of generality, as long as the weighing in the loss is varied
        gradient_reversed_x = ReverseLayerF.apply(x, 1.)

        return (self._mts_module.classify_latent_features(x),
                self._domain_discriminator(gradient_reversed_x))  # type: ignore[misc]

    def extract_latent_features(self, x):
        """Method for extracting latent features"""
        # (Maybe) concatenate all tensors. This should be possible, as this class should ony be used with a fixed number
        # of input channels
        if isinstance(x, dict):
            x = torch.cat(tuple(x.values()), dim=0)

        # Run through MTS module and return
        return self._mts_module.extract_latent_features(x)

    # ----------------
    # Methods for fitting CMMN layer
    # ----------------
    def fit_psd_barycenters(self, data, sampling_freq):
        if self._cmmn_layer is None:
            raise RuntimeError("Cannot fit PSD barycenters of the CMMN layers, when none is used")

        self._cmmn_layer.fit_psd_barycenters(data=data, sampling_freq=sampling_freq)

    def fit_monge_filters(self, data):
        if self._cmmn_layer is None:
            raise RuntimeError("Cannot fit monge filters of the CMMN layers, when none is used")

        self._cmmn_layer.fit_monge_filters(data=data, is_psds=False)

    # ---------------
    # Methods for training and testing
    # ---------------
    @classmethod
    def get_available_training_methods(cls):
        """Get all training methods available for the class. The training method must be decorated by @train_method to
        be properly registered"""
        # Get all train methods
        train_methods: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a training method
            if callable(attribute) and getattr(attribute, "_is_train_method", False):
                train_methods.append(method)

        # Convert to tuple and return
        return tuple(train_methods)

    def train_model(self, method, **kwargs):
        """Method for training"""
        if method not in self.get_available_training_methods():
            raise ValueError(f"The training method {method} was not recognised. The available ones are: "
                             f"{self.get_available_training_methods()}")

        # Train
        return getattr(self, method)(**kwargs)

    @train_method
    def downstream_training(self, *, train_loader, val_loader, test_loader=None, metrics, main_metric, num_epochs,
                            classifier_criterion, optimiser, device, prediction_activation_function=None,
                            verbose=True, target_scaler=None, sub_group_splits, sub_groups_verbose):
        """
        Method for normal downstream training

        todo: a lot is copied from RBP

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
        test_loader : torch.utils.data.DataLoader, optional
        metrics : str | tuple[str, ...]
        main_metric : str
        num_epochs : int
        classifier_criterion : cdl_eeg.models.losses.CustomWeightedLoss
        optimiser : torch.optim.Optimizer
        device : torch.device
        prediction_activation_function : typing.Callable | None
        verbose : bool
        target_scaler : cdl_eeg.data.scalers.target_scalers.TargetScalerBase, optional
        sub_group_splits
        sub_groups_verbose

        Returns
        -------
        tuple[Histories, Histories, Histories | None]
                    Training and validation histories
        """
        # Defining histories objects
        train_history = Histories(metrics=metrics, splits=sub_group_splits)
        val_history = Histories(metrics=metrics, name="val", splits=sub_group_splits)
        test_history = None if test_loader is None else Histories(metrics=metrics, name="test", splits=sub_group_splits)

        # ---------------
        # Fit model
        # ---------------
        best_metrics = None
        best_model_state = {k: v.cpu() for k, v in self.state_dict().items()}
        for epoch in range(num_epochs):
            # Start progress bar
            pbar = enlighten.Counter(total=int(numpy.ceil(len(train_loader.dataset) / train_loader.batch_size)),
                                     desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            # ---------------
            # Training
            # ---------------
            self.train()
            for x, y, subject_indices in train_loader:
                # Strip the dictionaries for 'ghost tensors'
                x = strip_tensors(x)
                y = strip_tensors(y)

                # Extract subjects and correct the ordering
                subjects = reorder_subjects(order=tuple(x.keys()),
                                            subjects=train_loader.dataset.get_subjects_from_indices(subject_indices))

                # Send data to correct device
                x = tensor_dict_to_device(x, device=device)
                y = flatten_targets(y).to(device)

                # Forward pass
                output = self(x, use_domain_discriminator=False)

                # Compute loss
                optimiser.zero_grad()
                loss = classifier_criterion(output, y, subjects=subjects)
                loss.backward()
                optimiser.step()

                # Update train history
                with torch.no_grad():
                    y_pred = torch.clone(output)
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)

                    # (Maybe) re-scale targets and predictions before computing metrics
                    if target_scaler is not None:
                        y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                        y = target_scaler.inv_transform(scaled_data=y)
                    train_history.store_batch_evaluation(y_pred=y_pred, y_true=y, subjects=subjects)

                # Update progress bar
                pbar.update()

            # Finalise epoch for train history object
            train_history.on_epoch_end(verbose=verbose, verbose_sub_groups=sub_groups_verbose)

            # ---------------
            # Validation
            # ---------------
            self.eval()
            with torch.no_grad():
                for x, y, subject_indices in val_loader:
                    # Strip the dictionaries for 'ghost tensors'
                    x = strip_tensors(x)
                    y = strip_tensors(y)

                    # Extract subjects and correct the ordering
                    subjects = reorder_subjects(
                        order=tuple(x.keys()),
                        subjects=val_loader.dataset.get_subjects_from_indices(subject_indices)
                    )

                    # Send data to correct device
                    x = tensor_dict_to_device(x, device=device)
                    y = flatten_targets(y).to(device)

                    # Forward pass
                    y_pred = self(x, use_domain_discriminator=False)

                    # Update validation history
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)

                    # (Maybe) re-scale targets and predictions before computing metrics
                    if target_scaler is not None:
                        y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                        y = target_scaler.inv_transform(scaled_data=y)
                    val_history.store_batch_evaluation(y_pred=y_pred, y_true=y, subjects=subjects)

                # Finalise epoch for validation history object
                val_history.on_epoch_end(verbose=verbose, verbose_sub_groups=sub_groups_verbose)

            # ----------------
            # (Maybe) testing
            # ----------------
            if test_loader is not None:
                with torch.no_grad():
                    for x, y, subject_indices in test_loader:
                        # Strip the dictionaries for 'ghost tensors'
                        x = strip_tensors(x)
                        y = strip_tensors(y)

                        # Extract subjects and correct the ordering
                        subjects = reorder_subjects(
                            order=tuple(x.keys()),
                            subjects=test_loader.dataset.get_subjects_from_indices(subject_indices)
                        )

                        # Send data to correct device
                        x = tensor_dict_to_device(x, device=device)
                        y = flatten_targets(y).to(device)

                        # Forward pass
                        y_pred = self(x, use_domain_discriminator=False)

                        # Update test history
                        if prediction_activation_function is not None:
                            y_pred = prediction_activation_function(y_pred)

                        # (Maybe) re-scale targets and predictions before computing metrics
                        if target_scaler is not None:
                            y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                            y = target_scaler.inv_transform(scaled_data=y)
                        test_history.store_batch_evaluation(y_pred=y_pred, y_true=y,  # type: ignore[union-attr]
                                                            subjects=subjects)

                    # Finalise epoch for validation history object
                    test_history.on_epoch_end(verbose=verbose,  # type: ignore[union-attr]
                                              verbose_sub_groups=sub_groups_verbose)

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

    @train_method
    def domain_discriminator_training(
            self, *, train_loader, val_loader, test_loader=None, metrics, main_metric, num_epochs, classifier_criterion,
            optimiser, discriminator_criterion, discriminator_weight, discriminator_metrics, device,
            prediction_activation_function=None, verbose=True, target_scaler=None, sub_group_splits, sub_groups_verbose
    ):
        # Defining histories objects
        train_history = Histories(metrics=metrics, splits=sub_group_splits)
        val_history = Histories(metrics=metrics, name="val", splits=sub_group_splits)
        test_history = None if test_loader is None else Histories(metrics=metrics, name="test", splits=sub_group_splits)

        dd_train_history = Histories(metrics=discriminator_metrics, name="dd", splits=None)
        dd_val_history = Histories(metrics=discriminator_metrics, name="val_dd", splits=None)

        # ---------------
        # Fit model
        # ---------------
        best_metrics = None
        best_model_state = {k: v.cpu() for k, v in self.state_dict().items()}
        for epoch in range(num_epochs):
            # Start progress bar
            pbar = enlighten.Counter(total=int(numpy.ceil(len(train_loader.dataset) / train_loader.batch_size)),
                                     desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            # ---------------
            # Training
            # ---------------
            self.train()
            for x, y, subject_indices in train_loader:
                # Strip the dictionaries for 'ghost tensors'
                x = strip_tensors(x)
                y = strip_tensors(y)

                # Extract subjects and correct the ordering
                subjects = reorder_subjects(order=tuple(x.keys()),
                                            subjects=train_loader.dataset.get_subjects_from_indices(subject_indices))

                # Send data to correct device
                x = tensor_dict_to_device(x, device=device)
                y = flatten_targets(y).to(device)

                # Forward pass
                classifier_output, discriminator_output = self(x, use_domain_discriminator=True)

                # Compute dataset belonging (targets for discriminator)
                discriminator_targets = train_loader.dataset.get_dataset_indices_from_subjects(
                    subjects=subjects).to(device)

                # Compute loss and optimise
                optimiser.zero_grad()
                loss = (classifier_criterion(classifier_output, y, subjects=subjects)
                        + discriminator_weight * discriminator_criterion(discriminator_output, discriminator_targets,
                                                                         subjects=subjects))
                loss.backward()
                optimiser.step()

                # Update train history
                with torch.no_grad():
                    y_pred = torch.clone(classifier_output)
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)

                    # (Maybe) re-scale targets and predictions before computing metrics
                    if target_scaler is not None:
                        y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                        y = target_scaler.inv_transform(scaled_data=y)
                    train_history.store_batch_evaluation(y_pred=y_pred, y_true=y, subjects=subjects)

                    # Domain discriminator metrics
                    dd_train_history.store_batch_evaluation(y_pred=discriminator_output, y_true=discriminator_targets,
                                                            subjects=subjects)

                # Update progress bar
                pbar.update()

            # Finalise epoch for train history object
            train_history.on_epoch_end(verbose=verbose, verbose_sub_groups=sub_groups_verbose)
            dd_train_history.on_epoch_end(verbose=verbose)

            # ---------------
            # Validation
            # ---------------
            self.eval()
            with torch.no_grad():
                for x, y, subject_indices in val_loader:
                    # Strip the dictionaries for 'ghost tensors'
                    x = strip_tensors(x)
                    y = strip_tensors(y)

                    # Extract subjects and correct the ordering
                    subjects = reorder_subjects(
                        order=tuple(x.keys()),
                        subjects=val_loader.dataset.get_subjects_from_indices(subject_indices)
                    )

                    # Send data to correct device
                    x = tensor_dict_to_device(x, device=device)
                    y = flatten_targets(y).to(device)

                    # Forward pass
                    y_pred, discriminator_output = self(x, use_domain_discriminator=True)

                    # Compute dataset belonging (targets for discriminator)
                    discriminator_targets = val_loader.dataset.get_dataset_indices_from_subjects(
                        subjects=subjects).to(device)

                    # Update validation history
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)

                    # (Maybe) re-scale targets and predictions before computing metrics
                    if target_scaler is not None:
                        y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                        y = target_scaler.inv_transform(scaled_data=y)
                    val_history.store_batch_evaluation(y_pred=y_pred, y_true=y, subjects=subjects)

                    # Domain discriminator metrics
                    dd_val_history.store_batch_evaluation(y_pred=discriminator_output, y_true=discriminator_targets,
                                                          subjects=subjects)

                # Finalise epoch for validation history object
                val_history.on_epoch_end(verbose=verbose, verbose_sub_groups=sub_groups_verbose)
                dd_val_history.on_epoch_end(verbose=verbose)

            # ----------------
            # (Maybe) testing
            # ----------------
            if test_loader is not None:
                with torch.no_grad():
                    for x, y, subject_indices in test_loader:
                        # Strip the dictionaries for 'ghost tensors'
                        x = strip_tensors(x)
                        y = strip_tensors(y)

                        # Extract subjects and correct the ordering
                        subjects = reorder_subjects(
                            order=tuple(x.keys()),
                            subjects=test_loader.dataset.get_subjects_from_indices(subject_indices)
                        )

                        # Send data to correct device
                        x = tensor_dict_to_device(x, device=device)
                        y = flatten_targets(y).to(device)

                        # Forward pass
                        y_pred = self(x, use_domain_discriminator=False)

                        # Update test history
                        if prediction_activation_function is not None:
                            y_pred = prediction_activation_function(y_pred)

                        # (Maybe) re-scale targets and predictions before computing metrics
                        if target_scaler is not None:
                            y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                            y = target_scaler.inv_transform(scaled_data=y)
                        test_history.store_batch_evaluation(y_pred=y_pred, y_true=y,  # type: ignore[union-attr]
                                                            subjects=subjects)

                    # Finalise epoch for validation history object
                    test_history.on_epoch_end(verbose=verbose,  # type: ignore[union-attr]
                                              verbose_sub_groups=sub_groups_verbose)

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
        return train_history, val_history, test_history, dd_train_history, dd_val_history

    def test_model(self, *, data_loader, metrics, device, prediction_activation_function=None, verbose=True,
                   target_scaler=None, sub_group_splits, sub_groups_verbose):
        # Defining histories objects
        history = Histories(metrics=metrics, name="test", splits=sub_group_splits)

        # No gradients needed
        self.eval()
        with torch.no_grad():
            for x, y, subject_indices in data_loader:
                # Strip the dictionaries for 'ghost tensors'
                x = strip_tensors(x)
                y = strip_tensors(y)

                # Send data to correct device
                x = tensor_dict_to_device(x, device=device)
                y = flatten_targets(y).to(device)

                # Forward pass
                y_pred = self(x, use_domain_discriminator=False)

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
            history.on_epoch_end(verbose=verbose, verbose_sub_groups=sub_groups_verbose)

        return history

    # ---------------
    # Properties
    # ---------------
    @property
    def has_domain_discriminator(self) -> bool:
        """Indicates if the model has a domain discriminator for domain adversarial learning (True) or not (False)"""
        return self._domain_discriminator is not None

    @property
    def has_cmmn_layer(self) -> bool:  # todo: inconsistent property name with respect to the RBP version
        """Boolean indicating if the model uses a CMMN layer (True) or not (False)"""
        return self._cmmn_layer is not None

    @property
    def cmmn_fitted_channel_systems(self):
        if not self.has_cmmn_layer:
            raise RuntimeError(f"{type(self).__name__} has not property 'cmmn_fitted_channel_systems' when no CMMN "
                               f"layer is used")
        return self._cmmn_layer.fitted_monge_filters  # type: ignore[union-attr]
