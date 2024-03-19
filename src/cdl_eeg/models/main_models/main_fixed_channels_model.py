import warnings

import enlighten
import torch
import torch.nn as nn

from cdl_eeg.models.domain_adaptation.cmmn import ConvMMN
from cdl_eeg.models.domain_adaptation.domain_discriminators.getter import get_domain_discriminator
from cdl_eeg.models.metrics import Histories
from cdl_eeg.models.mts_modules.getter import get_mts_module
from cdl_eeg.models.utils import ReverseLayerF


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
        """
        use_cmmn_layer = cmmn_config["use_cmmn_layer"]
        return cls(mts_module=mts_config["name"],
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
        # of input channels. However, such usage is not recommended, as it is not necessary to store in a dict. Storing
        # in dict also makes it a little more difficult to know which is the i-th subject
        if isinstance(x, dict):
            warnings.warn("Passing the data as a dictionary of torch.Tensor values is not recommended.")
            x = torch.cat(tuple(x.values()), dim=0)

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

        return self._mts_module.classify_latent_features(x), self._domain_discriminator(gradient_reversed_x)

    def extract_latent_features(self, x):
        """Method for extracting latent features"""
        # (Maybe) concatenate all tensors. This should be possible, as this class should ony be used with a fixed number
        # of input channels
        if isinstance(x, dict):
            warnings.warn("Passing the data as a dictionary of torch.Tensor values is not recommended.")
            x = torch.cat(tuple(x.values()), dim=0)

        # Run through MTS module and return
        return self._mts_module.extract_latent_features(x)

    # ----------------
    # Methods for fitting CMMN layer
    # ----------------
    def fit_psd_barycenters(self, data, sampling_freq):
        self._cmmn_layer.fit_psd_barycenters(data=data, sampling_freq=sampling_freq)

    def fit_monge_filters(self, data):
        self._cmmn_layer.fit_monge_filters(data=data, is_psds=False)

    # ---------------
    # Methods for training
    # ---------------
    # todo: this must be updated
    def fit_model(self, *, train_loader, val_loader, metrics, num_epochs, criterion, optimiser, device,
                  prediction_activation_function=None, verbose=True):
        """
        Method for fitting model

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
        metrics :  str | tuple[str, ...]
        num_epochs : int
        criterion : nn.modules.loss._Loss
        optimiser : torch.optim.Optimizer
        device : torch.device
        prediction_activation_function : typing.Callable | None
        verbose : bool

        Returns
        -------

        """
        # Defining histories objects  todo: must fix
        train_history = Histories(metrics=metrics)  # type: ignore[call-arg]
        val_history = Histories(metrics=metrics, name="val")  # type: ignore[call-arg]

        # ------------------------
        # Fit model
        # ------------------------
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Start progress bar
            pbar = enlighten.Counter(total=int(len(train_loader) / train_loader.batch_size + 1),
                                     desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            # ----------------
            # Training
            # ----------------
            self.train()
            for x_train, y_train in train_loader:
                # Send data to correct device
                x_train = x_train.to(device)
                y_train = y_train.to(device)

                # Forward pass
                output = self(x_train)

                # Compute loss
                loss = criterion(output, y_train)
                loss.backward()
                optimiser.step()

                # Update train history
                with torch.no_grad():
                    y_pred = torch.clone(output)
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)
                    train_history.store_batch_evaluation(y_pred=y_pred, y_true=y_train)

                # Update progress bar
                pbar.update()

            # Finalise epoch for train history object
            train_history.on_epoch_end(verbose=verbose)

            # ----------------
            # Validation
            # ----------------
            self.eval()
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    # Send data to correct device
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    # Forward pass
                    y_pred = self(x_val)

                    # Update validation history
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)
                    val_history.store_batch_evaluation(y_pred=y_pred, y_true=y_val)

                # Finalise epoch for validation history object
                val_history.on_epoch_end(verbose=verbose)

        return train_history, val_history

    # ---------------
    # Properties
    # ---------------
    @property
    def has_domain_discriminator(self) -> bool:
        """Indicates if the model has a domain discriminator for domain adversarial learning (True) or not (False)"""
        return self._domain_discriminator is not None

    @property
    def has_cmmn_layer(self):  # todo: inconsistent property name with respect to the RBP version
        """Boolean indicating if the model uses a CMMN layer (True) or not (False)"""
        return self._cmmn_layer is not None
