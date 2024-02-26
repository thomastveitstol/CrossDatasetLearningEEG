import enlighten
import torch
import torch.nn as nn

from cdl_eeg.models.metrics import Histories
from cdl_eeg.models.mts_modules.getter import get_mts_module


class MainFixedChannelsModel(nn.Module):
    """
    Main model when the number of input channels is fixed
    """

    def __init__(self, mts_module, **kwargs):
        super().__init__()

        # ----------------
        # Create MTS module
        # ----------------
        self._mts_module = get_mts_module(mts_module_name=mts_module, **kwargs)

    @classmethod
    def from_config(cls, config):
        return cls(mts_module=config["name"], **config["kwargs"])

    # ---------------
    # Methods for forward propagation
    # ---------------
    def forward(self, x):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]

        Returns
        -------
        torch.Tensor
            Outputs of MTS module without applying a final activation function

        Examples
        --------
        >>> my_model = MainFixedChannelsModel("InceptionTime", in_channels=23, num_classes=11)
        >>> my_model(torch.rand(size=(10, 23, 300))).size()
        torch.Size([10, 11])
        >>> my_model({"d1": torch.rand(size=(10, 23, 300)), "d2": torch.rand(size=(4, 23, 300))}).size()
        torch.Size([14, 11])
        """
        # (Maybe) concatenate all tensors. This should be possible, as this class should ony be used with a fixed number
        # of input channels
        if isinstance(x, dict):
            x = torch.cat(tuple(x.values()), dim=0)

        # Run through MTS module and return
        return self._mts_module(x)

    def extract_latent_features(self, x):
        """Method for extracting latent features"""
        # (Maybe) concatenate all tensors. This should be possible, as this class should ony be used with a fixed number
        # of input channels
        if isinstance(x, dict):
            x = torch.cat(tuple(x.values()), dim=0)

        # Run through MTS module and return
        return self._mts_module.extract_latent_features(x)

    # ---------------
    # Methods for training
    # ---------------
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
