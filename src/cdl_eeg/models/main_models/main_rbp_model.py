import enlighten
import torch
import torch.nn as nn

from cdl_eeg.models.metrics import Histories
from cdl_eeg.models.mts_modules.getter import get_mts_module
from cdl_eeg.models.region_based_pooling.region_based_pooling import RegionBasedPooling


class MainRBPModel(nn.Module):
    """
    (In early stages of development)

    Main model supporting use of RBP. That is, this class uses RBP as a first layer, followed by an MTS module

    PS: Merges channel splits by concatenation
    """

    def __init__(self, *, mts_module, mts_module_kwargs, rbp_designs):
        """
        Initialise

        Parameters
        ----------
        mts_module : str
        mts_module_kwargs : dict[str, typing.Any]
        rbp_designs : tuple[cdl_eeg.models.region_based_pooling.region_based_pooling.RBPDesign, ...]
        """
        super().__init__()

        # -----------------
        # Create RBP layer
        # -----------------
        self._region_based_pooling = RegionBasedPooling(rbp_designs)

        # ----------------
        # Create MTS module
        # ----------------
        # todo: the number of in channels must be calculated from/by the RBP layer
        self._mts_module = get_mts_module(mts_module_name=mts_module, **mts_module_kwargs)

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
        # TODO: Would be nice to be able to have different channel systems in the same batch
        # Pass through RBP layer
        x = self._region_based_pooling(input_tensors, channel_name_to_index=channel_name_to_index,
                                       pre_computed=pre_computed)

        # Merge by concatenation
        x = torch.cat(x, dim=1)

        # Pass through MTS module and return
        return self._mts_module(x)

    # ----------------
    # Methods for fitting channel systems
    # ----------------
    def fit_channel_system(self, channel_system):
        self._region_based_pooling.fit_channel_system(channel_system)

    def fit_channel_systems(self, channel_systems):
        self._region_based_pooling.fit_channel_systems(channel_systems)

    # ----------------
    # Methods for training
    # todo: don't know if I should have another class, and if this generalisation really works...
    # ----------------
    def pre_train(self, *, train_loader, val_loader, metrics, num_epochs, criterion, optimiser, device,
                  train_channel_system_name, train_channel_name_to_index, val_channel_system_name,
                  val_channel_name_to_index, prediction_activation_function=None, verbose=True):
        """
        Method for pre-training

        TODO: make tests. Also, is this really specific to pre-training?

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
        metrics : str | tuple[str, ...]
        num_epochs : int
        criterion : nn.modules.loss._Loss
        optimiser : torch.optim.Optimizer
        device : torch.device
        train_channel_system_name
        train_channel_name_to_index
        val_channel_system_name
        val_channel_name_to_index
        prediction_activation_function : typing.Callable | None
        verbose : bool

        Returns
        -------
        tuple[Histories, Histories]
            Training and validation histories
        """
        # todo: may want to think more on memory usage

        # Defining histories objects
        train_history = Histories(metrics=metrics)
        val_history = Histories(metrics=metrics)

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
            for x_train, train_pre_computed, y_train in train_loader:
                # todo: Only works for train loaders with this specific __getitem__ return
                # Send data to correct device
                x_train = x_train.to(device)  # TODO: this is a dict
                y_train = torch.tensor(y_train, dtype=torch.float).to(device)

                # Forward pass
                output = self(x_train, pre_computed=train_pre_computed, channel_system_name=train_channel_system_name,
                              channel_name_to_index=train_channel_name_to_index)

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
                for x_val, val_pre_computed, y_val in val_loader:
                    # Send data to correct device
                    x_val = x_val.to(device)  # TODO: this is a dict
                    y_val = torch.tensor(y_val, dtype=torch.float).to(device)

                    # Forward pass  todo: why did I use .clone() in the PhD course tasks?
                    y_pred = self(x_val, pre_computed=val_pre_computed, channel_system_name=val_channel_system_name,
                                  channel_name_to_index=val_channel_name_to_index)

                    # Update validation history
                    if prediction_activation_function is not None:
                        y_pred = prediction_activation_function(y_pred)
                    val_history.store_batch_evaluation(y_pred=y_pred, y_true=y_val)

                # Finalise epoch for validation history object
                val_history.on_epoch_end()

        return train_history, val_history
