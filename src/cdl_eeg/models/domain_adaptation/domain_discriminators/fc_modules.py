from torch import nn

from cdl_eeg.models.domain_adaptation.domain_discriminators.domain_discriminator_base import DomainDiscriminatorBase


class FCModule(DomainDiscriminatorBase):
    """
    A standard FC module with relu as activation function in all hidden layers, and no activation function on the output
    layer

    Examples
    --------
    >>> FCModule(77, 5)
    FCModule(
      (_model): ModuleList(
        (0): Linear(in_features=77, out_features=5, bias=True)
      )
    )

    Example with hidden units (yes, it works when passed as a list)

    >>> FCModule(77, 5, hidden_units=[4, 9, 6])
    FCModule(
      (_model): ModuleList(
        (0): Linear(in_features=77, out_features=4, bias=True)
        (1): Linear(in_features=4, out_features=9, bias=True)
        (2): Linear(in_features=9, out_features=6, bias=True)
        (3): Linear(in_features=6, out_features=5, bias=True)
      )
    )
    """

    def __init__(self, in_features, num_classes, *, hidden_units=()):
        """
        Initialise

        Parameters
        ----------
        in_features : int
        num_classes : int
        hidden_units : tuple[int, ...]
        """
        super().__init__()

        # Create model
        _in_features = (in_features,) + tuple(hidden_units)
        _out_features = tuple(hidden_units) + (num_classes,)
        self._model = nn.ModuleList([nn.Linear(in_features=f_in, out_features=f_out)
                                     for f_in, f_out in zip(_in_features, _out_features)])

    def forward(self, input_tensor):
        """
        Forward method

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> _ = torch.manual_seed(2)
        >>> outputs = FCModule(in_features=77, num_classes=8, hidden_units=(2, 5, 3))(torch.rand(size=(10, 77)))
        >>> outputs.size()
        torch.Size([10, 8])

        Relu not applied to output layer

        >>> outputs[0]
        tensor([-0.4527, -0.4727,  0.0706,  0.0054, -0.4165,  0.3035,  0.4215,  0.0216],
               grad_fn=<SelectBackward0>)
        """
        x = input_tensor

        # Loop through all layers
        for i, layer in enumerate(self._model):
            # Pass through layer
            x = layer(x)

            # If it is not the final layer, use relu activation
            if i != len(self._model) - 1:
                x = nn.functional.relu(x)

        return x
