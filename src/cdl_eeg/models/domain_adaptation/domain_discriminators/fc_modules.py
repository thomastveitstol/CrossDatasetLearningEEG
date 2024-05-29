import random
from typing import List, Any, Dict

from torch import nn

from cdl_eeg.models.domain_adaptation.domain_discriminators.domain_discriminator_base import DomainDiscriminatorBase, \
    sampling_method
from cdl_eeg.models.random_search.sampling_distributions import sample_hyperparameter


class FCModule(DomainDiscriminatorBase):
    """
    A standard FC module with relu as activation function in all hidden layers, and no activation function on the output
    layer

    Examples
    --------
    >>> FCModule(77, 5, activation_function="relu")
    FCModule(
      (_model): ModuleList(
        (0): Linear(in_features=77, out_features=5, bias=True)
      )
    )

    Example with hidden units

    >>> FCModule(77, 5, hidden_units=(4, 9, 6), activation_function="elu", activation_function_kwargs={"alpha": 1.2})
    FCModule(
      (_model): ModuleList(
        (0): Linear(in_features=77, out_features=4, bias=True)
        (1): ELU(alpha=1.2)
        (2): Linear(in_features=4, out_features=9, bias=True)
        (3): ELU(alpha=1.2)
        (4): Linear(in_features=9, out_features=6, bias=True)
        (5): ELU(alpha=1.2)
        (6): Linear(in_features=6, out_features=5, bias=True)
      )
    )

    It also works when passed as a list

    >>> FCModule(77, 5, hidden_units=[4, 9, 6], activation_function="relu")
    FCModule(
      (_model): ModuleList(
        (0): Linear(in_features=77, out_features=4, bias=True)
        (1): ReLU()
        (2): Linear(in_features=4, out_features=9, bias=True)
        (3): ReLU()
        (4): Linear(in_features=9, out_features=6, bias=True)
        (5): ReLU()
        (6): Linear(in_features=6, out_features=5, bias=True)
      )
    )
    """

    def __init__(self, in_features, num_classes, *, hidden_units=(), activation_function,
                 activation_function_kwargs=None):
        """
        Initialise

        Parameters
        ----------
        in_features : int
        num_classes : int
        hidden_units : tuple[int, ...]
        """
        super().__init__()

        # Set default
        activation_function_kwargs = dict() if activation_function_kwargs is None else activation_function_kwargs

        # Create model
        _in_features = (in_features,) + tuple(hidden_units)
        _out_features = tuple(hidden_units) + (num_classes,)
        modules: List[nn.Module] = []
        for i, (f_in, f_out) in enumerate(zip(_in_features, _out_features)):
            # Add the layer
            modules.append(nn.Linear(in_features=f_in, out_features=f_out))

            # Maybe add activation function
            if i != len(hidden_units):
                modules.append(_get_activation_function(activation_function, **activation_function_kwargs))

        self._model = nn.ModuleList(modules)

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
        >>> outputs = FCModule(in_features=77, num_classes=8, hidden_units=(2, 5, 3),
        ...                    activation_function="relu")(torch.rand(size=(10, 77)))
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

    # --------------
    # Methods for creating a random architecture
    # --------------
    @staticmethod
    @sampling_method
    def exponential_decay(config, in_features):
        """The number of layers are decreasing exponentially from first to last hidden layer"""
        # --------------
        # Sample hidden layers
        # --------------
        # Sample depth
        proposed_depth = sample_hyperparameter(config["depth"]["dist"], **config["depth"]["kwargs"])

        # Sample the first layer
        first_layer = int(sample_hyperparameter(config["first_layer_multiple"]["dist"],
                                                **config["first_layer_multiple"]["kwargs"]) * in_features)

        # Sample the exponential decrease
        exponential_decrease = sample_hyperparameter(config["exponential_decrease"]["dist"],
                                                     **config["exponential_decrease"]["kwargs"])

        # Create hidden layers
        hidden_units: List[int] = []
        num_units = first_layer
        for _ in range(proposed_depth):
            # Add hidden layers
            hidden_units.append(num_units)

            # Decrease units for the next layer
            num_units //= exponential_decrease

            # If the number of units is 1, the depth is decreased
            if num_units == 1:
                break

        # --------------
        # Sample activation function
        # --------------
        activation_function, activation_function_kwargs = _sample_activation_function(config["activation_function"])

        # Return configuration which can be used as input to __init__
        return {"hidden_units": tuple(hidden_units),
                "activation_function": activation_function,
                "activation_function_kwargs": activation_function_kwargs}


def _get_activation_function(activation_function, **kwargs):
    if activation_function == "relu":
        return nn.ReLU()
    elif activation_function == "elu":
        return nn.ELU(alpha=kwargs["alpha"])
    else:
        raise ValueError(f"Unexpected activation function: {activation_function}")


def _sample_activation_function(activation_func_config):
    """
    Function for sampling an activation function

    Parameters
    ----------
    activation_func_config : dict[str, Any]

    Returns
    -------
    tuple[str, dict[str, Any]]

    Examples
    --------
    >>> random.seed(7)
    >>> import numpy
    >>> numpy.random.seed(7)
    >>> my_config = {"relu": {}, "elu": {"alpha": {"dist": "uniform", "kwargs": {"a": 0.2, "b": 1.8}}}}
    >>> _sample_activation_function(my_config)  # doctest: +ELLIPSIS
    ('elu', {'alpha': 0.32...})
    >>> _sample_activation_function(my_config)
    ('relu', {})
    >>> _sample_activation_function(my_config)  # doctest: +ELLIPSIS
    ('elu', {'alpha': 1.44...})
    >>> _sample_activation_function(my_config)
    ('relu', {})
    """
    # Select activation function
    activation_function = random.choice(tuple(activation_func_config.keys()))

    # Sample hyperparameters
    kwargs: Dict[str, Any] = dict()
    for param, domain in activation_func_config[activation_function].items():
        if isinstance(domain, dict) and "dist" in domain:
            kwargs[param] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
        else:
            kwargs[param] = domain

    # Return an activation function
    return activation_function, kwargs
