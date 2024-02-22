import abc

import torch
import torch.nn as nn


def get_activation_function(name):
    """
    Get activation function

    Parameters
    ----------
    name : str, optional

    Returns
    -------
    typing.Callable
    """
    if name is None:
        return None

    available_functions = (torch.sigmoid,)

    # Loop through and select the correct one
    for func in available_functions:
        if name == func.__name__:
            return func

    # If no match, an error is raised
    raise ValueError(f"The activation function '{name}' was not recognised. Please select among the following: "
                     f"{tuple(func.__name__ for func in available_functions)}")


# -------------
# Classes for re-weighting the loss function
# -------------
class LossWeighter(abc.ABC):

    __slots__ = ()

    @abc.abstractmethod
    def compute_weights(self, subjects):
        """Method for computing the weights"""

    @abc.abstractmethod
    def reduce_loss(self, loss):
        """Method for reducing the loss-function, after the weights have been computed and applied. Should typically be
        computing the mean"""


class SampleWeighter(LossWeighter, abc.ABC):
    __slots__ = ()


class SamplePowerWeighter(SampleWeighter):
    """
    Weights are proportional to 1 / N^weight_power. Although it is weird to use weight_power smaller than 0 or greater
    than 1, an error will not be raised

    Examples
    --------
    >>> my_dataset_sizes = {"Mercedes": 170000, "RedBull": 17848, "Ferrari": 5000}
    >>> my_weight_power = 0.7
    >>> my_weighter = SamplePowerWeighter(my_dataset_sizes, my_weight_power)
    >>> expected_constant = (170000 + 17848 + 5000) / (170000 ** 0.3 + 17848 ** 0.3 + 5000 ** 0.3)
    >>> abs(expected_constant - my_weighter._normalisation_constant) < 1e-10
    True
    >>> my_weighter._normalisation_constant  # doctest: +ELLIPSIS
    2802.612...
    """

    __slots__ = ("_weight_power", "_normalisation_constant", "_dataset_sizes")

    def __init__(self, dataset_sizes, weight_power):
        # Store weight power
        self._weight_power = weight_power

        # Store dataset sizes
        self._dataset_sizes = dataset_sizes

        # Compute and store normalisation constant
        self._normalisation_constant = (sum(dataset_sizes.values()) /
                                        sum(size**(1 - weight_power) for size in dataset_sizes.values()))

    def compute_weights(self, subjects):
        # Compute weights
        weights = self._normalisation_constant * torch.tensor(
            [self._dataset_sizes[subject.dataset_name] ** (-self._weight_power) for subject in subjects]
        )

        # In this project, it is convenient to fix the output shape
        return torch.unsqueeze(weights, dim=-1)

    def reduce_loss(self, loss):
        return loss.mean()


# -------------
# Get functions
# -------------
def get_loss_weighter(name, **kwargs):
    # All available loss weighters must be included here
    available_weighters = (SamplePowerWeighter,)

    # Loop through and select the correct one
    for weighter in available_weighters:
        if name == weighter.__name__:
            return weighter(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The weighter '{name}' was not recognised. Please select among the following: "
                     f"{tuple(weighter.__name__ for weighter in available_weighters)}")


def get_pytorch_loss_function(name, **kwargs):
    # All available loss functions must be included here
    available_losses = (nn.MSELoss, nn.L1Loss, nn.BCELoss, nn.BCEWithLogitsLoss, nn.CrossEntropyLoss)

    # Loop through and select the correct one
    for available_loss in available_losses:
        if name == available_loss.__name__:
            return available_loss(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The loss function '{name}' was not recognised. Please select among the following: "
                     f"{tuple(available_loss.__name__ for available_loss in available_losses)}")


# -------------
# Classes
# -------------
class CustomWeightedLoss:
    """
    Class for customised weighting of the loss function, initially meant to use different weights for different
    datasets, depending on the dataset size

    Examples
    --------
    >>> from cdl_eeg.data.data_split import Subject
    >>> my_dataset_sizes = {"Mercedes": 170000, "RedBull": 17848, "Ferrari": 5000}
    >>> my_weight_power = 0.7
    >>> my_criterion = CustomWeightedLoss("L1Loss", {"reduction": "none"}, "SamplePowerWeighter",
    ...                                   {"dataset_sizes": my_dataset_sizes, "weight_power": my_weight_power})
    >>> my_subjects = (Subject("MV", "RedBull"), Subject("GR", "Mercedes"), Subject("CL", "Ferrari"))
    >>> my_x = torch.unsqueeze(torch.tensor([[56, 33, 54]], dtype=torch.float), dim=-1)
    >>> my_y = torch.unsqueeze(torch.tensor([[57, 30, 50]], dtype=torch.float), dim=-1)
    >>> # noinspection PyProtectedMember
    >>> my_w1 = my_criterion._weighter._normalisation_constant * 170000 ** (-my_weight_power)
    >>> # noinspection PyProtectedMember
    >>> my_w2 = my_criterion._weighter._normalisation_constant * 17848 ** (-my_weight_power)
    >>> # noinspection PyProtectedMember
    >>> my_w3 = my_criterion._weighter._normalisation_constant * 5000 ** (-my_weight_power)
    >>> my_expected_loss = torch.tensor((my_w2 * 1 + my_w1 * 3 + my_w3 * 4) / 3)
    >>> my_actual_loss = my_criterion(my_x, my_y, subjects=my_subjects)
    >>> torch.equal(my_expected_loss, my_actual_loss)
    True
    """

    __slots__ = ("_criterion", "_weighter")

    def __init__(self, loss, loss_kwargs, weighter, weighter_kwargs):
        self._criterion = get_pytorch_loss_function(name=loss, **loss_kwargs)
        self._weighter = None if weighter is None else get_loss_weighter(weighter, **weighter_kwargs)

    def __call__(self, input_tensor, target, *, subjects=None):
        # Compute loss
        loss = self._criterion(input_tensor, target)

        # If no 'weighter' is used, just return the loss as normally
        if self._weighter is None:
            return loss

        # Compute weights
        weights = self._weighter.compute_weights(subjects=subjects)

        # Send weights to correct device and apply them, and return
        loss *= weights.to(loss.device)

        # Reduce (e.g., sum or mean) and return
        return self._weighter.reduce_loss(loss)
