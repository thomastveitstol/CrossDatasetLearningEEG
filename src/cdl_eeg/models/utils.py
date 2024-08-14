import abc

import numpy
import torch
from torch.autograd import Function


# ---------------
# Classes
# ---------------
class ReverseLayerF(Function):
    """
    Gradient reversal layer. This is simply copypasted (with added '# noqa') from the implementation at
    https://github.com/wogong/pytorch-dann/blob/master/models/functions.py

    See LICENSE.txt for their original copyright notice and disclaimer
    """

    @staticmethod
    def forward(ctx, x, alpha):  # noqa
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        output = grad_output.neg() * ctx.alpha
        return output, None


# ---------------
# Functions for training
# ---------------
def tensor_dict_to_device(tensors, device):
    """
    Send a dictionary containing tensors to device

    Parameters
    ----------
    tensors : dict[str, torch.Tensor] | None
    device : torch.device

    Returns
    -------
    dict[str, torch.Tensor]
    """
    # If the tensor is None, then None is returned
    if tensors is None:
        return None

    # Input check
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError(f"Expected all values in the dictionary to be torch tensors, but found "
                        f"{set(type(tensor) for tensor in tensors.values())}")

    # Send to device and return
    return {dataset_name: tensor.to(device) for dataset_name, tensor in tensors.items()}


def flatten_targets(tensors):
    """
    Flatten the targets. The sorting is determined from the ordering of the keys

    Parameters
    ----------
    tensors : dict[str, torch.Tensor | numpy.ndarray]

    Returns
    -------
    torch.Tensor

    Examples
    --------
    >>> my_targets = {"d1": torch.ones(size=(10, 1)), "d2": torch.ones(size=(11, 1)) * 2,
    ...               "d3": torch.ones(size=(5, 1)) * 3}
    >>> my_output = flatten_targets(my_targets)
    >>> my_output.size()
    torch.Size([26, 1])

    The ordering is determined from the keys

    >>> torch.equal(my_output[:10], torch.ones(size=(10, 1)))
    True
    >>> torch.equal(my_output[10:21], 2 * torch.ones(size=(11, 1)))
    True
    >>> torch.equal(my_output[21:], 3 * torch.ones(size=(5, 1)))
    True
    """
    # Maybe convert to torch tensors
    if all(isinstance(tensor, numpy.ndarray) for tensor in tensors.values()):
        tensors = {dataset_name: torch.tensor(tensor, dtype=torch.float) for dataset_name, tensor in tensors.items()}

    # Flatten
    targets = torch.cat(tuple(tensor for tensor in tensors.values()), dim=0)

    return targets


# -------------------------
# Random number classes
# -------------------------
class RandomBase(abc.ABC):

    __slots__ = ()

    @abc.abstractmethod
    def draw(self, seed=None):
        """
        Draw a sample from the distribution

        Parameters
        ----------
        seed : int
            A seed can be passed for reproducibility

        Returns
        -------
        float
            A sample from the distribution
        """

    @abc.abstractmethod
    def scale(self, draw):
        """
        Method for scaling a random  draw

        Parameters
        ----------
        draw : float

        Returns
        -------
        float
        """


class UnivariateNormal(RandomBase):
    """
    Class for drawing samples from a univariate normal distribution. To reproduce, numpy.random.seed must be called
    """

    __slots__ = "_mean", "_std"

    def __init__(self, mean=0, std=1):
        """
        Initialisation method

        Parameters
        ----------
        std : float
            Standard deviation of the normal distribution
        mean : float
            Mean of the normal distribution
        """
        # ------------------
        # Set attributes
        # ------------------
        self._mean = mean
        self._std = std

    def draw(self, seed=None):
        """
        Examples
        -------
        >>> UnivariateNormal(3, 0.5).draw(seed=1)  # doctest: +ELLIPSIS
        3.812...

        It is the same to set seed outside, as passing the seed to the method

        >>> numpy.random.seed(1)
        >>> UnivariateNormal(3, 0.5).draw()  # doctest: +ELLIPSIS
        3.812...
        """
        # Maybe set seed
        if seed is not None:
            numpy.random.seed(seed)

        # Draw a sample from the distribution and return
        return numpy.random.normal(loc=self._mean, scale=self._std)

    def scale(self, draw):
        """
        Scaling method which subtracts the mean and divide by standard deviation

        Parameters
        ----------
        draw : float

        Returns
        -------
        float

        Examples
        --------
        >>> UnivariateNormal(3, 0.5).scale(3.5)
        1.0
        >>> round(UnivariateNormal(3, 0.5).scale(2.7), 5)
        -0.6
        """
        return (draw - self._mean) / self._std


class UnivariateUniform(RandomBase):

    __slots__ = "_lower", "_upper"

    def __init__(self, lower, upper):
        """
        Initialisation

        Parameters
        ----------
        lower : float
            Lower bound for the uniform distribution
        upper : float
            Upper bound for the uniform distribution
        """
        # ------------------
        # Set attributes
        # ------------------
        self._lower = lower
        self._upper = upper

    def draw(self, seed=None):
        """
        Examples:
        >>> UnivariateUniform(-1, 1).draw(seed=1)  # doctest: +ELLIPSIS
        -0.165...

        It is the same to set seed outside, as passing the seed to the method

        >>> numpy.random.seed(1)
        >>> UnivariateUniform(-1, 1).draw()  # doctest: +ELLIPSIS
        -0.165...
        """
        # Maybe set seed
        if seed is not None:
            numpy.random.seed(seed)

        # Draw a sample from the distribution and return
        return numpy.random.uniform(low=self._lower, high=self._upper)

    def scale(self, draw):
        """
        Scales such that the distribution is U[-1, 1]

        Parameters
        ----------
        draw : float

        Returns
        -------
        float

        Examples
        --------
        >>> UnivariateUniform(9, 10).scale(9)
        -1.0
        >>> UnivariateUniform(9, 10).scale(10)
        1.0
        >>> UnivariateUniform(9, 10).scale(9.5)
        0.0
        >>> UnivariateUniform(9, 10).scale(9.75)
        0.5
        """
        # Subtract mean
        draw -= (self._upper + self._lower) / 2

        # Scale interval
        draw /= (self._upper - self._lower) / 2

        return draw

    # --------------
    # Properties
    # --------------
    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper


def get_random_distribution(distribution, **kwargs):
    """
    Function for getting the specified random distribution

    Parameters
    ----------
    distribution : str
    kwargs

    Returns
    -------
    RandomBase
    """
    # All available distributions must be included here
    available_distributions = (UnivariateUniform, UnivariateNormal)

    # Loop through and select the correct one
    for dist in available_distributions:
        if distribution == dist.__name__:
            return dist(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The random distribution '{distribution}' was not recognised. Please select among the following: "
                     f"{tuple(dist.__name__ for dist in available_distributions)}")
