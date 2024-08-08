"""
Classes and functions for sampling from different distributions
"""
import random
from typing import List

import numpy


# ------------------
# Convenient decorators
# ------------------
def sampling_distribution(func):
    setattr(func, "_is_sampling_distribution", True)
    return func


# ------------------
# Classes and methods
# ------------------
class _SampleDistribution:
    """
    Convenience class for sampling from a specified distribution

    Examples
    --------
    >>> _SampleDistribution.get_available_distribution()
    ('log_uniform', 'log_uniform_int', 'n_log_uniform_int', 'normal', 'uniform', 'uniform_discrete', 'uniform_int')
    >>> numpy.random.seed(1)
    >>> round(_SampleDistribution.sample("log_uniform", base=10, a=0, b=3), 3)  # 10^x, x ~ U[a, b]
    17.826
    """

    @classmethod
    def sample(cls, distribution, **kwargs):
        # Input check
        if distribution not in cls.get_available_distribution():
            raise ValueError(f"The distribution {distribution} was not recognised. The available ones are: "
                             f"{cls.get_available_distribution()}")

        # Sample from the provided distribution
        return getattr(cls, distribution)(**kwargs)

    @classmethod
    def get_available_distribution(cls):
        """Get all sampling distributions available for the class. The distributions must be a method decorated by
        @sampling_distribution to be properly registered"""
        # Get all regression metrics
        distributions: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a sampling distribution
            if callable(attribute) and getattr(attribute, "_is_sampling_distribution", False):
                distributions.append(method)

        # Convert to tuple and return
        return tuple(distributions)

    # ------------
    # Distributions
    # ------------
    @staticmethod
    @sampling_distribution
    def log_uniform(*, base, a, b):
        """Samples a value x uniformly from [a, b], and outputs base^x"""
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"
        assert base > 0, f"Expected a positive base, but found {base}"

        return base ** numpy.random.uniform(a, b)

    @staticmethod
    @sampling_distribution
    def log_uniform_int(*, base, a, b):
        """Samples a value x uniformly from [a, b], and outputs round(base^x)"""
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"
        assert base > 0, f"Expected a positive base, but found {base}"

        return round(base ** numpy.random.uniform(a, b))

    @staticmethod
    @sampling_distribution
    def n_log_uniform_int(*, n, base, a, b):
        """Sample a value x uniformly from [a, b], and output n*round(base^x)"""
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"
        assert base > 0, f"Expected a positive base, but found {base}"

        return n * round(base ** numpy.random.uniform(a, b))

    @staticmethod
    @sampling_distribution
    def uniform(a, b):
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"

        return numpy.random.uniform(a, b)

    @staticmethod
    @sampling_distribution
    def uniform_int(a, b):
        """Random integer, including 'a', excluding 'b'"""
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"

        return numpy.random.randint(a, b)

    @staticmethod
    @sampling_distribution
    def normal(mean, std):
        return numpy.random.normal(mean, std)

    @staticmethod
    @sampling_distribution
    def uniform_discrete(domain):
        return random.choice(domain)


def sample_hyperparameter(distribution, **kwargs):
    """
    Function for sampling hyperparameters

    Parameters
    ----------
    distribution : str
    kwargs

    Returns
    -------
    typing.Any
    """
    return _SampleDistribution.sample(distribution, **kwargs)
