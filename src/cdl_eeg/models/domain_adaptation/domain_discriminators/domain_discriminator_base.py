from typing import List

from torch import nn


# ----------------
# Convenient decorator
# ----------------
def sampling_method(func):
    setattr(func, "_is_sampling_method", True)
    return func


class DomainDiscriminatorBase(nn.Module):

    # --------------
    # Methods for creating a random architecture
    # --------------
    @classmethod
    def get_sampling_methods(cls):
        # Get all sampling methods
        sampling_methods: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a sampling method
            if callable(attribute) and getattr(attribute, "_is_sampling_method", False):
                sampling_methods.append(method)

        # Convert to tuple and return
        return tuple(sampling_methods)

    @classmethod
    def sample_hyperparameters(cls, method, **kwargs):
        # Input check
        if method not in cls.get_sampling_methods():
            raise ValueError(f"The sampling method {method} was not recognised. The available ones are: "
                             f"{cls.get_sampling_methods()}")

        # Sample hyperparameters
        return getattr(cls, method)(**kwargs)
