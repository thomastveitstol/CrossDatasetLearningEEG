import abc
from typing import List


def transformation_method(func):
    """Decorator to be used on methods which are transformation methods"""
    setattr(func, "_is_transformation_method", True)
    return func


class TransformationBase(abc.ABC):
    __slots__ = ()

    @classmethod
    def get_available_transformations(cls):
        """Get all transformation methods available for the class. The transformation method must be decorated by
        @transformation_method to be properly registered"""
        # Get all transformation methods
        transformations: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a transformation method
            if callable(attribute) and getattr(attribute, "_is_transformation_method", False):
                transformations.append(method)

        # Convert to tuple and return
        return tuple(transformations)

    def transform(self, method):
        """
        Method for getting the specified transformation method

        Parameters
        ----------
        method : str
            The selected transformation method. Must correspond to the name of an implemented method

        Returns
        -------
        typing.Callable
            See the individual transformation methods for more details on return type
        """
        # Input check
        if method not in self.get_available_transformations():
            raise ValueError(f"Transformation method '{method}' was not recognised. Make sure that the method passed "
                             f"shares the name with the implemented method you want to use. The transformation methods "
                             f"available for this class ({type(self).__name__}) are: "
                             f"{self.get_available_transformations()}")

        # Return the method itself
        return getattr(self, method)
