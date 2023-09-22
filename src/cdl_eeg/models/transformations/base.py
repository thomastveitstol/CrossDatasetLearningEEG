import abc
from typing import List


def transformation_method(func):
    setattr(func, "_is_transformation_method", True)
    return func


class TransformationBase(abc.ABC):
    __slots__ = ()

    @classmethod
    def get_available_transformations(cls):
        # Get all transformation methods
        transformations: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a transformation method
            if callable(attribute) and getattr(attribute, "_is_transformation_method", False):
                transformations.append(method)

        # Convert to tuple and return
        return tuple(transformations)
