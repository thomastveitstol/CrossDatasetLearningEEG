import abc


def transformation_method(func):
    setattr(func, "_is_transformation", True)
    return func


class TransformationBase(abc.ABC):
    __slots__ = ()
