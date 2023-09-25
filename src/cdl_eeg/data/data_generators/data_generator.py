import torch
from torch.utils.data import Dataset


class SelfSupervisedDataGenerator(Dataset):  # type: ignore[type-arg]
    """
    (In the very early stage of development)

    Data generator for self supervised pretext tasks, where the targets are generated on the fly

    Examples
    >>> import numpy
    >>> from cdl_eeg.models.transformations.frequency_slowing import FrequencySlowing
    >>> from cdl_eeg.models.transformations.utils import UnivariateNormal
    >>> my_data_gen = SelfSupervisedDataGenerator(numpy.random.normal(size=(10, 64, 2000)),
    ...                                           transformation=FrequencySlowing(UnivariateNormal(0.7, 0.1)),
    ...                                           pretext_task="phase_slowing")

    An error is raised if a non-existing pretext task is attempted set

    >>> my_data_gen.pretext_task = "NotAPretextTask"  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The pretext task 'NotAPretextTask' is not available for the current transformation class ...
    """

    def __init__(self, x, *, transformation, pretext_task, pre_computed=None):
        """
        Initialise

        Parameters
        ----------
        x : numpy.ndarray
            Input data to the model. TODO: I must decide if I want to use numpy arrays or torch tensors
        transformation : cdl_eeg.models.transformations.base.TransformationBase
            The object responsible for performing the transformations
        pretext_task : str
            The pretext task to use
        pre_computed
        """
        super().__init__()

        self._x = x
        self._transformation = transformation
        self._pretext_task = pretext_task
        self._pre_computed = pre_computed

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, item):
        # todo: if the __getitem__ is to be standardised, the input to the transformation methods must also be
        transformed, details = self._transformation.transform(self._pretext_task)(self._x[item])

        # TODO: quite hard coded?
        if self._pre_computed is None:
            return torch.tensor(transformed, dtype=torch.float), details
        else:
            pre_computed = tuple(pre_comp[item] for pre_comp in self._pre_computed)
            return torch.tensor(transformed, dtype=torch.float), pre_computed, details

    # ---------------
    # Properties
    # ---------------
    @property
    def pretext_task(self):
        return self._pretext_task

    @pretext_task.setter
    def pretext_task(self, task: str) -> None:
        # Input check
        if task not in self._transformation.get_available_transformations():
            raise ValueError(f"The pretext task '{task}' is not available for the current transformation class "
                             f"({type(self._transformation).__name__}). Please select among the following: "
                             f"{self._transformation.get_available_transformations()}")

        # Set attribute
        self._pretext_task = task
