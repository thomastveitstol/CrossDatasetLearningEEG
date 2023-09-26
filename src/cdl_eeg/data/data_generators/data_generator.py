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
    >>> my_data_gen = SelfSupervisedDataGenerator({"dataset_1": numpy.random.normal(size=(10, 64, 2132)),
    ...                                            "dataset_2": numpy.random.normal(size=(10, 32, 2149))},
    ...                                           transformation=FrequencySlowing(UnivariateNormal(0.7, 0.1)),
    ...                                           pretext_task="phase_slowing")

    An error is raised if a non-existing pretext task is attempted set

    >>> my_data_gen.pretext_task = "NotAPretextTask"  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The pretext task 'NotAPretextTask' is not available for the current transformation class ...
    """

    def __init__(self, data, *, transformation, pretext_task, pre_computed=None):
        """
        Initialise

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            Input data to the model. TODO: I must decide if I want to use numpy arrays or torch tensors
        transformation : cdl_eeg.models.transformations.base.TransformationBase
            The object responsible for performing the transformations
        pretext_task : str
            The pretext task to use
        pre_computed : dict[str, typing.Any]
        """
        super().__init__()

        self._data = data
        self._transformation = transformation
        self._pretext_task = pretext_task
        self._pre_computed = pre_computed

    def __len__(self):
        return sum(x.shape[0] for x in self._data.values())

    def __getitem__(self, item):
        # todo: if the __getitem__ is to be standardised, the input to the transformation methods must also be
        # Select dataset and subject in the dataset
        dataset_size = {dataset_name: x.shape[0] for dataset_name, x in self._data.items()}
        dataset_name, idx = _select_dataset_and_index(item, dataset_size)

        # Perform permutation and get details
        transformed, details = self._transformation.transform(self._pretext_task)(self._data[dataset_name][idx])

        # TODO: quite hard coded?
        if self._pre_computed is None:
            return {dataset_name: torch.tensor(transformed, dtype=torch.float)}, {dataset_name: details}
        else:
            pre_computed = tuple({dataset_name: pre_comp[dataset_name][idx]} for pre_comp in self._pre_computed)
            # TODO: must fix return, as KeyError is raised when collating
            return {dataset_name: torch.tensor(transformed, dtype=torch.float)}, pre_computed, {dataset_name: details}

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


# ----------------
# Functions
# ----------------
def _select_dataset_and_index(item, dataset_sizes):
    """
    Function for selecting dataset. Only works for positive integer items

    Parameters
    ----------
    item : int
    dataset_sizes : tuple[dict[str, int], ...]

    Returns
    -------
    tuple[str, int]

    Examples
    --------
    >>> my_sizes = {"a": 3, "b": 6, "c": 10, "d": 2}
    >>> _select_dataset_and_index(item=11, dataset_sizes=my_sizes)
    ('c', 2)
    >>> _select_dataset_and_index(item=-1, dataset_sizes=my_sizes)
    Traceback (most recent call last):
    ...
    ValueError: Expected item to be a positive integer, but found -1 (type=<class 'int'>)
    """
    # Input check
    if not isinstance(item, int) or item < 0:
        raise ValueError(f"Expected item to be a positive integer, but found {item} (type={type(item)})")

    # Find the dataset name and position
    accumulated_sizes = 0
    for name, size in dataset_sizes.items():
        accumulated_sizes += size
        if item < accumulated_sizes:
            # Not very elegant...
            idx = item - (accumulated_sizes - size)
            return name, idx

    # This should not happen...
    raise RuntimeError("Found not select a dataset and an index")
