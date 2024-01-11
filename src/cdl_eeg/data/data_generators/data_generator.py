import numpy
import torch
from torch.utils.data import Dataset

from cdl_eeg.data.data_split import Subject


class SelfSupervisedFixedChannelsDataGenerator(Dataset):  # type: ignore[type-arg]
    """
    Data generator for self-supervised learning when the number of input channels is fixed
    """

    def __init__(self, input_data, *, transformation, pretext_task):
        super().__init__()

        # Input check
        self._input_data_check(input_data)

        # ------------
        # Set attributes
        # ------------
        self._data = numpy.concatenate(tuple(arr for arr in input_data.values()), axis=0) \
            if isinstance(input_data, dict) else input_data
        self._transformation = transformation
        self._pretext_task = pretext_task

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, item):
        # Perform permutation and get details
        transformed, details = self._transformation.transform(self._pretext_task)(self._data[item])

        # Convert to torch tensors and return
        return torch.tensor(transformed, dtype=torch.float), torch.tensor(details, dtype=torch.float)

    @staticmethod
    def _input_data_check(input_data):
        # Type checking
        if not isinstance(input_data, (dict, numpy.ndarray)):
            raise TypeError(f"Expected input data to be a numpy array or a dict of numpy arrays, but found "
                            f"{type(input_data)}")

        if isinstance(input_data, dict):
            # Type check of the keys
            if not all(isinstance(dataset_name, str) for dataset_name in input_data):
                raise TypeError(f"Expected all keys of the dictionary to be strings (dataset names), but found "
                                f"{set(type(dataset_name) for dataset_name in input_data)}")

            # Type check of the values
            if not all(isinstance(dataset_name, numpy.ndarray) for dataset_name in input_data.values()):
                raise TypeError(f"Expected all values of the dictionary to be numpy ndarrays, but found "
                                f"{set(type(arr) for arr in input_data.values())}")


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

    # Remember to remove the yielded -1 tensors!
    strip_outputs = True

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
        pre_computed : tuple[dict[str, typing.Any], ...]
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

        # Varying keys in the returned dictionary is not possible with the DataLoader of PyTorch. This solution to the
        # problem is to simply return a tensor of -1s for the datasets not used
        data = {dataset_name: torch.ones(size=x.shape[1:]) * (-1) for dataset_name, x in self._data.items()}
        all_details = {dataset_name: torch.unsqueeze(torch.tensor(-1), dim=-1)
                       for dataset_name in self._data}  # todo: hard-coded fill values

        # Select dataset and subject in the dataset
        dataset_size = {dataset_name: x.shape[0] for dataset_name, x in self._data.items()}
        dataset_name, idx = _select_dataset_and_index(item, dataset_size)

        # Perform permutation and get details
        transformed, details = self._transformation.transform(self._pretext_task)(self._data[dataset_name][idx])

        # Add the data which should be used
        data[dataset_name] = torch.tensor(transformed, dtype=torch.float)
        all_details[dataset_name] = torch.unsqueeze(torch.tensor(details, dtype=torch.float), dim=-1)

        # TODO: quite hard coded?
        if self._pre_computed is None:
            return data, all_details
        else:
            # assert False, {type(pre_comp) for pre_comp in self._pre_computed}
            pre_computed = []
            for pre_comp in self._pre_computed:
                my_dict = {data_name: torch.ones(tensor.size()[1:]) * (-1) for data_name, tensor in pre_comp.items()}
                my_dict[dataset_name] = pre_comp[dataset_name][idx]
                pre_computed.append(my_dict)

            # TODO: must fix return, as KeyError is raised when collating
            # Don't think I should convert pre_computed to tuple, as I must strip it anyway
            return data, pre_computed, all_details

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


class DownstreamDataGenerator(Dataset):  # type: ignore[type-arg]

    # Remember to remove the yielded -1 tensors!
    strip_outputs = True

    # --------------
    # Magic/dunder methods
    # --------------
    def __init__(self, data, targets, subjects, *, pre_computed=None):
        """
        Initialise

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
        targets : dict[str, numpy.ndarray]
        subjects : dict[str, tuple[str, ...]]
        pre_computed : tuple[dict[str, typing.Any], ...]
        """
        super().__init__()

        self._data = data
        self._targets = targets
        self._subjects = subjects
        self._pre_computed = pre_computed

    def __len__(self):
        return sum(x.shape[0] for x in self._data.values())

    def __getitem__(self, item):
        # TODO: copied from SelfSupervisedDataGenerator

        # Varying keys in the returned dictionary is not possible with the DataLoader of PyTorch. This solution to the
        # problem is to simply return a tensor of -1s for the datasets not used
        data = {dataset_name: torch.ones(size=x.shape[1:]) * (-1) for dataset_name, x in self._data.items()}
        targets = {dataset_name: torch.unsqueeze(torch.ones(size=y.shape[1:]) * (-1), dim=-1)
                   for dataset_name, y in self._targets.items()}

        # Select dataset and subject in the dataset
        dataset_size = {dataset_name: x.shape[0] for dataset_name, x in self._data.items()}
        dataset_name, idx = _select_dataset_and_index(item, dataset_size)

        # Add the data which should be used
        data[dataset_name] = torch.tensor(self._data[dataset_name][idx], dtype=torch.float)
        targets[dataset_name] = torch.unsqueeze(torch.tensor(self._targets[dataset_name][idx], dtype=torch.float),
                                                dim=-1)

        # TODO: quite hard coded?
        if self._pre_computed is None:
            return data, targets, item
        else:
            # assert False, {type(pre_comp) for pre_comp in self._pre_computed}
            pre_computed = []
            for pre_comp in self._pre_computed:
                my_dict = {data_name: torch.ones(tensor.size()[1:]) * (-1) for data_name, tensor in pre_comp.items()}
                my_dict[dataset_name] = pre_comp[dataset_name][idx]
                pre_computed.append(my_dict)

            # TODO: must fix return, as KeyError is raised when collating
            # Don't think I should convert pre_computed to tuple, as I must strip it anyway
            return data, pre_computed, targets, item

    # --------------
    # Convenient methods
    # --------------
    def get_subject_from_idx(self, item):
        """
        Get the subject from the index. It is needed because the subject information cannot easily be returned in the
        __getitem__ method. Therefore, the index is returned instead, and the subject information can be extracted by
        passing the index to this method.

        Parameters
        ----------
        item : torch.Tensor

        Returns
        -------
        Subject
        """
        # Get the dataset name and index
        dataset_sizes = {dataset_name: len(subjects) for dataset_name, subjects in self._subjects.items()}
        dataset_name, idx = _select_dataset_and_index(item=int(item), dataset_sizes=dataset_sizes)

        # Use correct type and return
        return Subject(subject_id=self._subjects[dataset_name][idx], dataset_name=dataset_name)

    def get_subjects_from_indices(self, items):
        """
        Get the subjects from the indices returned by the __getitem__ method (and later collated).

        Parameters
        ----------
        items : torch.Tensor

        Returns
        -------
        tuple[Subject, ...]
        """
        return tuple(self.get_subject_from_idx(item=item) for item in items)

    def get_dataset_indices_from_subjects(self, subjects):
        """
        Get the dataset indices from a tuple of subjects

        Parameters
        ----------
        subjects : tuple[Subject, ...]

        Returns
        -------
        torch.Tensor
        """
        # Get the dictionary mapping from dataset name to dataset index
        dataset_mapping = self.dataset_indices

        # return indices as a torch tensor
        return torch.tensor([dataset_mapping[subject.dataset_name] for subject in subjects])

    # --------------
    # Properties
    # --------------
    @property
    def dataset_names(self):
        """Get the dataset names included in the data. The order is as the keys of the data passed to the __init__
        method"""
        return tuple(self._data.keys())

    @property
    def dataset_indices(self):
        """Get a dictionary mapping the dataset name to the dataset index"""
        return {dataset_name: i for i, dataset_name in enumerate(self._data)}


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


def strip_tensors(tensors, fill_val=-1):
    """
    Function which may be used to remove unused tensors

    This function changes the input in-place (meaning it is not actually important to use the return)

    Pro-tip: use this before sending the data to a GPU

    Parameters
    ----------
    tensors : dict[str, torch.Tensor]
    fill_val : int
        The value which was used to indicate that a tensor should not be there

    Returns
    -------
    dict[str, torch.Tensor]

    Examples
    --------
    >>> my_fill = -1
    >>> tensor_a = torch.rand(size=(1, 5, 30))
    >>> tensor_b = torch.rand(size=(1, 3, 30))
    >>> tensor_c = torch.rand(size=(1, 6, 30))
    >>> my_tensors = {"a": torch.cat((tensor_a, torch.ones(size=(1, 5, 30)) * my_fill,
    ...                               torch.ones(size=(1, 5, 30)) * my_fill), dim=0),
    ...               "b": torch.cat((torch.ones(size=(1, 3, 30)) * my_fill, torch.ones(size=(1, 3, 30)) * my_fill,
    ...                               tensor_b), dim=0),
    ...               "c": torch.cat((torch.ones(size=(1, 6, 30)) * my_fill, tensor_c,
    ...                               torch.ones(size=(1, 6, 30)) * my_fill), dim=0),
    ...               "d": torch.cat((torch.ones(size=(1, 11, 30)) * my_fill, torch.ones(size=(1, 11, 30)) * my_fill,
    ...                               torch.ones(size=(1, 11, 30)) * my_fill), dim=0)}
    >>> my_stripped_tensors = strip_tensors(my_tensors)
    >>> tuple(my_stripped_tensors.keys()), tuple(my_stripped_tensors.keys())  # Left out dataset 'd'
    (('a', 'b', 'c'), ('a', 'b', 'c'))

    The operations were also made in-place (the input dict is changed as well)

    >>> all(torch.equal(new_tensor, old_tensor) for new_tensor, old_tensor  # type: ignore[attr-defined]
    ...     in zip(my_stripped_tensors.values(), my_tensors.values()))
    True
    """
    # Loop through all datasets. Changing values while iterating is ok, inserting/deleting is not. Thanks, James
    # 'mCoding' Murphy (Sec. 13): https://www.youtube.com/watch?v=E8NijUYfyus
    to_delete = set()
    for dataset_name, x in tensors.items():
        # Get the indices of which indices to keep
        ghost_tensor = torch.ones(size=x.size()[1:]) * fill_val
        kept_indices = [i for i, tensor in enumerate(x) if not torch.equal(tensor, ghost_tensor)]

        # If no data is supposed to be used in the batch, the dataset should be deleted. Otherwise, keep only the real
        # ones
        if not kept_indices:
            to_delete.add(dataset_name)
        else:
            tensors[dataset_name] = x[kept_indices]

    # Delete
    for dataset_name in to_delete:
        del tensors[dataset_name]

    # Return the dictionary of tensors (although the operations also happen in-place)
    return tensors
