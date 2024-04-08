import numpy
import torch
from torch.utils.data import Dataset

from cdl_eeg.data.subject_split import Subject


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
        return sum(x.shape[0] * x.shape[1] for x in self._data.values())

    def __getitem__(self, item):
        # todo: if the __getitem__ is to be standardised, the input to the transformation methods must also be

        # Varying keys in the returned dictionary is not possible with the DataLoader of PyTorch. This solution to the
        # problem is to simply return a tensor of -1s for the datasets not used
        data = {dataset_name: torch.ones(size=x.shape[1:]) * (-1) for dataset_name, x in self._data.items()}
        all_details = {dataset_name: torch.unsqueeze(torch.tensor(-1), dim=-1)
                       for dataset_name in self._data}  # todo: hard-coded fill values

        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Perform permutation and get details
        transformed, details = self._transformation.transform(self._pretext_task)(
            self._data[dataset_name][subject_idx][epoch_idx]
        )

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
                my_dict[dataset_name] = pre_comp[dataset_name][subject_idx][epoch_idx]
                pre_computed.append(my_dict)

            # TODO: must fix return, as KeyError is raised when collating
            # Don't think I should convert pre_computed to tuple, as I must strip it anyway
            return data, pre_computed, all_details

    # ---------------
    # Properties
    # ---------------
    @property
    def dataset_shapes(self):
        return {x.shape for x in self._data.values()}

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
    """
    Pytorch dataset for downstream training of RBP models of type MainRBPModel

    todo: should change the name to something with RBP
    """

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
        # Input check
        if not all(x.ndim == 4 for x in data.values()):
            _all_sizes = set(x.ndim for x in data.values())
            raise ValueError(f"Expected all input arrays to be 4D with dimensions (subjects, EEG epochs, channels, "
                             f"time_steps), but found {_all_sizes}")

        super().__init__()

        self._data = data
        self._targets = targets
        self._subjects = subjects
        self._pre_computed = pre_computed

    def __len__(self):
        return sum(x.shape[0] * x.shape[1] for x in self._data.values())

    def __getitem__(self, item):
        # TODO: copied from SelfSupervisedDataGenerator

        # Varying keys in the returned dictionary is not possible with the DataLoader of PyTorch. This solution to the
        # problem is to simply return a tensor of -1s for the datasets not used
        data = {dataset_name: torch.ones(size=x.shape[1:]) * (-1) for dataset_name, x in self._data.items()}
        targets = {dataset_name: torch.unsqueeze(torch.ones(size=y.shape[1:]) * (-1), dim=-1)
                   for dataset_name, y in self._targets.items()}

        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used
        data[dataset_name] = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets[dataset_name] = torch.unsqueeze(torch.tensor(self._targets[dataset_name][subject_idx],
                                                             dtype=torch.float, requires_grad=False),
                                                dim=-1)

        # TODO: quite hard coded?
        if self._pre_computed is None:
            return data, targets, item
        else:
            # assert False, {type(pre_comp) for pre_comp in self._pre_computed}
            pre_computed = []
            for pre_comp in self._pre_computed:
                my_dict = {data_name: torch.ones(tensor.size()[1:]) * (-1) for data_name, tensor in pre_comp.items()}
                my_dict[dataset_name] = pre_comp[dataset_name][subject_idx][epoch_idx]
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
        dataset_name, subject_idx, _ = _select_dataset_and_index(item=int(item), dataset_shapes=self.dataset_shapes)

        # Use correct type and return
        return Subject(subject_id=self._subjects[dataset_name][subject_idx], dataset_name=dataset_name)

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
    def dataset_shapes(self):
        return {dataset_name: x.shape for dataset_name, x in self._data.items()}

    @property
    def dataset_sizes(self):
        """Get the sizes of the datasets. The keys are the dataset names, the values are the number of subjects in the
        dataset"""
        return {dataset_name: x.shape[0] for dataset_name, x in self._data.items()}

    @property
    def dataset_indices(self):
        """Get a dictionary mapping the dataset name to the dataset index"""
        return {dataset_name: i for i, dataset_name in enumerate(self._data)}


class InterpolationDataGenerator(Dataset):  # type: ignore[type-arg]
    """
    Pytorch dataset for downstream training of models which require interpolation for spatial dimension consistency

    Examples
    --------
    >>> import numpy
    >>> my_data = {"d1": numpy.random.rand(3, 7, 300), "d2": numpy.random.rand(4, 7, 300),
    ...            "d3": numpy.random.rand(1, 7, 300)}
    >>> my_targets = {"d1": numpy.random.rand(3), "d2": numpy.random.rand(4), "d3": numpy.random.rand(1)}
    >>> my_subjects = {"d1": (Subject("P1", "d1"), Subject("P2", "d1"), Subject("P3", "d1")),
    ...                "d2": (Subject("P1", "d2"), Subject("P2", "d2"), Subject("P3", "d2"), Subject("P4", "d2")),
    ...                "d3": (Subject("P1", "d2"),)}
    >>> _ = InterpolationDataGenerator(my_data, my_targets, my_subjects)

    A ValueError is raise if spatial dimension is inconsistent

    >>> my_data["d2"] = numpy.random.rand(4, 77, 300)
    >>> InterpolationDataGenerator(my_data, my_targets, my_subjects)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ValueError: Expected spatial dimension consistency of all EEG data passed, as the data should already be
    interpolated. Instead, the following shapes were found {'d1': (3, 7, 300), 'd2': (4, 77, 300), 'd3': (1, 7, 300)}
    """

    # --------------
    # Magic/dunder methods
    # --------------
    def __init__(self, data, targets, subjects):
        """
        Initialise

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            The data should be interpolated prior to passing them to this method
        targets : dict[str, numpy.ndarray]
        subjects : dict[str, tuple[str, ...]]
        """
        # Input check (data should already be interpolated). Thus checking spatial dimension consistency
        if not len(set(eeg_data.shape[1] for eeg_data in data.values())) == 1:
            _all_shapes = {dataset: eeg_data.shape for dataset, eeg_data in data.items()}
            raise ValueError(f"Expected spatial dimension consistency of all EEG data passed, as the data should "
                             f"already be interpolated. Instead, the following shapes were found {_all_shapes}")

        super().__init__()

        self._data = data
        self._targets = targets
        self._subjects = subjects

    def __len__(self):
        return sum(x.shape[0] * x.shape[1] for x in self._data.values())

    def __getitem__(self, item):
        # Varying keys in the returned dictionary is not possible with the DataLoader of PyTorch. This solution to the
        # problem is to simply return a tensor of -1s for the datasets not used
        data = {dataset_name: torch.ones(size=x.shape[1:]) * (-1) for dataset_name, x in self._data.items()}
        targets = {dataset_name: torch.unsqueeze(torch.ones(size=y.shape[1:]) * (-1), dim=-1)
                   for dataset_name, y in self._targets.items()}

        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used
        data[dataset_name] = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets[dataset_name] = torch.unsqueeze(torch.tensor(self._targets[dataset_name][subject_idx],
                                                             dtype=torch.float, requires_grad=False),
                                                dim=-1)

        # Return input data, targets, and the item (will be converted to Subject later)
        return data, targets, item

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
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item=int(item),
                                                                         dataset_shapes=self.dataset_shapes)

        # Use correct type and return
        return Subject(subject_id=self._subjects[dataset_name][subject_idx], dataset_name=dataset_name)

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
    def dataset_shapes(self):
        return {dataset_name: x.shape for dataset_name, x in self._data.items()}

    @property
    def dataset_sizes(self):
        """Get the sizes of the datasets. The keys are the dataset names, the values are the number of subjects in the
        dataset"""
        return {dataset_name: x.shape[0] for dataset_name, x in self._data.items()}

    @property
    def dataset_indices(self):
        """Get a dictionary mapping the dataset name to the dataset index"""
        return {dataset_name: i for i, dataset_name in enumerate(self._data)}


# ----------------
# Functions
# ----------------
def _select_dataset_and_index(item, dataset_shapes):
    """
    Function for selecting dataset. Only works for positive integer items

    Parameters
    ----------
    item : int
    dataset_shapes : dict[str, tuple[int, ...]]

    Returns
    -------
    tuple[str, int]

    Examples
    --------
    >>> my_shapes = {"a": (3, 3, 19, 3000), "b": (4, 4, 19, 3000), "c": (6, 1, 19, 3000), "d": (7, 2, 19, 3000)}
    >>> _select_dataset_and_index(item=15, dataset_shapes=my_shapes)
    ('b', 1, 2)
    >>> _select_dataset_and_index(item=27, dataset_shapes=my_shapes)
    ('c', 2, 0)
    >>> _select_dataset_and_index(item=36, dataset_shapes=my_shapes)
    ('d', 2, 1)
    >>> _select_dataset_and_index(item=44, dataset_shapes=my_shapes)
    ('d', 6, 1)
    >>> _select_dataset_and_index(item=45, dataset_shapes=my_shapes)
    Traceback (most recent call last):
    ...
    IndexError: Index 45 exceeds the total size of the combined dataset 45
    >>> _select_dataset_and_index(item=-1, dataset_shapes=my_shapes)
    Traceback (most recent call last):
    ...
    ValueError: Expected item to be a positive integer, but found -1 (type=<class 'int'>)
    """
    # Input check
    if not isinstance(item, int) or item < 0:
        raise ValueError(f"Expected item to be a positive integer, but found {item} (type={type(item)})")

    # Find the dataset name and position
    accumulated_sizes = 0
    for name, shape in dataset_shapes.items():
        num_subjects, num_eeg_epochs, *_ = shape
        size = num_subjects * num_eeg_epochs
        accumulated_sizes += size
        if item < accumulated_sizes:
            # Now, the current dataset is the correct one. Need to extract the correct subject and EEG epoch indices
            idx = item - (accumulated_sizes - size)

            subject_idx, eeg_epoch_idx = numpy.divmod(idx, num_eeg_epochs)
            return name, subject_idx, eeg_epoch_idx

    # This should not happen...
    raise IndexError(f"Index {item} exceeds the total size of the combined dataset {accumulated_sizes}")


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
