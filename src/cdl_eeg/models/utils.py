import numpy
import torch


# ---------------
# Functions for training
# ---------------
def tensor_dict_to_device(tensors, device):
    """
    Send a dictionary containing tensors to device

    Parameters
    ----------
    tensors : dict[str, torch.Tensor]
    device : torch.device

    Returns
    -------
    dict[str, torch.Tensor]
    """
    # Input check
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError(f"Expected all values in the dictionary to be torch tensors, but found "
                        f"{set(type(tensor) for tensor in tensors.values())}")

    # Send to device and return
    return {dataset_name: tensor.to(device) for dataset_name, tensor in tensors.items()}


def flatten_targets(tensors):
    """
    Flatten the targets

    TODO: Make tests on the sorting
    Parameters
    ----------
    tensors : dict[str, torch.Tensor | numpy.ndarray]

    Returns
    -------
    torch.Tensor
    """
    # Maybe convert to torch tensors
    if all(isinstance(tensor, numpy.ndarray) for tensor in tensors.values()):
        tensors = {dataset_name: torch.tensor(tensor, dtype=torch.float) for dataset_name, tensor in tensors.items()}

    # Flatten  todo: why do we need to loop for converting to tuple??
    targets = torch.cat(tuple(tensor for tensor in tensors.values()), dim=0)

    return targets
