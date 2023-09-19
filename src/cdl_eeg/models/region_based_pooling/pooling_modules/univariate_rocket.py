"""
All ROCKET-based pooling modules are implemented here.

Original paper:
https://link.springer.com/article/10.1007/s10618-020-00701-z

This code is likely to have overlap with a former implementation of mine (Thomas Tveitstøl):
https://github.com/thomastveitstol/RegionBasedPoolingEEG/
"""
import random

import numpy
import torch
import torch.nn as nn

from cdl_eeg.data.datasets.dataset_base import channel_names_to_indices
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import SingleChannelSplitPoolingBase


class SingleCSSharedRocket(SingleChannelSplitPoolingBase):
    """
    Pooling by linear combination of the channels, where the importance score is computed from ROCKET-based features and
    the ROCKET kernels are shared (at least) across all regions in the channel/region split. It operates on an entire
    channel/region split

    todo: this must be compatible in combination with e.g. SingleCSMean, otherwise use MultiChannelSplitsPoolingBase

    Examples
    --------
    >>> _ = SingleCSSharedRocket(4, num_kernels=100, max_receptive_field=200)
    """

    def __init__(self, num_regions, *, num_kernels, max_receptive_field, seed=None):
        super().__init__()

        # ----------------
        # Define ROCKET-feature extractor
        # ----------------
        self._rocket = RocketConv1d(num_kernels=num_kernels, max_receptive_field=max_receptive_field, seed=seed)

        # ----------------
        # Define mappings from ROCKET features
        # to importance scores, for all regions
        # ----------------
        self._fc_modules = nn.ModuleList([nn.Linear(in_features=num_kernels * 2, out_features=1)
                                          for _ in range(num_regions)])

    def pre_compute(self, x):
        """
        Method for pre-computing the ROCKET features

        Parameters
        ----------
        x : torch.Tensor
            A tensor with shape=(batch, channel, time_steps)

        Returns
        -------
        torch.Tensor
            ROCKET features, with shape=(batch, channels, num_features) with num_features=2 for the current
            implementation.

        Examples
        --------
        >>> my_data = torch.rand(size=(10, 64, 500))
        >>> SingleCSSharedRocket(6, num_kernels=123, max_receptive_field=50).pre_compute(my_data).size()
        torch.Size([10, 64, 246])
        """
        return self._rocket(x)

    def forward(self, x, *, pre_computed, channel_split, channel_name_to_index):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
            A tensor containing EEG data with shape=(batch, channels, time_steps). Note that the channels are correctly
            selected within this method, and the EEG data should be the full data matrix (such that
            channel_name_to_index maps correctly)
        pre_computed : torch.Tensor
            Pre-computed features of all channels (is in the input 'x') todo: can this be improved memory-wise?
        channel_split : cdl_eeg.models.region_based_pooling.utils.ChannelsInRegionSplit
        channel_name_to_index : dict[str, int]

        Returns
        -------
        torch.Tensor
        """
        # Input check
        assert len(channel_split) == self.num_regions, (f"Expected {self.num_regions} number of regions, but input "
                                                        f"channel split suggests {len(channel_split)}")

        # Initialise tensor which will contain all region representations
        batch, _, time_steps = x.size()
        region_representations = torch.empty(size=(batch, self.num_regions, time_steps)).to(x.device)

        # Loop through all regions
        for i, (fc_module, channels) in enumerate(zip(self._fc_modules, channel_split.ch_names.values())):
            # Extract the indices of the legal channels for this region
            allowed_node_indices = channel_names_to_indices(ch_names=channels.ch_names,
                                                            channel_name_to_index=channel_name_to_index)

            # ---------------------
            # Compute coefficients
            # ---------------------
            # Pass through FC module
            coefficients = fc_module(pre_computed[:, allowed_node_indices])

            # Normalise
            coefficients = torch.softmax(coefficients, dim=1)

            # --------------------------------
            # Apply attention vector on the EEG
            # data, and insert as a region representation
            # --------------------------------
            # Add it to the slots
            region_representations[:, i] = torch.matmul(torch.transpose(coefficients, dim0=1, dim1=2),
                                                        x[:, allowed_node_indices])

        return region_representations

    # -------------
    # Properties
    # -------------
    @property
    def num_regions(self):
        return len(self._fc_modules)


class RocketConv1d(nn.Module):
    """
    Class for computing ROCKET features. This implementation was the preferred one in the RBP paper, due to a good
    trade-off between memory and time-consumption. All parameters are freezed. as in the original paper

    This class is not a pooling method on its own

    Examples
    --------
    >>> _ = RocketConv1d(num_kernels=100, max_receptive_field=250)
    """

    def __init__(self, *, num_kernels, max_receptive_field, seed=None):
        """
        Initialise

        Parameters
        ----------
        num_kernels : int
            Number of ROCKET kernels to use
        max_receptive_field : int
            Maximum receptive field of the kernels
        seed : int, optional
            Seed for reproducibility purposes. If specified, it will be used to initialise the random number generators
            of numpy, random (python built-in), and pytorch
        """
        super().__init__()

        # (Maybe) set seed
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.random.manual_seed(seed)

        # ------------------
        # Define kernels
        # ------------------
        kernels = []
        for _ in range(num_kernels):
            # Sample dilation and kernel length
            kernel_length = _sample_kernel_length()
            dilation = _sample_dilation(max_receptive_field=max_receptive_field, kernel_length=kernel_length)

            # Define kernel (hyperparameters in_channels, out_channels and groups are somewhat misleading here, as they
            # are 'repeated' in the forward method instead)
            rocket_kernel = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_length, dilation=dilation,
                                      padding="same", groups=1)

            # Initialise weights
            _sample_weights(weights=rocket_kernel.weight.data)
            _sample_bias(bias=rocket_kernel.bias.data)  # type: ignore[union-attr]

            # Add to kernels list
            kernels.append(rocket_kernel)

        # Register kernels using module list
        self._kernels = nn.ModuleList(kernels)

        # ------------------
        # Freeze all parameters
        # ------------------
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward method. It essentially applies the same 1D convolutions (with kernels shape=(1, kernel_length), where
        kernel length varies from kernel to kernel) to all channels

        Parameters
        ----------
        x : torch.Tensor
            A tensor with shape=(batch, channels, time_steps). All input channels will be used, so if only a subset of
            the channels are meant to be convoluted, this must be fixed before passing the tensor to this method.

        Returns
        -------
        torch.Tensor
            ROCKET-like features, with shape=(batch, channels, num_features) with num_features=2 for the current
            implementation.

        Examples
        --------
        # todo: Test if changing one channel affects the output of the others (should not be the case)
        >>> my_model = RocketConv1d(num_kernels=321, max_receptive_field=123)
        >>> my_model(torch.rand(size=(10, 64, 500))).size()
        torch.Size([10, 64, 642])
        """
        # Initialise tensor. The features will be stored to this tensor
        batch, num_channels, _ = x.size()
        outputs = torch.empty(size=(batch, num_channels, 2 * self.num_kernels)).to(x.device)

        # Loop through all kernels
        for i, kernel in enumerate(self._kernels):
            # Perform convolution
            convoluted = nn.functional.conv1d(input=x, weight=kernel.weight.data.repeat(num_channels, 1, 1),
                                              bias=kernel.bias.data.repeat(num_channels), stride=1, padding="same",
                                              dilation=kernel.dilation, groups=num_channels)

            # Compute PPV and max values, and insert in the output tensor
            outputs[..., (2 * i):(2 * i + 2)] = compute_ppv_and_max(convoluted)

        # Return after looping through all kernels
        return outputs

    # --------------
    # Properties
    # --------------
    @property
    def num_kernels(self):
        return len(self._kernels)


# ---------------------
# Functions for sampling ROCKET parameters
# and hyperparameters
# ---------------------
def _sample_weights(weights):
    """
    Sample weights such as in the paper. The changes to the tensor is both in-place and returned
    The sampling of weights are done in two steps:
        1) sample every weight from a normal distribution, w ~ N(0, 1)
        2) Mean centre the weights, w = W - mean(W)

    Parameters
    ----------
    weights : torch.Tensor
        Weights to be initialised. In the future: consider passing only the shape
    Returns
    -------
    torch.Tensor
        A properly initialised tensor

    Examples
    --------
    >>> _ = torch.random.manual_seed(4)
    >>> my_weights = torch.empty(3, 5)
    >>> _sample_weights(weights=my_weights)
    tensor([[-1.9582, -0.1204,  1.8870,  0.4944,  0.8478],
            [-0.7544, -1.7789,  0.5511,  0.5028,  0.3360],
            [ 0.5321,  1.4178, -0.4338, -0.3016, -1.2216]])
    >>> my_weights  # The input weights are changed in-place due to mutability
    tensor([[-1.9582, -0.1204,  1.8870,  0.4944,  0.8478],
            [-0.7544, -1.7789,  0.5511,  0.5028,  0.3360],
            [ 0.5321,  1.4178, -0.4338, -0.3016, -1.2216]])
    """
    # Step 1) Sample weights from a normal distribution N(0, 1)
    weights = nn.init.normal_(weights, mean=0, std=1)

    # Step 2) Mean centre weights
    weights -= torch.mean(weights)

    return weights


def _sample_bias(bias):
    """
    Sample bias parameters. As in the paper, the weights are sampled from a uniform distribution, b ~ U(-1, 1). Note
    that the initialisation also happens in-place

    Parameters
    ----------
    bias : torch.Tensor
        A bias tensor. In the future: consider only using passing in input shape

    Returns
    -------
    torch.Tensor
        A properly initialised bias tensor

    Examples
    --------
    >>> _ = torch.random.manual_seed(4)
    >>> my_bias =  torch.empty(9)
    >>> _sample_bias(my_bias)
    tensor([ 0.1193,  0.1182, -0.8171, -0.5800, -0.9856, -0.9221,  0.9858,  0.8262,
             0.2372])
    >>> my_bias  # In-place initialisation as well
    tensor([ 0.1193,  0.1182, -0.8171, -0.5800, -0.9856, -0.9221,  0.9858,  0.8262,
             0.2372])
    """
    return nn.init.uniform_(bias, a=-1, b=1)


def _sample_kernel_length():
    """
    Following the original paper, the kernel length is selected randomly from {7, 9, 11} with equal probability. Note
    that by 'length' in this context, the number of elements is meant. That is, not taking dilation into account

    Returns
    -------
    int
        A value in {7, 9, 11}

    Examples
    --------
    >>> random.seed(1)
    >>> _sample_kernel_length()
    7
    """
    return random.choice((7, 9, 11))


def _sample_dilation(*, max_receptive_field, kernel_length):
    """
    Sample dilation. That is, d = floor(2**x) with x ~ U(0, A) with A as calculated in the paper. Due to the possibly
    very long input time series lengths in EEG, it rather uses a max_receptive_field as upper bound

    Parameters
    ----------
    max_receptive_field : int
        Maximum receptive field of the kernel
    kernel_length : int
        Length of kernel (in {7, 9, 11} in the paper)

    Returns
    -------
    int
        Dilation

    Examples
    --------
    >>> numpy.random.seed(3)
    >>> _sample_dilation(max_receptive_field=500, kernel_length=7)
    11
    >>> _sample_dilation(max_receptive_field=1000, kernel_length=9)
    30
    """
    # Set upper bound as in the ROCKET paper, with max_receptive_field instead of input length
    upper_bound = numpy.log2((max_receptive_field - 1) / (kernel_length - 1))

    # Sample from U(0, high)
    x = numpy.random.uniform(low=0, high=upper_bound)

    # Return floor of 2^x
    return int(2 ** x)


def compute_ppv_and_max(x):
    """
    Compute proportion of positive values (PPV) and max values

    Parameters
    ----------
    x : torch.Tensor
        A tensor with shape=(batch, channels, time_steps)

    Returns
    -------
    torch.Tensor
        Features of the time series. Output will have shape=(batch, channels, num_features) with num_features=2 for the
        current implementation.

    Examples
    >>> my_data = torch.rand(size=(10, 5, 300))
    >>> compute_ppv_and_max(my_data).size()  # type: ignore
    torch.Size([10, 5, 2])
    """
    # Compute PPV and max
    # todo: should see if I can optimise the computations further here
    ppv = torch.mean(torch.heaviside(x, values=torch.tensor(0., dtype=torch.float)), dim=-1)
    max_ = torch.max(x, dim=-1)[0]  # Keep only the values, not the indices

    # Concatenate and return
    return torch.cat([torch.unsqueeze(ppv, dim=-1), torch.unsqueeze(max_, dim=-1)], dim=-1)
