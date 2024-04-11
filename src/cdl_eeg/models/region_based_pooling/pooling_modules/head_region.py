"""
Implementation of RBP with head region
"""
import random
from itertools import cycle

import torch
from torch import nn

from cdl_eeg.data.datasets.dataset_base import channel_names_to_indices
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import MultiMontageSplitsPoolingBase, \
    precomputing_method
from cdl_eeg.models.region_based_pooling.pooling_modules.univariate_rocket import RocketConv1d


class MultiMSSharedRocketHeadRegion(MultiMontageSplitsPoolingBase):
    """
    RBP with head region using features from ROCKET shared across montage splits

    Examples
    --------
    >>> MultiMSSharedRocketHeadRegion((3, 4), num_kernels=100, latent_search_features=32, max_receptive_field=123,
    ...                               share_search_receiver_modules=True, head_region_indices=(1, 3),
    ...                               bias=False)  # doctest: +ELLIPSIS
    MultiMSSharedRocketHeadRegion(
      (_rocket): RocketConv1d(
        ...
      )
      (_receiver_modules): ModuleList(
        (0): ModuleList(
          (0-1): 2 x Linear(in_features=200, out_features=32, bias=False)
        )
        (1): ModuleList(
          (0-2): 3 x Linear(in_features=200, out_features=32, bias=False)
        )
      )
      (_head_region_self_modules): ModuleList(
        (0-1): 2 x Linear(in_features=200, out_features=1, bias=False)
      )
      (_head_search_gates): ModuleList(
        (0): ModuleList(
          (0-1): 2 x Linear(in_features=200, out_features=32, bias=False)
        )
        (1): ModuleList(
          (0-2): 3 x Linear(in_features=200, out_features=32, bias=False)
        )
      )
    )

    If the search and reciever modules are not shared, there will be an additional linear layer for the search module

    >>> MultiMSSharedRocketHeadRegion((3, 4), num_kernels=100, latent_search_features=32, max_receptive_field=123,
    ...                               share_search_receiver_modules=False, head_region_indices=(1, 3),
    ...                               bias=False)  # doctest: +ELLIPSIS
    MultiMSSharedRocketHeadRegion(
      (_rocket): RocketConv1d(
        ...
      )
      (_receiver_modules): ModuleList(
        (0): ModuleList(
          (0-1): 2 x Linear(in_features=200, out_features=32, bias=False)
        )
        (1): ModuleList(
          (0-2): 3 x Linear(in_features=200, out_features=32, bias=False)
        )
      )
      (_head_region_self_modules): ModuleList(
        (0-1): 2 x Linear(in_features=200, out_features=1, bias=False)
      )
      (_head_search_linear): ModuleList(
        (0): ModuleList(
          (0-1): 2 x Linear(in_features=200, out_features=32, bias=False)
        )
        (1): ModuleList(
          (0-2): 3 x Linear(in_features=200, out_features=32, bias=False)
        )
      )
      (_head_search_gates): ModuleList(
        (0): ModuleList(
          (0-1): 2 x Linear(in_features=200, out_features=32, bias=False)
        )
        (1): ModuleList(
          (0-2): 3 x Linear(in_features=200, out_features=32, bias=False)
        )
      )
    )

    This class supports pre-computing

    >>> MultiMSSharedRocketHeadRegion((3, 4), num_kernels=100, latent_search_features=32, max_receptive_field=123,
    ...                               share_search_receiver_modules=True, head_region_indices=(1, 3),
    ...                               bias=False).supports_precomputing()
    True
    """

    def __init__(self, num_regions, *, num_kernels, latent_search_features, share_search_receiver_modules, bias,
                 max_receptive_field, head_region_indices=None, seed=None):
        """
        Initialise

        Parameters
        ----------
        num_regions : tuple[str, ...]
            The number of regions for all montage splits. len(num_regions) should equal the number of montage splits
        num_kernels: int
            Number of ROCKET kernels to use
        latent_search_features : int
            The dimensionality of the search vector
        head_region_indices : tuple[int, ...], optional
            The index of which region to be used as head region, for all montage splits. Should be passed as an integer,
            not as a RegionID
        max_receptive_field : int
            Maximum receptive field of the ROCKET kernels
        share_search_receiver_modules : bool
            If the embeddings of the head region for computing search vectors should be the same as the module for
            comparing these with the receiver vectors of the non-head regions
        bias : bool
            To include bias term in the linear layers (True) or not (False)
        seed : int, optional
            Seed for reproducibility purposes. If specified, it will be used to initialise the random number generators
            of numpy, random (python built-in), and pytorch
        """
        super().__init__()

        # Maybe sample the head region indices
        if head_region_indices is None:
            head_region_indices = tuple(random.randint(0, num_regs - 1) for num_regs in num_regions)

        # ---------------
        # Input checks
        # ---------------
        # Check that the selected head regions do not exceed the number of regions, for all montage splits
        if not all(head_idx < regions for head_idx, regions in zip(head_region_indices, num_regions)):
            _num_wrong_head_indices = sum(head_idx >= regions for head_idx, regions in zip(head_region_indices,
                                                                                           num_regions))
            raise ValueError(f"The index of the head region cannot exceed the number of regions in a montage split. "
                             f"This error was found in {_num_wrong_head_indices} montage split(s)")

        # ---------------
        # Create module for pre-computing the ROCKET features
        # ---------------
        self._rocket = RocketConv1d(num_kernels=num_kernels, max_receptive_field=max_receptive_field, seed=seed)

        # ---------------
        # Create modules for non-head regions. They will 'receive' search vectors
        # ---------------
        # Create modules for each montage split
        receiver_modules = []
        for num_regs in num_regions:
            # Assuming ROCKET-features
            receiver_modules.append(
                nn.ModuleList([nn.Linear(in_features=num_kernels * 2, out_features=latent_search_features, bias=bias)
                               for _ in range(num_regs - 1)])  # Subtracting one head region
            )
        self._receiver_modules = nn.ModuleList(receiver_modules)

        # ---------------
        # Create modules for head regions
        #
        # They need to both create search vector embeddings and region representations of themselves
        # ---------------
        # Modules for computing region representations of themselves
        self._head_region_self_modules = nn.ModuleList([nn.Linear(in_features=num_kernels*2, out_features=1, bias=bias)
                                                        for _ in num_regions])

        # Modules for computing search vectors
        if share_search_receiver_modules:
            # Will simply use the receiver_modules instead
            self._head_search_linear = None
        else:
            # Create modules for each montage split
            head_search_linear = []
            for num_regs in num_regions:
                head_search_linear.append(
                    nn.ModuleList([nn.Linear(in_features=num_kernels * 2, out_features=latent_search_features,
                                             bias=bias) for _ in range(num_regs - 1)])  # Subtracting one head region
                )
            self._head_search_linear = nn.ModuleList(head_search_linear)

        # Modules which acts as gates
        head_search_gates = []
        for num_regs in num_regions:
            head_search_gates.append(nn.ModuleList(
                [nn.Linear(in_features=num_kernels * 2, out_features=latent_search_features, bias=bias)
                 for _ in range(num_regs - 1)])
            )
        self._head_search_gates = nn.ModuleList(head_search_gates)

        # ---------------
        # Store some attributes
        # ---------------
        self._head_region_indices = head_region_indices

    @precomputing_method
    def pre_compute(self, input_tensors):
        """
        Method for pre-computing the ROCKET features

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
            A tensor per dataset with dimensions=(batch, channel, time_steps). Note that the dimensions are pr. dataset,
            and can vary between the different datasets. The keys are dataset names

        Returns
        -------
        dict[str, torch.Tensor]
            ROCKET features, with shape=(batch, channels, num_features) pr. dataset with num_features=2 for the current
            implementation. Batch and channel dimension may vary for each dataset. The keys are dataset names
        """
        return {dataset_name: self._rocket(x) for dataset_name, x in input_tensors.items()}

    def _forward_single_dataset(self, x, *, pre_computed, channel_splits, channel_name_to_index):
        # --------------
        # Loop through all montage splits
        # --------------
        outputs = []
        for channel_split, head_region_idx, receivers, head_region_self_module, search_linear, search_gates \
                in zip(channel_splits,  # type: ignore[call-overload]
                       self._head_region_indices,
                       self._receiver_modules,
                       self._head_region_self_modules,
                       cycle((None,)) if self._head_search_linear is None else self._head_search_linear,
                       self._head_search_gates):
            # Input check
            num_regions = len(receivers) + 1  # Need to add one for the head region
            assert len(channel_split) == num_regions, (f"Expected {num_regions} number of regions, but input "
                                                       f"channel split suggests {len(channel_split)}")

            # Initialise tensor which will contain all region representations of the montage split
            batch, _, time_steps = x.size()
            region_representations = torch.empty(size=(batch, num_regions, time_steps)).to(x.device)

            # Compute channel indices of head region  todo: poor variable naming?
            ch_names = tuple(channel_split.values())
            head_region_indices = channel_names_to_indices(
                ch_names=ch_names[head_region_idx],
                channel_name_to_index=channel_name_to_index
            )

            # --------------
            # Loop through all non-head regions
            # --------------
            non_head_ch_names = list(ch_names)
            non_head_ch_names.pop(head_region_idx)
            for i, (legal_ch_names, receiver_module, search_gate, linear) \
                    in enumerate(zip(non_head_ch_names, receivers, search_gates,
                                     cycle((None,)) if search_linear is None else search_linear)):
                # Extract the indices of the legal channels for this region
                allowed_node_indices = channel_names_to_indices(ch_names=legal_ch_names,
                                                                channel_name_to_index=channel_name_to_index)

                # Compute receive vectors. Will have shape=(batch, channel, latent)
                receive_vectors = receiver_module(pre_computed[:, allowed_node_indices])

                # Compute search vector
                linear_mapping = receiver_module if linear is None else linear

                search_linear_vector = linear_mapping(pre_computed[:, head_region_indices])
                search_gate_vector = search_gate(pre_computed[:, head_region_indices])

                search_vector = torch.sum(search_linear_vector * torch.softmax(search_gate_vector, dim=1),
                                          dim=1, keepdim=True)

                # Compute similarities with search vector, and normalise by softmax
                normalised_similarities = torch.softmax(torch.cosine_similarity(receive_vectors, search_vector, dim=-1),
                                                        dim=-1)

                # Compute region representation
                region_representations[:, i] = torch.squeeze(
                    torch.matmul(torch.unsqueeze(normalised_similarities, dim=1), x[:, allowed_node_indices]),
                    dim=1
                )

            # -------------------
            # Compute region representation for head region
            # -------------------
            head_attention = torch.softmax(head_region_self_module(pre_computed[:, head_region_indices]), dim=1)

            # Insert head region representation as the last region
            region_representations[:, -1] = torch.squeeze(torch.matmul(torch.transpose(head_attention, dim0=1, dim1=2),
                                                                       x[:, head_region_indices]), dim=1)

            # Append as output
            outputs.append(region_representations)

        return outputs

    def forward(self, input_tensors, *, pre_computed, channel_splits, channel_name_to_index):
        """
        Forward method

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
        pre_computed : dict[str, torch.Tensor]
        channel_splits : dict[str, tuple[cdl_eeg.models.region_based_pooling.utils.CHANNELS_IN_MONTAGE_SPLIT, ...]]
        channel_name_to_index : dict[str, dict[str, int]]

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        # --------------
        # Input checks
        # --------------
        assert all(len(ch_splits) == self.num_montage_splits for ch_splits in channel_splits.values()), \
            (f"Expected {self.num_montage_splits} number of montage splits, but inputs suggests "
             f"{set(len(ch_splits) for ch_splits in channel_splits.values())}")

        # --------------
        # Loop through all datasets
        # todo: not sure if this is the best approach (triple for-loop!!)... maybe padding+masking is better?
        # --------------
        dataset_region_representations = []
        for dataset_name, x in input_tensors.items():
            # (I take no chances on all input being ordered similarly)
            dataset_pre_computed = pre_computed[dataset_name]
            dataset_ch_name_to_idx = channel_name_to_index[dataset_name]
            ch_splits = channel_splits[dataset_name]

            # Perform forward pass
            dataset_region_representations.append(
                self._forward_single_dataset(x, pre_computed=dataset_pre_computed, channel_splits=ch_splits,
                                             channel_name_to_index=dataset_ch_name_to_idx))

        # Concatenate the data together
        return tuple(torch.cat(tensor, dim=0) for tensor in list(zip(*dataset_region_representations)))

    # -------------
    # Properties
    # -------------
    @property
    def num_montage_splits(self):
        """Get the number of montage splits"""
        return len(self._receiver_modules)
