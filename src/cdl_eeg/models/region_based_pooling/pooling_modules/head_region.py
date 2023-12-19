"""
Implementation of RBP with head region
"""
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import MultiMontageSplitsPoolingBase


class MultiMSSharedRocketHeadRegion(MultiMontageSplitsPoolingBase):
    """
    RBP with head region using features from ROCKET shared across montage splits

    Examples
    --------
    >>> _ = MultiMSSharedRocketHeadRegion((3, 4), latent_search_features=32, head_region_indices=(1, 3))
    """

    def __init__(self, num_regions, *, num_kernels, latent_search_features, head_region_indices):
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
        head_region_indices : tuple[int, ...]
            The index of which region to be used as head region, for all montage splits. Should be passed as an integer,
            not as a RegionID
        """
        super().__init__()

        # ---------------
        # Input checks
        # ---------------
        # Check that the selected head regions do not exceed the number of regions, for all montage splits
        if not all(head_idx < regions for head_idx, regions in zip(head_region_indices, num_regions)):
            _num_wrong_head_indices = sum(head_idx >= regions for head_idx, regions in zip(head_region_indices,
                                                                                           num_regions))
            raise ValueError(f"The index of the head region cannot exceed the number of regions in a montage split. "
                             f"This error was found in {_num_wrong_head_indices} montage split(s)")
