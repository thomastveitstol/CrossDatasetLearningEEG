import dataclasses
from typing import Dict, NamedTuple, Tuple


# --------------------
# Convenient classes for regions and channels
# --------------------
class CartCoord(NamedTuple):
    """Class for cartesian coordinates, such as electrode positions"""
    coordinates: Tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class RegionID:
    id: int


class ChannelsInRegion(NamedTuple):
    """
    Use this class to store the channel names inside a region. The object created should contain all channel in
    R^(i)_j cap C, following the notation from the Region Based Pooling paper (Tveitstøl et al., 2023, submitted)
    """
    ch_names: Tuple[str, ...]

    def __len__(self) -> int:
        """The length of a ChannelsInRegion() object should correspond to the number of channels in that region (given
        the channel system). Following the notation from the Region Based Pooling paper (Tveitstøl et al., 2023,
        submitted) this is |R^(i)_j cap C|"""
        return len(self.ch_names)


@dataclasses.dataclass(frozen=True)
class ChannelsInRegionSplit:
    """
    Use this class to store the channel names inside the regions of a channel split. Following the notation from the
    Region Based Pooling paper (Tveitstøl et al., 2023, submitted), it is in mathematical terms
    {(R^(i)_1 cap C, R^(i)_2 cap C, ..., R^(i)_n cap C)}
    """
    ch_names: Dict[RegionID, ChannelsInRegion]  # As of Python version >= 3.7, dicts are ordered

    def __len__(self) -> int:
        """The length of a ChannelsInRegionSplit() object should correspond to the number of regions"""
        return len(self.ch_names)
