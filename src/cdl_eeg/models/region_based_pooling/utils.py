import dataclasses
from typing import Dict, NamedTuple, Tuple


# --------------------
# Convenient classes for regions
# --------------------
@dataclasses.dataclass
class RegionID:
    id: int


class ChannelsInRegion(NamedTuple):
    """
    Use this class to store the channel names inside a region. The object created should contain all channel in
    R^(i)_j cap C, following the notation from the Region Based Pooling paper (Tveitstøl et al., 2023, submitted)
    """
    ch_names: Tuple[str, ...]


@dataclasses.dataclass
class ChannelsInRegionSplit:
    ch_names: Dict[RegionID, ChannelsInRegion]  # As of Python version >= 3.7, dicts are ordered
