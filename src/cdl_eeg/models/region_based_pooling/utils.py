import dataclasses
from typing import Dict, NamedTuple, Tuple, Union

import numpy
from mne.transforms import _cart_to_sph, _pol_to_cart


# --------------------
# Convenient classes for regions and channels
# todo: in general, I don't think these classes are good...
# --------------------
@dataclasses.dataclass(frozen=True)
class Electrodes2D:
    """Class for 2D coordinates of multiple electrodes"""

    positions: Dict[str, Tuple[float, float]]

    def __getitem__(self, item):
        """
        Get item method

        Parameters
        ----------
        item : str

        Returns
        -------
        tuple[float, float]
        """
        return self.positions[item]


ELECTRODES_3D = Dict[str, Tuple[float, float, float]]


@dataclasses.dataclass(frozen=True)
class RegionID:
    id: Union[int, str]


# todo: somewhat annoying to work with these. e.g. 'ch_name' in ChannelsInRegion should run 'ch_name' in
#  ChannelsInRegion.ch_names
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

    # todo: update to 'montage split'
    """
    ch_names: Dict[RegionID, ChannelsInRegion]  # As of Python version >= 3.7, dicts are ordered

    def __len__(self) -> int:
        """The length of a ChannelsInRegionSplit() object should correspond to the number of regions"""
        return len(self.ch_names)


# --------------------
# Functions
# --------------------
def project_to_2d(electrode_positions):
    """
    Function for projecting 3D points to 2D, as done in MNE for plotting sensor location.

    Most of this code was taken from the _auto_topomap_coordinates function, to obtain the same mapping as MNE. Link to
    this function can be found at (source code):
    https://github.com/mne-tools/mne-python/blob/9e4a0b492299d3638203e2e6d2264ea445b13ac0/mne/channels/layout.py#L633

    Parameters
    ----------
    electrode_positions : cdl_eeg.models.region_based_pooling.utils.ELECTRODES_3D
        Electrodes to project

    Returns
    -------
    cdl_eeg.models.region_based_pooling.utils.Electrodes2D
        The 2D projection of the electrodes

    Examples
    --------
    >>> import mne
    >>> my_positions = mne.channels.make_standard_montage(kind="GSN-HydroCel-129").get_positions()["ch_pos"]
    >>> tuple(project_to_2d(my_positions).positions.keys())[:3]
    ('E1', 'E2', 'E3')
    >>> tuple(project_to_2d(my_positions).positions.values())[:3]
    (array([0.07890224, 0.0752648 ]), array([0.05601906, 0.07102252]), array([0.03470422, 0.06856416]))
    """
    # ---------------------------
    # Apply the same steps as _auto_topomap_coordinates
    # from MNE.transforms
    # ---------------------------
    cartesian_coords = _cart_to_sph(tuple(electrode_positions.values()))
    out = _pol_to_cart(cartesian_coords[:, 1:][:, ::-1])
    out *= cartesian_coords[:, [0]] / (numpy.pi / 2.)

    # Convert to Dict and return
    return Electrodes2D({channel_name: projection_2d for channel_name, projection_2d in
                         zip(electrode_positions, out)})
