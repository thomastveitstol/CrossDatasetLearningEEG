import dataclasses
from typing import Dict, Tuple, Union

import numpy
from mne.transforms import _cart_to_sph, _pol_to_cart


# --------------------
# Convenient classes for regions and channels
# --------------------
@dataclasses.dataclass(frozen=True)
class RegionID:
    id: Union[int, str]


# --------------------
# Types for type hinting
# --------------------
CHANNELS_NAMES = Tuple[str, ...]

ELECTRODES_2D = Dict[str, Tuple[float, float]]
ELECTRODES_3D = Dict[str, Tuple[float, float, float]]

CHANNELS_IN_MONTAGE_SPLIT = Dict[RegionID, CHANNELS_NAMES]


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
    cdl_eeg.models.region_based_pooling.utils.ELECTRODES_2D
        The 2D projection of the electrodes

    Examples
    --------
    >>> import mne
    >>> my_positions = mne.channels.make_standard_montage(kind="GSN-HydroCel-129").get_positions()["ch_pos"]
    >>> tuple(project_to_2d(my_positions).keys())[:3]
    ('E1', 'E2', 'E3')
    >>> tuple(project_to_2d(my_positions).values())[:3]
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
    return {channel_name: projection_2d for channel_name, projection_2d in zip(electrode_positions, out)}
