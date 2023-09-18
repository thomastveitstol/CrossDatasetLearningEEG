import mne

from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.models.region_based_pooling.region_based_pooling import SingleChannelRegionBasedPooling
from cdl_eeg.models.region_based_pooling.utils import Electrodes3D


def test_fit_channel_system():
    # ----------------
    # Generate dummy channel system
    # todo: check mCoding if this can be improved
    # ----------------
    channel_system_name = "TestName"
    x_min, x_max, y_min, y_max = -.17, .17, -.17, .17

    electrode_positions = Electrodes3D(mne.channels.make_standard_montage("GSN-HydroCel-129").get_positions()["ch_pos"])
    channel_name_to_index = {name: i for i, name in enumerate(electrode_positions.positions)}
    channel_system = ChannelSystem(name=channel_system_name, channel_name_to_index=channel_name_to_index,
                                   electrode_positions=electrode_positions)

    # ----------------
    # Make RBP object
    # todo: check mCoding if this can be improved
    # ----------------
    pooling_methods = ("SingleCSMean", "SingleCSMean", "SingleCSMean", "SingleCSMean")
    pooling_methods_kwargs = ({}, {}, {}, {})
    split_methods = ("VoronoiSplit", "VoronoiSplit", "VoronoiSplit", "VoronoiSplit")
    split_methods_kwargs = ({"num_points": 11, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
                            {"num_points": 7, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
                            {"num_points": 26, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
                            {"num_points": 18, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})

    rbp_module = SingleChannelRegionBasedPooling(pooling_methods=pooling_methods,
                                                 pooling_methods_kwargs=pooling_methods_kwargs,
                                                 split_methods=split_methods,
                                                 split_methods_kwargs=split_methods_kwargs)

    # ----------------
    # Fit channel system
    # ----------------
    rbp_module.fit_channel_system(channel_system=channel_system)

    assert channel_system_name in rbp_module.channel_splits, ("Expected the channel system to be registered as a "
                                                              "fitted channel system, but it was not found")
