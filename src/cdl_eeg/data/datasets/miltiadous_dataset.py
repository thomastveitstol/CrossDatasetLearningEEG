from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, ChannelSystemBase


class MiltiadousChannelSystem(ChannelSystemBase):
    ...


class Miltiadous(EEGDatasetBase):

    def __init__(self):
        super().__init__(channel_system=MiltiadousChannelSystem())
