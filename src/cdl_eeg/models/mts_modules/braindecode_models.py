"""
Models provided by Braindecode are implemented here

todo: cite
todo: not sure about automatic activation function?
"""
from braindecode.models import EEGNetv4, EEGResNet, ShallowFBCSPNet, Deep4Net

from cdl_eeg.models.mts_modules.mts_module_base import MTSModuleBase


class EEGNetv4MTS(MTSModuleBase):
    """
    EEGNetv4

    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A Compact
    Convolutional Network for EEG-based Brain-Computer Interfaces. arXiv preprint arXiv:1611.08024.

    Examples
    --------
    >>> _ = EEGNetv4MTS(in_channels=4, num_classes=8, num_time_steps=300)
    """

    def __init__(self, in_channels, num_classes, num_time_steps, **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = EEGNetv4(in_chans=in_channels, n_classes=num_classes, input_window_samples=num_time_steps,
                               **kwargs)

    def forward(self, x):
        return self._model(x)


class EEGResNetMTS(MTSModuleBase):
    """
    EEGResNet

    todo: missing reference also in braindecode

    Examples
    --------
    >>> _ = EEGResNetMTS(in_channels=4, num_classes=8, num_time_steps=300)
    """

    def __init__(self, in_channels, num_classes, num_time_steps, final_pool_length="auto", n_first_filters=6,
                 **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        #
        # todo: default value of n_first_filters not provided by braindecode
        # ----------------
        self._model = EEGResNet(in_chans=in_channels, n_classes=num_classes, input_window_samples=num_time_steps,
                                final_pool_length=final_pool_length, n_first_filters=n_first_filters,
                                **kwargs)

    def forward(self, x):
        return self._model(x)


class ShallowFBCSPNetMTS(MTSModuleBase):
    """
    ShallowFBCSPNetMTS

    This was used in Engemann et al. (2022).

    Examples
    --------
    >>> _ = ShallowFBCSPNetMTS(4, 7, 200)
    """

    def __init__(self, in_channels, num_classes, num_time_steps, **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = ShallowFBCSPNet(in_chans=in_channels, n_classes=num_classes, input_window_samples=num_time_steps,
                                      **kwargs)

    def forward(self, x):
        return self._model(x)


class Deep4NetMTS(MTSModuleBase):
    """
    Deep4Net

    This was used in Engemann et al. (2022).

    Examples
    --------
    >>> _ = Deep4NetMTS(19, 3, 1000)

    The number of input time samples must be greater than 440

    >>> _ = Deep4NetMTS(19, 3, 440)
    Traceback (most recent call last):
    ...
    RuntimeError: Given input size: (200x2x1). Calculated output size: (200x0x1). Output size is too small
    """

    def __init__(self, in_channels, num_classes, num_time_steps, final_conv_length="auto", **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = Deep4Net(in_chans=in_channels, n_classes=num_classes, input_window_samples=num_time_steps,
                               final_conv_length=final_conv_length, **kwargs)

    def forward(self, x):
        return self._model(x)

