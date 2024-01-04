"""
Models provided by Braindecode are implemented here

todo: cite
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

    How the model looks like (the softmax/LogSoftmax activation function has been removed)

    >>> EEGNetv4MTS(in_channels=4, num_classes=8, num_time_steps=300)  # doctest: +NORMALIZE_WHITESPACE
    EEGNetv4MTS(
      (_model): EEGNetv4(
        (ensuredims): Ensure4d()
        (dimshuffle): Expression(expression=_transpose_to_b_1_c_0)
        (conv_temporal): Conv2d(1, 8, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32), bias=False)
        (bnorm_temporal): BatchNorm2d(8, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (conv_spatial): Conv2dWithConstraint(8, 16, kernel_size=(4, 1), stride=(1, 1), groups=8, bias=False)
        (bnorm_1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (elu_1): Expression(expression=elu)
        (pool_1): AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        (drop_1): Dropout(p=0.25, inplace=False)
        (conv_separable_depth): Conv2d(16, 16, kernel_size=(1, 16), stride=(1, 1), padding=(0, 8), groups=16,
                                       bias=False)
        (conv_separable_point): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bnorm_2): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (elu_2): Expression(expression=elu)
        (pool_2): AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        (drop_2): Dropout(p=0.25, inplace=False)
        (conv_classifier): Conv2d(16, 8, kernel_size=(1, 9), stride=(1, 1))
        (permute_back): Expression(expression=_transpose_1_0)
        (squeeze): Expression(expression=squeeze_final_output)
      )
    )
    """

    def __init__(self, in_channels, num_classes, num_time_steps, **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = EEGNetv4(in_chans=in_channels, n_classes=num_classes, input_window_samples=num_time_steps,
                               **kwargs)

        # Remove LogSoftmax activation function
        del self._model.softmax

    def forward(self, x):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 300
        >>> my_model = EEGNetv4MTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        """
        return self._model(x)


class EEGResNetMTS(MTSModuleBase):
    """
    EEGResNet

    todo: missing reference also in braindecode

    Examples
    --------
    >>> _ = EEGResNetMTS(in_channels=4, num_classes=8, num_time_steps=300)

    How the model looks like (the softmax/LogSoftmax activation function has been removed)

    >>> EEGResNetMTS(in_channels=4, num_classes=8, num_time_steps=300)  # doctest: +NORMALIZE_WHITESPACE
    EEGResNetMTS(
      (_model): EEGResNet(
        (ensuredims): Ensure4d()
        (dimshuffle): Expression(expression=transpose_time_to_spat)
        (conv_time): Conv2d(1, 6, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        (conv_spat): Conv2d(6, 6, kernel_size=(1, 4), stride=(1, 1), bias=False)
        (bnorm): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_nonlin): Expression(expression=elu)
        (res_1_0): _ResidualBlock(
          (conv_1): Conv2d(6, 6, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
          (bn1): BatchNorm2d(6, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(6, 6, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
          (bn2): BatchNorm2d(6, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_1_1): _ResidualBlock(
          (conv_1): Conv2d(6, 6, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
          (bn1): BatchNorm2d(6, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(6, 6, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
          (bn2): BatchNorm2d(6, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_2_0): _ResidualBlock(
          (conv_1): Conv2d(6, 12, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), dilation=(2, 1))
          (bn1): BatchNorm2d(12, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(12, 12, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), dilation=(2, 1))
          (bn2): BatchNorm2d(12, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_2_1): _ResidualBlock(
          (conv_1): Conv2d(12, 12, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), dilation=(2, 1))
          (bn1): BatchNorm2d(12, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(12, 12, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), dilation=(2, 1))
          (bn2): BatchNorm2d(12, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_3_0): _ResidualBlock(
          (conv_1): Conv2d(12, 18, kernel_size=(3, 1), stride=(1, 1), padding=(4, 0), dilation=(4, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(4, 0), dilation=(4, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_3_1): _ResidualBlock(
          (conv_1): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(4, 0), dilation=(4, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(4, 0), dilation=(4, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_4_0): _ResidualBlock(
          (conv_1): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(8, 0), dilation=(8, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(8, 0), dilation=(8, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_4_1): _ResidualBlock(
          (conv_1): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(8, 0), dilation=(8, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(8, 0), dilation=(8, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_5_0): _ResidualBlock(
          (conv_1): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(16, 0), dilation=(16, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(16, 0), dilation=(16, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_5_1): _ResidualBlock(
          (conv_1): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(16, 0), dilation=(16, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(16, 0), dilation=(16, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_6_0): _ResidualBlock(
          (conv_1): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(32, 0), dilation=(32, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(32, 0), dilation=(32, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_6_1): _ResidualBlock(
          (conv_1): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(32, 0), dilation=(32, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(32, 0), dilation=(32, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_7_0): _ResidualBlock(
          (conv_1): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(64, 0), dilation=(64, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(64, 0), dilation=(64, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (res_7_1): _ResidualBlock(
          (conv_1): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(64, 0), dilation=(64, 1))
          (bn1): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
          (conv_2): Conv2d(18, 18, kernel_size=(3, 1), stride=(1, 1), padding=(64, 0), dilation=(64, 1))
          (bn2): BatchNorm2d(18, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (mean_pool): AdaptiveAvgPool2d(output_size=(1, 1))
        (conv_classifier): Conv2d(18, 8, kernel_size=(1, 1), stride=(1, 1))
        (squeeze): Expression(expression=squeeze_final_output)
      )
    )
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

        # Remove LogSoftmax activation function
        del self._model.softmax

    def forward(self, x):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 300
        >>> my_model = EEGResNetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        """
        return self._model(x)


class ShallowFBCSPNetMTS(MTSModuleBase):
    """
    ShallowFBCSPNetMTS

    This was used in Engemann et al. (2022).

    Examples
    --------
    >>> _ = ShallowFBCSPNetMTS(4, 7, 200)

    How the model looks like (the softmax/LogSoftmax activation function has been removed)

    >>> ShallowFBCSPNetMTS(4, 7, 200)  # doctest: +NORMALIZE_WHITESPACE
    ShallowFBCSPNetMTS(
      (_model): ShallowFBCSPNet(
        (ensuredims): Ensure4d()
        (dimshuffle): Expression(expression=transpose_time_to_spat)
        (conv_time): Conv2d(1, 40, kernel_size=(25, 1), stride=(1, 1))
        (conv_spat): Conv2d(40, 40, kernel_size=(1, 4), stride=(1, 1), bias=False)
        (bnorm): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_nonlin_exp): Expression(expression=square)
        (pool): AvgPool2d(kernel_size=(75, 1), stride=(15, 1), padding=0)
        (pool_nonlin_exp): Expression(expression=safe_log)
        (drop): Dropout(p=0.5, inplace=False)
        (conv_classifier): Conv2d(40, 7, kernel_size=(30, 1), stride=(1, 1))
        (squeeze): Expression(expression=squeeze_final_output)
      )
    )
    """

    def __init__(self, in_channels, num_classes, num_time_steps, **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = ShallowFBCSPNet(in_chans=in_channels, n_classes=num_classes, input_window_samples=num_time_steps,
                                      **kwargs)

        # Remove LogSoftmax activation function
        del self._model.softmax

    def forward(self, x):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 600*3
        >>> my_model = ShallowFBCSPNetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])  TODO: failing
        """
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

    How the model looks like (the softmax/LogSoftmax activation function has been removed)

    >>> Deep4NetMTS(19, 3, 1000)  # doctest: +NORMALIZE_WHITESPACE
    Deep4NetMTS(
      (_model): Deep4Net(
        (ensuredims): Ensure4d()
        (dimshuffle): Expression(expression=transpose_time_to_spat)
        (conv_time): Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
        (conv_spat): Conv2d(25, 25, kernel_size=(1, 19), stride=(1, 1), bias=False)
        (bnorm): BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_nonlin): Expression(expression=elu)
        (pool): MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        (pool_nonlin): Expression(expression=identity)
        (drop_2): Dropout(p=0.5, inplace=False)
        (conv_2): Conv2d(25, 50, kernel_size=(10, 1), stride=(1, 1), bias=False)
        (bnorm_2): BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlin_2): Expression(expression=elu)
        (pool_2): MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        (pool_nonlin_2): Expression(expression=identity)
        (drop_3): Dropout(p=0.5, inplace=False)
        (conv_3): Conv2d(50, 100, kernel_size=(10, 1), stride=(1, 1), bias=False)
        (bnorm_3): BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlin_3): Expression(expression=elu)
        (pool_3): MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        (pool_nonlin_3): Expression(expression=identity)
        (drop_4): Dropout(p=0.5, inplace=False)
        (conv_4): Conv2d(100, 200, kernel_size=(10, 1), stride=(1, 1), bias=False)
        (bnorm_4): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlin_4): Expression(expression=elu)
        (pool_4): MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        (pool_nonlin_4): Expression(expression=identity)
        (conv_classifier): Conv2d(200, 3, kernel_size=(7, 1), stride=(1, 1))
        (squeeze): Expression(expression=squeeze_final_output)
      )
    )
    """

    def __init__(self, in_channels, num_classes, num_time_steps, final_conv_length="auto", **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = Deep4Net(in_chans=in_channels, n_classes=num_classes, input_window_samples=num_time_steps,
                               final_conv_length=final_conv_length, **kwargs)

        # Remove LogSoftmax activation function
        del self._model.softmax

    def forward(self, x):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 600*3
        >>> my_model = Deep4NetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        """
        return self._model(x)
