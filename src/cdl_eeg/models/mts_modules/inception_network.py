"""
Inception network is implemented. Defaults are set as the original Keras implementation.

Paper: https://arxiv.org/pdf/1909.04939.pdf
Original implementation in keras at https://github.com/hfawaz/InceptionTime

This implementation was authored by Thomas Tveitstøl (Oslo University Hospital) in a different project of mine
(https://github.com/thomastveitstol/RegionBasedPoolingEEG/), although some minor changes mainly in the documentation
have been made
"""
from typing import Optional

import torch
import torch.nn as nn

from cdl_eeg.models.mts_modules.mts_module_base import MTSModuleBase, latent_feature_extraction_method


# ---------------------------
# Sub-modules
# ---------------------------
class _InceptionModule(nn.Module):
    """
    Examples
    --------
    >>> _ = _InceptionModule(in_channels=9)
    """

    num_kernel_sizes = 3

    def __init__(self, in_channels, units=32, *, activation=None, use_bottleneck=True, max_kernel_size=40):
        """
        Initialise

        As opposed to the original keras implementation, strides is strictly set to 1 and cannot be specified to any
        other value. This is because setting padding='same' is not supported when strides are greater than 1

        Parameters
        ----------
        in_channels : int
            Number of expected input channels
        units : int
            Output (channel) dimension of the Conv layers. Equivalent to nb_filters in original keras implementation
        activation: typing.Callable, optional
            Activation function. If None is passed, no activation function will be used
        use_bottleneck : bool
            To use the first input_conv layer or not
        max_kernel_size : int
            Largest kernel size used. In the original keras implementation, the equivalent argument is stored as
            kernel_size - 1, the same is not done here
        """
        super().__init__()

        # Store selected activation function
        self._activation_function = _no_activation_function if activation is None else activation

        # -------------------------------
        # Define Conv layer maybe operating on
        # the input
        # -------------------------------
        self._input_conv: Optional[nn.Module]
        if use_bottleneck:
            out_channels = 32
            self._input_conv = nn.Conv1d(in_channels, out_channels=out_channels, kernel_size=1, padding="same",
                                         bias=False)
        else:
            self._input_conv = None
            out_channels = in_channels

        # -------------------------------
        # Define convolutional layers with different
        # kernel sizes (to be concatenated at the end)
        # -------------------------------
        kernel_sizes = (max_kernel_size // (2 ** i) for i in range(_InceptionModule.num_kernel_sizes))

        self._conv_list = nn.ModuleList([nn.Conv1d(in_channels=out_channels, out_channels=units,
                                                   kernel_size=kernel_size, stride=1, padding="same", bias=False)
                                         for kernel_size in kernel_sizes])

        # -------------------------------
        # Define Max pooling and conv layer to be
        # applied after max pooling
        # -------------------------------
        self._max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self._conv_after_max_pool = nn.Conv1d(in_channels=in_channels, out_channels=units, kernel_size=1,
                                              padding="same", bias=False)

        # Finally, define batch norm
        self._batch_norm = nn.BatchNorm1d(num_features=units * (len(self._conv_list) + 1))  # Must multiply due to
        # concatenation with all outputs from self._conv_list and self._con_after_max_pool

    def forward(self, x):
        """
        Forward method

        Parameters
        ----------
        x: torch.Tensor
            Input tensor with shape=(batch, channels, time steps)

        Returns
        -------
        torch.Tensor
            Output of inception module, with shape=(batch_size, feature_maps, time_steps)

        Examples
        --------
        >>> my_inception_module = _InceptionModule(in_channels=53, units=7)
        >>> my_inception_module(torch.rand(size=(10, 53, 345))).size()
        torch.Size([10, 28, 345])
        """
        # Maybe pass through input conv
        if self._input_conv is not None:
            inception_input = self._activation_function(self._input_conv(x))
        else:
            inception_input = torch.clone(x)

        # Pass through the conv layers with different kernel sizes
        outputs = []
        for conv_layer in self._conv_list:
            outputs.append(self._activation_function(conv_layer(inception_input)))

        # Pass input tensor through max pooling, followed by a conv layer
        max_pool_output = self._max_pool(x)
        outputs.append(self._activation_function(self._conv_after_max_pool(max_pool_output)))

        # Concatenate, add batch norm, apply Relu activation function and return
        x = torch.cat(outputs, dim=1)  # concatenate in channel dimension
        x = nn.functional.relu(self._batch_norm(x))

        return x


class _ShortcutLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Initialise

        Parameters
        ----------
        in_channels : int
            Expected number of input channels
        out_channels : int
            Expected number of channels of the tensor we want to add short layer output to (see Examples in
            forward method)
        """
        super().__init__()
        # Define Conv layer and batch norm
        self._conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same")
        self._batch_norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, input_tensor, output_tensor):
        """
        Forward method

        Parameters
        ----------
        input_tensor : torch.Tensor
            A tensor with shape=(batch, in_channels, time_steps), where in_channels is equal to what was passed to
            __init__
        output_tensor : torch.Tensor
            A torch.Tensor with shape=(batch, out_channels, time_steps), where out_channels is equal to what was passed
            to __init__

        Returns
        -------
        torch.Tensor
            Output of shortcut layer, with shape=(batch, feature_dimension, time_dimension)

        Examples
        --------
        >>> my_model = _ShortcutLayer(in_channels=43, out_channels=76)
        >>> my_model(input_tensor=torch.rand(size=(10, 43, 500)),
        ...          output_tensor=torch.rand(size=(10, 76, 500))).size()  # The size is the same as output_tensor
        torch.Size([10, 76, 500])
        >>> # Raises a RuntimeError if the tensors do not have expected shapes
        >>> my_model(input_tensor=torch.rand(size=(10, 43, 500)),
        ...          output_tensor=torch.rand(size=(10, 75, 500))).size()
        Traceback (most recent call last):
        ...
        RuntimeError: The size of tensor a (76) must match the size of tensor b (75) at non-singleton dimension 1
        """
        # Pass through conv layer and batch norm
        x = self._conv(input_tensor)
        x = self._batch_norm(x)

        # Add to output tensor, apply Relu and return
        return nn.functional.relu(x + output_tensor)


# ---------------------------
# Main module
# ---------------------------
class InceptionNetwork(MTSModuleBase):
    """
    Inception network

    Examples
    --------
    >>> InceptionNetwork(64, 5).supports_latent_feature_extraction()
    True

    How it looks like (but note that the ordering does not reflect the forward pass, as this is not a Sequential model)

    >>> InceptionNetwork(64, 5)
    InceptionNetwork(
      (_inception_modules): ModuleList(
        (0): _InceptionModule(
          (_input_conv): Conv1d(64, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
          (_conv_list): ModuleList(
            (0): Conv1d(32, 32, kernel_size=(40,), stride=(1,), padding=same, bias=False)
            (1): Conv1d(32, 32, kernel_size=(20,), stride=(1,), padding=same, bias=False)
            (2): Conv1d(32, 32, kernel_size=(10,), stride=(1,), padding=same, bias=False)
          )
          (_max_pool): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (_conv_after_max_pool): Conv1d(64, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
          (_batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1-5): 5 x _InceptionModule(
          (_input_conv): Conv1d(128, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
          (_conv_list): ModuleList(
            (0): Conv1d(32, 32, kernel_size=(40,), stride=(1,), padding=same, bias=False)
            (1): Conv1d(32, 32, kernel_size=(20,), stride=(1,), padding=same, bias=False)
            (2): Conv1d(32, 32, kernel_size=(10,), stride=(1,), padding=same, bias=False)
          )
          (_max_pool): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (_conv_after_max_pool): Conv1d(128, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
          (_batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (_shortcut_layers): ModuleList(
        (0-1): 2 x _ShortcutLayer(
          (_conv): Conv1d(64, 128, kernel_size=(1,), stride=(1,), padding=same)
          (_batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (_fc_layer): Linear(in_features=128, out_features=5, bias=True)
    )
    """

    def __init__(self, in_channels, num_classes, *, cnn_units=32, depth=6, use_bottleneck=True, activation=None,
                 max_kernel_size=40, use_residual=True):
        """
        Initialise

        Parameters
        ----------
        in_channels : int
            Expected number of input channels
        num_classes : int
            Output dimension of prediction. That is, the output of the forward method will have
            shape=(batch, num_classes)
        cnn_units : int
            Number of output channels of the Inception modules
        depth : int
            Number of Inception modules used
        use_bottleneck : bool
            Using bottleneck or not
        activation : typing.Callable, optional
            Activation function to use in Inception modules. If None, no activation function is used
        max_kernel_size : int
            Max kernel size of in Inception modules
        use_residual : bool
            To use Shortcut layers or not
        """
        # Call super method (should be that of nn.Module)
        super().__init__()

        # -----------------------------
        # Define Inception modules
        # -----------------------------
        output_channels = cnn_units * (_InceptionModule.num_kernel_sizes + 1)  # Output channel dim of inception modules
        self._inception_modules = nn.ModuleList(
            [_InceptionModule(in_channels=in_channel, units=cnn_units,
                              use_bottleneck=use_bottleneck, activation=activation,
                              max_kernel_size=max_kernel_size)
             for i, in_channel in enumerate([in_channels] + [output_channels]*(depth - 1))]
        )

        # -----------------------------
        # Define Shortcut layers
        # -----------------------------
        self._shortcut_layers: Optional[nn.ModuleList]
        if use_residual:
            # A shortcut layer should be used for every third inception module
            self._shortcut_layers = nn.ModuleList([_ShortcutLayer(in_channels=in_channels,
                                                                  out_channels=output_channels)
                                                   for _ in range(len(self._inception_modules) // 3)])
        else:
            self._shortcut_layers = None

        # -----------------------------
        # Define FC layer for output (global
        # average pooling is implemented in
        # forward method)
        # -----------------------------
        self._fc_layer = nn.Linear(in_features=output_channels, out_features=num_classes)

    @latent_feature_extraction_method
    def default_latent_feature_extraction(self, input_tensor):
        """
        Get the features right after performing global average pooling in temporal dimension

        Parameters
        ----------
        input_tensor : torch.Tensor
            A torch.Tensor with shape=(batch, output_channels), see __init__ for output channels

        Returns
        -------
        torch.Tensor
            A torch.Tensor with shape=(batch, output_channels), see __init__ for output channels

        Examples
        --------
        >>> my_model = InceptionNetwork(in_channels=43, num_classes=3, cnn_units=23)
        >>> my_model.default_latent_feature_extraction(torch.rand(size=(10, 43, 500))).size()
        torch.Size([10, 92])
        >>> my_model.extract_latent_features(torch.rand(size=(10, 43, 500))).size()
        torch.Size([10, 92])
        """
        return self(input_tensor, return_features=True)

    def forward(self, input_tensor, return_features=False):
        """
        Forward method of Inception

        Parameters
        ----------
        input_tensor : torch.Tensor
            A torch.Tensor with shape=(batch, channels, time steps)
        return_features : bool
            To return the features after computing Global Average Pooling in the temporal dimension (True) or the
            predictions (False)

        Returns
        -------
        torch.Tensor
            Predictions without activation function or features after computing Global Average Pooling in the temporal
            dimension

        Examples
        --------
        >>> my_model = InceptionNetwork(in_channels=43, num_classes=3)
        >>> my_model(torch.rand(size=(10, 43, 500))).size()
        torch.Size([10, 3])
        >>> my_model(torch.rand(size=(13, 43, 1000))).size()  # The model is compatible with different num time steps
        torch.Size([13, 3])

        Verify that it runs with other arguments specified

        >>> my_model = InceptionNetwork(in_channels=533, num_classes=2, cnn_units=43, depth=7, use_residual=False,
        ...                      use_bottleneck=False, activation=nn.functional.elu, max_kernel_size=8)
        >>> my_model(torch.rand(size=(11, 533, 400))).size()
        torch.Size([11, 2])
        >>> my_model(torch.rand(size=(11, 533, 400)), return_features=True).size()  # cnn_units * 4
        torch.Size([11, 172])
        """
        x = torch.clone(input_tensor)

        # Make shortcut layers iterable, if not None
        shortcut_layers = None if self._shortcut_layers is None else iter(self._shortcut_layers)

        for i, inception_module in enumerate(self._inception_modules):
            # Pass though Inception module
            x = inception_module(x)

            # If shortcut layers are included, use them for every third inception module
            if shortcut_layers is not None and i % 3 == 2:
                shortcut_layer = next(shortcut_layers)
                x = shortcut_layer(input_tensor=input_tensor, output_tensor=x)

        # Global Average Pooling in time dimension. Note that this operation allows a varied numer of time steps to be
        # used
        x = torch.mean(x, dim=-1)  # Averages the temporal dimension and obtains shape=(batch, channel_dimension)

        # Return the features if desired
        if return_features:
            return x

        # Pass through FC layer and return. No activation function used
        return self._fc_layer(x)


# ------------------
# Functions
# ------------------
def _no_activation_function(x: torch.Tensor) -> torch.Tensor:
    """This can be used as activation function if no activation function is wanted. It is typically more convenient to
    use this function, instead of handling activation functions of type None"""
    return x
