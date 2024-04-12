"""
Models provided by Braindecode are implemented here

todo: cite
"""
import torch
from braindecode.models import EEGNetv4, EEGResNet, ShallowFBCSPNet, Deep4Net

from cdl_eeg.models.mts_modules.mts_module_base import MTSModuleBase
from cdl_eeg.models.random_search.sampling_distributions import sample_hyperparameter


class EEGNetv4MTS(MTSModuleBase):
    """
    EEGNetv4

    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A Compact
    Convolutional Network for EEG-based Brain-Computer Interfaces. arXiv preprint arXiv:1611.08024.

    Examples
    --------
    >>> _ = EEGNetv4MTS(in_channels=4, num_classes=8, num_time_steps=300)

    Latent features

    >>> EEGNetv4MTS(in_channels=4, num_classes=8, num_time_steps=3000).latent_features_dim
    1488

    How the model looks like (the softmax/LogSoftmax activation function has been removed)

    >>> EEGNetv4MTS(in_channels=4, num_classes=8, num_time_steps=3000)  # doctest: +NORMALIZE_WHITESPACE
    EEGNetv4MTS(
      (_model): EEGNetv4(
        (ensuredims): Ensure4d()
        (dimshuffle): Rearrange('batch ch t 1 -> batch 1 ch t')
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
        (final_layer): Sequential(
          (conv_classifier): Conv2d(16, 8, kernel_size=(1, 93), stride=(1, 1))
          (permute_back): Rearrange('batch x y z -> batch x z y')
          (squeeze): Expression(expression=squeeze_final_output)
        )
      )
    )
    """

    def __init__(self, in_channels, num_classes, num_time_steps, **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = EEGNetv4(n_chans=in_channels, n_outputs=num_classes, n_times=num_time_steps, **kwargs)

    def extract_latent_features(self, input_tensor):
        """
        Method for extracting latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_model = EEGNetv4MTS(in_channels=4, num_classes=8, num_time_steps=500)
        >>> my_model.extract_latent_features(torch.rand(size=(10, 4, 500))).size()
        torch.Size([10, 240])
        """
        return self(input_tensor, return_features=True)

    def classify_latent_features(self, input_tensor):
        """
        Method for classifying the latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_model = EEGNetv4MTS(in_channels=4, num_classes=8, num_time_steps=500)
        >>> my_model.classify_latent_features(torch.rand(size=(10, 240))).size()
        torch.Size([10, 8])

        Running (1) feature extraction and (2) classifying is the excact same as just running forward

        >>> my_model = EEGNetv4MTS(in_channels=19, num_classes=8, num_time_steps=1500)
        >>> _ = my_model.eval()
        >>> my_input = torch.rand(size=(10, 19, 1500))
        >>> my_output_1 = my_model.classify_latent_features(my_model.extract_latent_features(my_input))
        >>> my_output_2 = my_model(my_input)
        >>> torch.equal(my_output_1, my_output_2)
        True
        """
        shape = (input_tensor.size()[0], self._model.F2, 1, self._model.final_conv_length)
        return self._model.final_layer(torch.reshape(input_tensor, shape=shape))

    def forward(self, x, return_features=False):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        return_features : bool

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 3000
        >>> my_model = EEGNetv4MTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps)), return_features=True).size()
        torch.Size([10, 1488])
        """
        # If predictions are to be made, just run forward method of the braindecode method
        if not return_features:
            return self._model(x)

        # If features prior to the classifier are to be returned instead, these features will be stored inside the
        # activation dict. Following a combination of https://www.youtube.com/watch?v=syLFCVYua6Q and
        # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
        activations_name = "latent_features"
        activation = dict()

        # noinspection PyUnusedLocal
        def hook(model, inputs):
            if len(inputs) != 1:
                raise ValueError(f"Expected only one input, but received {len(inputs)}")
            activation[activations_name] = inputs[0].detach()

        self._model.final_layer.register_forward_pre_hook(hook)

        # Run forward method, but we are interested the latent features
        _ = self._model(x)
        latent_features = activation[activations_name]

        # Fix the dimension. Currently, shape=(batch, F2, 1, final_conv_length)
        latent_features = torch.squeeze(latent_features, dim=2)  # Removing redundant dimension
        latent_features = torch.reshape(latent_features, shape=(latent_features.size()[0], -1))
        return latent_features

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self):
        # The latent features dimension is inferred from the dimension of their 'classifier_conv'
        return self._model.final_conv_length * self._model.F2


class EEGResNetMTS(MTSModuleBase):
    """
    EEGResNet

    todo: missing reference also in braindecode

    Examples
    --------
    >>> _ = EEGResNetMTS(in_channels=4, num_classes=8, num_time_steps=300)

    Latent features dimension

    >>> EEGResNetMTS(in_channels=4, num_classes=8, num_time_steps=300, n_first_filters=22).latent_features_dim
    66

    How the model looks like (the softmax/LogSoftmax activation function has been removed)

    >>> EEGResNetMTS(in_channels=4, num_classes=8, num_time_steps=3000)  # doctest: +NORMALIZE_WHITESPACE
    EEGResNetMTS(
      (_model): EEGResNet(
        (ensuredims): Ensure4d()
        (dimshuffle): Rearrange('batch C T 1 -> batch 1 T C')
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
        (final_layer): Sequential(
          (conv_classifier): Conv2d(18, 8, kernel_size=(1, 1), stride=(1, 1))
          (squeeze): Expression(expression=squeeze_final_output)
        )
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
        # Also, I don't think the 'n_times' is actually used?
        # ----------------
        self._model = EEGResNet(n_chans=in_channels, n_outputs=num_classes, n_times=num_time_steps,
                                final_pool_length=final_pool_length, n_first_filters=n_first_filters,
                                add_log_softmax=False, **kwargs)

    def extract_latent_features(self, input_tensor):
        """
        Method for extracting latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_model = EEGResNetMTS(in_channels=4, num_classes=8, num_time_steps=500, n_first_filters=22)
        >>> my_model.extract_latent_features(torch.rand(size=(10, 4, 500))).size()
        torch.Size([10, 66])
        """
        return self(input_tensor, return_features=True)

    def classify_latent_features(self, input_tensor):
        """
        Method for classifying the latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_model = EEGResNetMTS(in_channels=4, num_classes=8, num_time_steps=500, n_first_filters=22)
        >>> my_model.classify_latent_features(torch.rand(size=(10, 66))).size()
        torch.Size([10, 8])

        Running (1) feature extraction and (2) classifying is the excact same as just running forward

        >>> my_model = EEGResNetMTS(in_channels=19, num_classes=8, num_time_steps=1500, n_first_filters=22)
        >>> _ = my_model.eval()
        >>> my_input = torch.rand(size=(10, 19, 1500))
        >>> my_output_1 = my_model.classify_latent_features(my_model.extract_latent_features(my_input))
        >>> my_output_2 = my_model(my_input)
        >>> torch.equal(my_output_1, my_output_2)
        True
        """
        return self._model.final_layer(torch.unsqueeze(torch.unsqueeze(input_tensor, dim=-1), dim=-1))

    def forward(self, x, return_features=False):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        return_features : bool

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
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps)), return_features=True).size()
        torch.Size([10, 18])
        """
        # If predictions are to be made, just run forward method of the braindecode method
        if not return_features:
            return self._model(x)

        activations_name = "latent_features"
        activation = dict()

        # noinspection PyUnusedLocal
        def hook(model, inputs):
            if len(inputs) != 1:
                raise ValueError(f"Expected only one input, but received {len(inputs)}")
            activation[activations_name] = inputs[0].detach()

        self._model.final_layer.register_forward_pre_hook(hook)

        # Run forward method, but we are interested the latent features
        _ = self._model(x)
        latent_features = activation[activations_name]

        # Remove redundant dimensions. Currently, shape=(batch, features, 1, 1)
        return torch.squeeze(latent_features)

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self):
        # The latent features dimension is inferred from the dimension of their 'classifier_conv'
        return self._model.final_layer.conv_classifier.in_channels


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
        (dimshuffle): Rearrange('batch C T 1 -> batch 1 T C')
        (conv_time_spat): CombinedConv(
          (conv_time): Conv2d(1, 40, kernel_size=(25, 1), stride=(1, 1))
          (conv_spat): Conv2d(40, 40, kernel_size=(1, 4), stride=(1, 1), bias=False)
        )
        (bnorm): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_nonlin_exp): Expression(expression=square)
        (pool): AvgPool2d(kernel_size=(75, 1), stride=(15, 1), padding=0)
        (pool_nonlin_exp): Expression(expression=safe_log)
        (drop): Dropout(p=0.5, inplace=False)
        (final_layer): Sequential(
          (conv_classifier): Conv2d(40, 7, kernel_size=(7, 1), stride=(1, 1))
          (squeeze): Expression(expression=squeeze_final_output)
        )
      )
    )
    """

    def __init__(self, in_channels, num_classes, num_time_steps, **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = ShallowFBCSPNet(n_chans=in_channels, n_outputs=num_classes, n_times=num_time_steps,
                                      final_conv_length="auto", add_log_softmax=False, **kwargs)

    def extract_latent_features(self, input_tensor):
        return self(input_tensor, return_features=True)

    def classify_latent_features(self, input_tensor):
        """
        Method for classifying the latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_batch, my_channels, my_time_steps = 10, 103, 600*3
        >>> my_model = ShallowFBCSPNetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> my_model.classify_latent_features(torch.rand(size=(10, 4560))).size()
        torch.Size([10, 3])

        Running (1) feature extraction and (2) classifying is the excact same as just running forward

        >>> my_model = ShallowFBCSPNetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> _ = my_model.eval()
        >>> my_input = torch.rand(size=(my_batch, my_channels, my_time_steps))
        >>> my_output_1 = my_model.classify_latent_features(my_model.extract_latent_features(my_input))
        >>> my_output_2 = my_model(my_input)
        >>> torch.equal(my_output_1, my_output_2)
        True
        """
        shape = (input_tensor.size()[0], self._model.final_layer.conv_classifier.in_channels,
                 self._model.final_conv_length, 1)
        return self._model.final_layer(torch.reshape(input_tensor, shape=shape))

    def forward(self, x, return_features=False):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        return_features : bool

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 600*3
        >>> my_model = ShallowFBCSPNetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps)), return_features=True).size()
        torch.Size([10, 4560])
        """
        # If predictions are to be made, just run forward method of the braindecode method
        if not return_features:
            return self._model(x)

        activations_name = "latent_features"
        activation = dict()

        # noinspection PyUnusedLocal
        def hook(model, inputs):
            if len(inputs) != 1:
                raise ValueError(f"Expected only one input, but received {len(inputs)}")
            activation[activations_name] = inputs[0].detach()

        self._model.final_layer.register_forward_pre_hook(hook)

        # Run forward method, but we are interested the latent features
        _ = self._model(x)
        latent_features = activation[activations_name]

        # Fix dimensions
        latent_features = torch.squeeze(latent_features, dim=-1)  # Removing redundant dimension
        latent_features = torch.reshape(latent_features, shape=(latent_features.size()[0], -1))
        return latent_features

    # ----------------
    # Hyperparameter sampling
    # ----------------
    @staticmethod
    def sample_hyperparameters(config):
        """
        Method for sampling hyperparameters

        Parameters
        ----------
        config : dict[str, typing.Any]

        Returns
        -------
        dict[str, typing.Any]

        Examples
        --------
        >>> my_num_filters = {"dist": "uniform_int", "kwargs": {"a": 5, "b": 8}}
        >>> my_filter_lengths = {"dist": "uniform_int", "kwargs": {"a": 9, "b": 14}}
        >>> my_pool_time_stride = {"dist": "uniform_int", "kwargs": {"a": 10, "b": 20}}
        >>> my_drop_out = {"dist": "uniform", "kwargs": {"a": 0.0, "b": 0.5}}
        >>> import numpy
        >>> numpy.random.seed(3)
        >>> ShallowFBCSPNetMTS.sample_hyperparameters(
        ...     {"n_filters": my_num_filters, "filter_time_length": my_filter_lengths,
        ...      "pool_time_stride": my_pool_time_stride, "drop_prob": my_drop_out}
        ... )  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        {'n_filters_time': 7, 'n_filters_spat': 7, 'filter_time_length': 9, 'pool_time_stride': 19,
         'pool_time_length': 95, 'drop_prob': 0.419...}
        """
        # Sample number of filters. Will be the same for temporal and spatial
        n_filters = sample_hyperparameter(config["n_filters"]["dist"], **config["n_filters"]["kwargs"])

        # Sample length of temporal filter
        filter_time_length = sample_hyperparameter(config["filter_time_length"]["dist"],
                                                   **config["filter_time_length"]["kwargs"])

        # Sample length of temporal filter
        pool_time_stride = sample_hyperparameter(config["pool_time_stride"]["dist"],
                                                 **config["pool_time_stride"]["kwargs"])

        # We set the ratio of length/stride to the same as in the original paper
        pool_time_length = 5 * pool_time_stride

        # Sample drop out
        drop_prob = sample_hyperparameter(config["drop_prob"]["dist"], **config["drop_prob"]["kwargs"])

        return {"n_filters_time":  n_filters,
                "n_filters_spat": n_filters,
                "filter_time_length": filter_time_length,
                "pool_time_stride": pool_time_stride,
                "pool_time_length": pool_time_length,
                "drop_prob": drop_prob}


class Deep4NetMTS(MTSModuleBase):
    """
    Deep4Net

    This was used in Engemann et al. (2022).

    Examples
    --------
    >>> _ = Deep4NetMTS(19, 3, 1000)

    The number of input time samples must be greater than 440

    >>> _ = Deep4NetMTS(19, 3, 440)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ValueError: During model prediction RuntimeError was thrown showing that at some layer ` Output size is too small`
    (see above in the stacktrace). This could be caused by providing too small `n_times`/`input_window_seconds`. Model
    may require longer chunks of signal in the input than (1, 19, 440).


    How the model looks like (the softmax/LogSoftmax activation function has been removed)

    >>> Deep4NetMTS(19, 3, 1800)  # doctest: +NORMALIZE_WHITESPACE
    Deep4NetMTS(
      (_model): Deep4Net(
        (ensuredims): Ensure4d()
        (dimshuffle): Rearrange('batch C T 1 -> batch 1 T C')
        (conv_time_spat): CombinedConv(
          (conv_time): Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
          (conv_spat): Conv2d(25, 25, kernel_size=(1, 19), stride=(1, 1), bias=False)
        )
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
        (final_layer): Sequential(
          (conv_classifier): Conv2d(200, 3, kernel_size=(17, 1), stride=(1, 1))
          (squeeze): Expression(expression=squeeze_final_output)
        )
      )
    )

    Does not work with 'split_last_layer' set to False

    >>> Deep4NetMTS(19, 3, 1000, split_first_layer=False)
    Traceback (most recent call last):
    ...
    AttributeError: 'Deep4Net' object has no attribute 'conv_time_spat'
    """

    def __init__(self, in_channels, num_classes, num_time_steps, final_conv_length="auto", **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = Deep4Net(n_chans=in_channels, n_outputs=num_classes, n_times=num_time_steps,
                               final_conv_length=final_conv_length, add_log_softmax=False, **kwargs)

    def extract_latent_features(self, input_tensor):
        """
        Method for extracting latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_model = Deep4NetMTS(in_channels=4, num_classes=3, num_time_steps=1800,
        ...                        n_filters_4=206)
        >>> my_model.extract_latent_features(torch.rand(size=(10, 4, 1800))).size()
        torch.Size([10, 3502])
        """
        return self(input_tensor, return_features=True)

    def classify_latent_features(self, input_tensor):
        """
        Method for classifying the latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_model = Deep4NetMTS(in_channels=4, num_classes=8, num_time_steps=1800, n_filters_4=206)
        >>> my_model.classify_latent_features(torch.rand(size=(10, 3502))).size()
        torch.Size([10, 8])

        Running (1) feature extraction and (2) classifying is the excact same as just running forward

        >>> my_model = Deep4NetMTS(in_channels=19, num_classes=8, num_time_steps=1500)
        >>> _ = my_model.eval()
        >>> my_input = torch.rand(size=(10, 19, 1500))
        >>> my_output_1 = my_model.classify_latent_features(my_model.extract_latent_features(my_input))
        >>> my_output_2 = my_model(my_input)
        >>> torch.equal(my_output_1, my_output_2)
        True
        """
        shape = (input_tensor.size()[0], self._model.n_filters_4, self._model.final_conv_length, 1)
        return self._model.final_layer(torch.reshape(input_tensor, shape=shape))

    def forward(self, x, return_features=False):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        return_features : bool

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 1800
        >>> my_model = Deep4NetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps,
        ...                        n_filters_4=206)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps)), return_features=True).size()
        torch.Size([10, 3502])
        """
        # If predictions are to be made, just run forward method of the braindecode method
        if not return_features:
            return self._model(x)

        activations_name = "latent_features"
        activation = dict()

        # noinspection PyUnusedLocal
        def hook(model, inputs):
            if len(inputs) != 1:
                raise ValueError(f"Expected only one input, but received {len(inputs)}")
            activation[activations_name] = inputs[0].detach()

        self._model.final_layer.register_forward_pre_hook(hook)

        # Run forward method, but we are interested the latent features
        _ = self._model(x)
        latent_features = activation[activations_name]

        # Fix dimensions. Currently, shape=(batch, n_filters_4, final_conv_length, 1)
        latent_features = torch.squeeze(latent_features, dim=-1)  # Removing redundant dimension
        latent_features = torch.reshape(latent_features, shape=(latent_features.size()[0], -1))
        return latent_features

    # ----------------
    # Hyperparameter sampling
    # ----------------
    @staticmethod
    def sample_hyperparameters(config):
        """
        The ratio between the number of filters will be maintained as in the original work

        Parameters
        ----------
        config : dict[str, typing.Any]

        Returns
        -------
        dict[str, typing.Any]

        Examples
        --------
        >>> my_num_filters = {"dist": "uniform_int", "kwargs": {"a": 5, "b": 8}}
        >>> my_filter_lengths = {"dist": "uniform_int", "kwargs": {"a": 9, "b": 14}}
        >>> my_drop_out = {"dist": "uniform", "kwargs": {"a": 0.0, "b": 0.5}}
        >>> import numpy
        >>> numpy.random.seed(3)
        >>> Deep4NetMTS.sample_hyperparameters({"n_first_filters": my_num_filters, "filter_length": my_filter_lengths,
        ...                                     "drop_prob": my_drop_out})  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        {'n_filters_time': 7, 'n_filters_spat': 7, 'n_filters_2': 14, 'n_filters_3': 28, 'n_filters_4': 56,
         'filter_time_length': 9, 'filter_length_2': 9, 'filter_length_3': 9, 'filter_length_4': 9,
         'drop_prob': 0.3540...}

        """
        # Get the number of filters for the first conv block
        num_first_filters = sample_hyperparameter(config["n_first_filters"]["dist"],
                                                  **config["n_first_filters"]["kwargs"])

        num_filters_hyperparameters = {"n_filters_time": num_first_filters,
                                       "n_filters_spat": num_first_filters,
                                       "n_filters_2": 2 * num_first_filters,
                                       "n_filters_3": 4 * num_first_filters,
                                       "n_filters_4": 8 * num_first_filters}

        # Get the filter lengths
        filter_length = sample_hyperparameter(config["filter_length"]["dist"], **config["filter_length"]["kwargs"])

        # Compute the length of the filters for the conv blocks
        filter_lengths_hyperparameters = {"filter_time_length": filter_length,
                                          "filter_length_2": filter_length,
                                          "filter_length_3": filter_length,
                                          "filter_length_4": filter_length}
        # Get the drop out
        drop_prob = sample_hyperparameter(config["drop_prob"]["dist"], **config["drop_prob"]["kwargs"])

        return {**num_filters_hyperparameters, **filter_lengths_hyperparameters, **{"drop_prob": drop_prob}}

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self):
        # The latent features dimension is inferred from the dimension of their 'classifier_conv'
        return self._model.n_filters_4 * self._model.final_conv_length
