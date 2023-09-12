import abc

import torch.nn as nn


class MTSClassifierBase(nn.Module, abc.ABC):
    # todo: consider overriding __new__ to store all inputs and defaults of __init__
    ...
