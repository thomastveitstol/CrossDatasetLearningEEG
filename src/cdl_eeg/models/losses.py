import torch.nn as nn


def get_loss_function(loss):
    # All available loss functions must be included here
    available_losses = (nn.MSELoss(reduction="mean"), nn.L1Loss(reduction="mean"), nn.BCELoss(reduction="mean"),
                        nn.BCEWithLogitsLoss(reduction="mean"))

    # Loop through and select the correct one
    for available_loss in available_losses:
        if loss == type(available_loss).__name__:
            return available_loss(reduction="mean")

    # If no match, an error is raised
    raise ValueError(f"The loss function '{loss}' was not recognised. Please select among the following: "
                     f"{tuple(type(available_loss).__name__ for available_loss in available_losses)}")
