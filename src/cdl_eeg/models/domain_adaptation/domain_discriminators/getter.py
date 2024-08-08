from typing import Type

from cdl_eeg.models.domain_adaptation.domain_discriminators.domain_discriminator_base import DomainDiscriminatorBase
from cdl_eeg.models.domain_adaptation.domain_discriminators.fc_modules import FCModule


def get_domain_discriminator(name, **kwargs):
    """
    Function for getting the specified domain discriminator

    Parameters
    ----------
    name : str
        Class name of the domain discriminator
    kwargs
        Key word arguments, which depends on the selected domain discriminator module

    Returns
    -------
    cdl_eeg.models.domain_discriminators.domain_discriminator_base.DomainDiscriminatorBase

    Examples
    --------
    >>> _ = get_domain_discriminator("FCModule", in_features=55, num_classes=3)
    >>> get_domain_discriminator("NotADomainDiscriminator")  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The domain discriminator module 'NotADomainDiscriminator' was not recognised. Please select among the
    following: ('FCModule',...)
    """
    # All available domain discriminators must be included here
    available_domain_discriminators = (FCModule,)

    # Loop through and select the correct one
    for domain_discriminator in available_domain_discriminators:
        if name == domain_discriminator.__name__:
            return domain_discriminator(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The domain discriminator module '{name}' was not recognised. Please select among the following: "
                     f"{tuple(discriminator.__name__ for discriminator in available_domain_discriminators)}")


def get_domain_discriminator_type(name) -> Type[DomainDiscriminatorBase]:
    """
    Function for getting the specified domain discriminator class

    Parameters
    ----------
    name : str
        Class name of the domain discriminator

    Examples
    --------
    >>> _ = get_domain_discriminator_type("FCModule")
    >>> get_domain_discriminator_type("NotADomainDiscriminator")  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The domain discriminator module 'NotADomainDiscriminator' was not recognised. Please select among the
    following: ('FCModule',...)
    """
    # All available domain discriminators must be included here
    available_domain_discriminators = (FCModule,)

    # Loop through and select the correct one
    for domain_discriminator in available_domain_discriminators:
        if name == domain_discriminator.__name__:
            return domain_discriminator

    # If no match, an error is raised
    raise ValueError(f"The domain discriminator module '{name}' was not recognised. Please select among the following: "
                     f"{tuple(discriminator.__name__ for discriminator in available_domain_discriminators)}")
