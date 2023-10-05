from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_ml_model(model, **kwargs):
    """
    Function for getting the specified (non-deep learning) machine learning model

    Parameters
    ----------
    model : str
    kwargs

    Returns
    -------
    typing.Any
        A machine learning model
    """
    # All available ML models must be added here
    available_models = (RandomForestClassifier, DecisionTreeClassifier, SVC, LinearDiscriminantAnalysis,
                        QuadraticDiscriminantAnalysis, AdaBoostClassifier)

    # Loop through and select the correct one
    for ml_model in available_models:
        if model == ml_model.__name__:
            return ml_model(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The ML model module '{model}' was not recognised. Please select among the following: "
                     f"{tuple(ml_model.__name__ for ml_model in available_models)}")
