from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import openml

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def get_model_parameters(model):
    """Returns the parameters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_]
    return params

def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    model.classes_ = np.array([0,1])
    return model

def set_initial_params(model: LogisticRegression):
    """
    Sets initial parameters as zeros
    """
    n_classes = 1 # MNIST has 10 classes
    n_features = 2448 # Number of features in dataset
    model.classes_ = np.array([0,1])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def get_initial_params():
    """
    Sets initial parameters as zeros
    """
    n_classes = 1 # MNIST has 10 classes
    n_features = 2448 # Number of features in dataset

    return [np.zeros((n_classes, n_features)),np.zeros((n_classes,))]
    # model.coef_ = np.zeros((n_classes, n_features))
    # if model.fit_intercept:
    #     model.intercept_ = np.zeros((n_classes,))

# def load_mnist() -> Dataset:
#     """
#     Loads the MNIST dataset using OpenML
#     Dataset link: https://www.openml.org/d/554
#     """
#     mnist_openml = openml.datasets.get_dataset(554)
#     Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
#     X = Xy[:, :-1] # the last column contains labels
#     y = Xy[:, -1]
#     # First 60000 samples consist of the train set
#     x_train, y_train = X[:60000], y[:60000]
#     x_test, y_test = X[60000:], y[60000:]
#     return (x_train, y_train), (x_test, y_test)

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions),
        np.array_split(y, num_partitions))
    )
