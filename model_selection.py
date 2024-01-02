"""
This module contains all the functions that are necessary to select the models that will be used to detect the presence
of a device with microphone streaming capabilities.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV

Dataset = np.ndarray | pd.DataFrame
CandidateModels = List[ClassifierMixin]
ParameterGrids = List[Dict]


def best_model_selection(classifier: ClassifierMixin, x_train: Dataset, y_train: Dataset, hyperparams: dict,
                         selection_params: dict) -> ClassifierMixin:
    """
    Given the parameters, this function returns the trained model.
    The model is selected by using the k-folded cross validation algorithm.

    :param dict selection_params: Dictionary of the parameters for the model selection algorithm.
           An instance of scikit-learn GridSearchCV is used.
    :param ParameterGrids hyperparams: Dictionary of the candidates for the models' hyperparameters.
    :param Dataset y_train: Set of true labels for @x_train.
    :param Dataset x_train: Set of unlabeled samples of the dataset that is used to train the models.
    :param ClassifierMixin classifier: Instance of the classifier object whose best hyperparameter combination will be found.
    :return: Optimized model for @classifier.
    """

    gridsearch = GridSearchCV(estimator=classifier,
                              param_grid=hyperparams,
                              n_jobs=selection_params['n_jobs'],
                              cv=selection_params['cv'],
                              verbose=selection_params['verbose'])

    gridsearch.fit(X=x_train, y=y_train)
    return gridsearch.best_estimator_


def model_selection(x: Dataset, y: Dataset, candidates: CandidateModels, parameter_grids: ParameterGrids,
                    selection_params: dict, test_split: float = 0.25) -> Tuple[List, List]:
    """

    :param Dataset x: Pre-processed sampled from the dataset.
    :param Dataset y: True labels for @x
    :param CandidateModels candidates:  List of candidate model instances of which the optimized version will be extracted.
    :param ParameterGrids parameter_grids: List of parameter grids on which all the candidate models will be selected.
    :param dict selection_params: Dictionary of the parameters for the model selection algorithm.
           An instance of scikit-learn GridSearchCV is used.
    :param float test_split: Portion of the dataset to reserve for testing purposes.
    :return: A (list, list) pair that contains respectively the best models for each candidate and their scores.
    """

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, shuffle=True)
    best_candidates, scores = [], []
    for model, hyperparameters in zip(candidates, parameter_grids):
        best_model = best_model_selection(classifier=model,
                                          x_train=x_train,
                                          y_train=y_train,
                                          hyperparams=hyperparameters,
                                          selection_params=selection_params)
        best_candidates.append(best_model)
        scores.append(best_model.score(X=x_test, y=y_test))

    return best_candidates, scores
