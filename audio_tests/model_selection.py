"""
This module contains all the functions that are necessary to select the models that will be used to detect the presence
of a device with microphone streaming capabilities.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import pickle
import pprint

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

    :param Dataset x: Pre-processed samples from the dataset.
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


if __name__ == "__main__":
       from cleaner import DATASET_NUMBER
       
       # Importing the data from the dataset file
       x = pd.read_csv(f"datasets/full_dataset_test-{DATASET_NUMBER}.csv")
       y = x.loc[:, "IsMicrophone"]
       x = x.loc[:, list(set(x.columns) - set(["IsMicrophone", "SourceAddress", "Unnamed: 0"]))]      
       
       # Importing the selection parameters
       selection_parameters = {
              'cv': 5,
              'n_jobs': -1,
              'verbose': True
       }
       
       # Generating the candidate models' list
       model_candidates = [
              DecisionTreeClassifier(),
              ExtraTreeClassifier(),
              RandomForestClassifier(),
              ExtraTreesClassifier(),
              KNeighborsClassifier(),
              SVC()              
       ]
       
       # Generating the hyperparameters for all the models
       hyperparameters_candidates = [
              { # DecisionTreeClassifier
                     'criterion': ['gini', 'entropy'],
                     'splitter': ['best', 'random']
              },
              { # ExtraTreeClassifier
                     'criterion': ['gini', 'entropy'],
                     'splitter': ['best', 'random']
              },
              { # RandomForestClassifier
                     'n_estimators': [20, 50, 100, 250],
                     'criterion': ['gini', 'entropy'],
                     'bootstrap': [True, False],
              },
              { # ExtraTreesClassifier
                     'n_estimators': [20, 50, 100, 250],
                     'criterion': ['gini', 'entropy'],
                     'bootstrap': [True, False],
              },
              { # KNeighborsClassifier
                     'n_neighbors': [2, 3, 5, 7, 10],
                     'weights': ['uniform', 'distance'],
                     'p': [1,2,3]
              },
              { # SVC
                     'C': [1e-3, 1e-2, 1e-1, .5, 1., 5.],
                     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                     'degree': [2, 3],
              }
       ]
       
       best_candidates, scores = model_selection(x=x, 
                                                 y=y, 
                                                 candidates=model_candidates, 
                                                 parameter_grids=hyperparameters_candidates, 
                                                 selection_params=selection_parameters)
       
       with open("training_results.txt", 'w') as f:
              for model, score in zip(best_candidates, scores):
                     f.write(model.__class__.__name__)
                     f.write(pprint.pprint(model.get_params(), stream=f))
                     f.write(score)
                     f.write(5 * '=')
                     with open("model_dump.obj", 'w') as model_file_dump:
                            pickle.dump(model, model_file_dump)
              
       