import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np


class BaseEstimator(object):
    def check_input_decorator(func: callable) -> callable:
        """ 
        Decorator to check the input shapes of the data
        """
        def wrapper(self, covariates: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray, **kwargs) -> None:
            assert covariates.shape[0] == treatments.shape[0] == outcomes.shape[0], "Input shapes do not match"
            assert treatments.shape[1] == 1, "T should be a column vector"
            assert outcomes.shape[1] == 1, "Y should be a column vector"
            return func(
                self, 
                covariates=covariates, 
                treatments=treatments, 
                outcomes=outcomes
            )
        return wrapper