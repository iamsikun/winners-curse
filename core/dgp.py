import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd

class DataGenerationProcess(object):
    def sample(self) -> pd.DataFrame:
        raise NotImplementedError

    def sample_individuals(self) -> np.ndarray:
        raise NotImplementedError
    

class SingleSegment(DataGenerationProcess):
    def __init__(
        self, te: float, treatment_space: np.ndarray, 
        noise_std: float, 
    ):
        self.te = te
        self.treatment_space = treatment_space
        self.noise_std = noise_std

    def sample(self, sample_size: int, seed: int = None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)
        
        # random treatment assignment
        treatments = np.random.choice(self.treatment_space, size=sample_size)  # shape = (sample_size, )

        # calculate outcomes
        noises = np.random.normal(loc=0, scale=self.noise_std, size=(sample_size, ))  # shape = (sample_size, )
        outcomes = self.te * treatments + noises # shape = (sample_size, )

        return pd.DataFrame({
            'outcome': outcomes, 'treatment': treatments
        })
    
    def sample_individuals(self, sample_size: int) -> np.ndarray:
        """ 
        Because there is only one segment, return a vector of zeros
        """
        return self.zeros(shape=(sample_size, ))  # shape = (sample_size, )
    

class SegmentTargetingDGP(object):
    """ 
    Data Generating Process for segment targeting
    """
    def __init__(
        self, n_segments: int, te_diff: float, segment_func: callable, 
        noise_std: float = 1.0, treatment_assign_prob: float = 0.5, **kwargs
    ):
        """  
        DGP for model 1

        Params:
        -------
        num_segments: int, number of segments
        te_diff: float, treatment effect difference
        noise_std: float, standard deviation of the noise in the outcome model
        treatment_assign_prob: float, probability of treatment assignment
        """
        # attributes
        self.n_segments = n_segments
        self.te_diff = te_diff
        self.segment_te_arr = np.array([1 + i * te_diff for i in range(n_segments)])
        self.segment_func = segment_func
        self.noise_std = noise_std
        self.treatment_assign_prob = treatment_assign_prob

    def generate_training_data(self, sample_size: int, outcome_noise: np.ndarray = None, seed: int = None) -> tuple:
        """
        Generate training data

        Params:
        -------
        sample_size: int, number of samples to generate
        outcome_noise: np.ndarray, noise in the outcome model
        seed: int, random seed

        Returns:
        -------
        tuple: 
            - X: np.ndarray, covariate
            - T: np.ndarray, treatment assignment
            - Y: np.ndarray, outcome
        """
        if seed is not None:
            np.random.seed(seed)

        # treatment effect function: given a covariate x, return the treatment effect of the group
        te_func = lambda x: self.segment_te_arr[self.segment_func(x)]
        
        # generate data
        X = self.sample_individuals(sample_size, seed=seed)  # (sample_size, )
        T = np.random.binomial(1, 0.5, sample_size)  # (sample_size, )
        customer_te_arr = np.array([te_func(x) for x in X])  # (sample_size, )

        noise_arr = outcome_noise if outcome_noise is not None else self.sample_outcome_noise(sample_size, seed=seed)

        Y = customer_te_arr * T  + noise_arr  # (sample_size, )
        
        return X.reshape(-1, 1), T.reshape(-1, 1), Y.reshape(-1, 1)
    
    def sample_outcome_noise(self, sample_size: int, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(0, self.noise_std, sample_size)
    
    def sample_individuals(self, sample_size: int, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-3, 3, sample_size)