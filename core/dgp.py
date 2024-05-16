import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np

class DGP1:
    """ 
    Data Generating Process 1:
    \tau_i \in \{a_1, \dots, a_K\}  # K groups, each with a different treatment effect a_k 
    Y_i = \tau_i T_i + \epsilon_i, \epsilon_i \sim N(0, \sigma^2)  # outcome model
    T_i \sim Bernoulli(p)  # treatment assignment model
    """
    def __init__(
        self, num_groups: int, te_diff: float, group_func: callable, 
        noise_std: float = 1.0, treatment_assign_prob: float = 0.5, **kwargs
    ):
        """  
        DGP for model 1

        Params:
        -------
        num_groups: int, number of groups
        te_diff: float, treatment effect difference
        noise_std: float, standard deviation of the noise in the outcome model
        treatment_assign_prob: float, probability of treatment assignment
        """
        # attributes
        self.n_groups = num_groups
        self.te_diff = te_diff
        self.te_list = [1 + i * te_diff for i in range(num_groups)]
        self.group_func = group_func
        self.noise_std = noise_std
        self.treatment_assign_prob = treatment_assign_prob

    def generate_training_data(self, sample_size: int, seed=None) -> tuple:
        """
        Generate training data

        Params:
        -------
        sample_size: int, number of samples to generate

        Returns:
        -------
        tuple: 
            - X: np.ndarray, covariate
            - T: np.ndarray, treatment assignment
            - Y: np.ndarray, outcome
        """
        # set seed
        seed = seed if seed else np.random.randint(0, 1e6)
        np.random.seed(seed)

        # treatment effect function: given a covariate x, return the treatment effect of the group
        te_func = lambda x: self.te_list[self.group_func(x)]
        
        # generate data
        X = self.__generate_individual_characteristics(sample_size, seed)  # (sample_size, )
        T = np.random.binomial(1, 0.5, sample_size)  # (sample_size, )
        te_arr = np.array([te_func(x) for x in X])  # (sample_size, )
        Y = te_arr * T  + np.random.normal(0, self.noise_std, sample_size)  # (sample_size, )
        
        return X.reshape(-1, 1), T.reshape(-1, 1), Y.reshape(-1, 1)  # (sample_size, 1)
    
    def generate_testing_data(self, sample_size: int, seed=None) -> np.ndarray:
        seed = seed if seed else np.random.randint(0, 1e6)
        
        # generate data
        X = self.__generate_individual_characteristics(sample_size, seed) # (sample_size, )

        return X  # (sample_size, )

    def __generate_individual_characteristics(self, sample_size: int, seed: int) -> np.ndarray:
        np.random.seed(seed)
        return np.random.uniform(-3, 3, sample_size)