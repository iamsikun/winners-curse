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

    def generate_training_data(self, sample_size: int) -> tuple:
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
        # treatment effect function: given a covariate x, return the treatment effect of the group
        te_func = lambda x: self.te_list[self.group_func(x)]
        
        # generate data
        X = self.__generate_individual_characteristics(sample_size, seed)  # (sample_size, )
        T = np.random.binomial(1, 0.5, sample_size)  # (sample_size, )
        te_arr = np.array([te_func(x) for x in X])  # (sample_size, )
        Y = te_arr * T  + np.random.normal(0, self.noise_std, sample_size)  # (sample_size, )
        
        return X.reshape(-1, 1), T.reshape(-1, 1), Y.reshape(-1, 1)  # (sample_size, 1)
    
    def generate_testing_data(self, sample_size: int) -> np.ndarray:
        X = self.__generate_individual_characteristics(sample_size) # (sample_size, )

        return X  # (sample_size, )

    def __generate_individual_characteristics(self, sample_size: int) -> np.ndarray:
        return np.random.uniform(-3, 3, sample_size)
    

class PersonalizedPricingDGP(object):
    def __init__(self, cov_dim: int = 1):
        self.cov_dim = cov_dim
        self.util_const_map = np.random.uniform(0, 1, size=(cov_dim, 1))  # (cov_dim, 1)
        self.util_price_map = np.random.uniform(-0.1, 0, size=(cov_dim, 1))  # (cov_dim, 1)

    def generate_training_data(
        self, sample_size: int, price_lb: float, price_ub: float, price_diff: float, seed=None
    ) -> tuple:
        """
        Generate training data

        Params:
        -------
        sample_size: int, number of samples to generate
        price_lb: float, lower bound of the price
        price_ub: float, upper bound of the price
        price_diff: float, price difference
        seed: int, random seed

        Returns:
        -------
        tuple: 
            - X: np.ndarray, covariate
            - T: np.ndarray, treatment assignment
            - Y: np.ndarray, outcome
        """
        # generate data
        # individual characteristics
        X_arr = self.__generate_individual_characteristics(sample_size)  # (sample_size, cov_dim)
        
        # price (treaments)
        price_arr = np.random.choice(np.arange(price_lb, price_ub, price_diff), sample_size).reshape(-1, 1)  # (sample_size, 1)

        # error
        err_arr = np.random.gumbel(loc=0, scale=1, size=(sample_size, 1))  # (sample_size, 1)

        # utility: (sample_size, 1)
        util_arr = X_arr @ self.util_const_map + X_arr @ self.util_price_map * price_arr + err_arr

        # demand: (sample_size, 1), buy if utility > 0, else don't buy.
        demand_arr = (util_arr > 0).astype(int)

        return X_arr, price_arr, demand_arr
    
    def generate_testing_data(self, sample_size: int) -> np.ndarray:        
        return self.__generate_individual_characteristics(sample_size)  # (sample_size, )

    def __generate_individual_characteristics(self, sample_size: int) -> np.ndarray:
        return np.random.normal(loc=3, scale=0.5, size=(sample_size, self.cov_dim))