import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd

from scipy.special import expit

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
        assert len(treatment_space) > 1, "Number of treatments should be greater than 1."
        assert 0 in treatment_space, "Control group should be included in the treatment space."

        self.te = te
        self.treatment_space = np.sort(treatment_space)
        self.noise_std = noise_std

    def sample(self, sample_size: int, seed: int = None) -> tuple:
        """ 
        Sample data from the DGP

        Params:
        -------
        sample_size: int
            Number of samples to generate
        seed: int
            Random seed

        Returns:
        --------
        tuple:
            - treatment: np.ndarray, shape = (sample_size, )
            - outcome: np.ndarray, shape = (sample_size, )
        """
        if seed is not None:
            np.random.seed(seed)
        
        # random treatment assignment
        treatments = np.random.choice(self.treatment_space, size=sample_size)  # shape = (sample_size, )

        # calculate outcomes
        noises = np.random.normal(loc=0, scale=self.noise_std, size=(sample_size, ))  # shape = (sample_size, )
        outcomes = self.te * treatments + noises # shape = (sample_size, )

        return treatments, outcomes 
    
    def sample_individuals(self, sample_size: int) -> np.ndarray:
        return self.te * self.ones(shape=(sample_size, ))  # shape = (sample_size, )
    
    @property
    def lift_arr(self) -> np.ndarray:
        """ 
        Calculate the lift for each treatment value except control group

        Returns:
        --------
        np.ndarray: shape = (n_treatments, )
        """
        return np.array([self.te * treatment_val for treatment_val in self.treatment_space])


class MultipleSegments(DataGenerationProcess):
    def __init__(
        self, te: float, n_segments: int, 
        treatment_space: np.ndarray, noise_std: float = 1.0, 
    ):
        self.segment_arr = np.arange(n_segments)
        self.te = te
        self.segment_te_arr = np.array([(i + 1) * self.te for i in range(n_segments)])
        self.noise_std = noise_std
        self.treatment_space = treatment_space

        # potential outcome and treatment effects
        # * We assume control group in all segments have expected outcome of zero, so 
        # * lift in each segment is equal to the expected outcome in that segment.
        self.exp_outcome_table = pd.DataFrame.from_records([
            {'segment': segment, 'treatment': treatment, 'outcome': treatment * self.segment_te_arr[segment]}
            for segment in self.segment_arr for treatment in self.treatment_space
        ]).pivot(index='segment', columns='treatment', values='outcome')
        self.lift_table = self.exp_outcome_table.apply(lambda x: x - x[0], axis=1)
    
    def sample_individuals(self, sample_size, seed: int = None) -> np.ndarray:
        """ 
        Sample individuals from the DGP. 

        Params:
        -------
        sample_size: int
            Number of individuals to sample.
        seed: int
            Random seed for reproducibility.

        Returns:
        --------
        np.ndarray, shape = (sample_size, )
            Array of sampled individuals' segments. 
        """
        if seed is not None:
            np.random.seed(seed)

        return np.random.choice(self.segment_arr, size=sample_size)  
    
    def sample(self, sample_size, seed: int = None) -> tuple:
        """ 
        Sample data from the DGP. 

        Params:
        -------
        sample_size: int
            Number of individuals to sample.
        seed: int
            Random seed for reproducibility.

        Returns:
        --------
        (segments, treatments, outcomes)
            - segments: np.ndarray, shape = (sample_size, )
                Array of sampled individuals' segments.
            - treatments: np.ndarray, shape = (sample_size, )
                Array of sampled individuals' treatments.
            - outcomes: np.ndarray, shape = (sample_size, )
                Array of sampled individuals' outcomes.
        """
        if seed is not None:
            np.random.seed(seed)

        # sample customer segments
        segment_arr = self.sample_individuals(sample_size)  # shape = (sample_size, )

        # assign treatment effect to each customers based on their segment
        customer_te_arr = self.segment_te_arr[segment_arr]  # shape = (sample_size, )

        # randomly assign treatment to each customer
        treatment_arr = np.random.choice(self.treatment_space, size=sample_size)  # shape = (sample_size, )

        # compute outcomes
        outcome_arr = treatment_arr * customer_te_arr + np.random.normal(loc=0, scale=self.noise_std, size=sample_size)

        return segment_arr, treatment_arr, outcome_arr


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
    

class Pricing(DataGenerationProcess):
    def __init__(self, n_covariates: int = 133, seed: int = None):
        super().__init__()
        
        # attributes
        self.n_covariates = n_covariates

        # Set seed
        if seed is not None:
            np.random.seed(seed)

        # Generate true beta coefficients
        self.theta_arr = np.random.uniform(size=(self.n_covariates, 2))  # shape = (n_covariates, 2)
        self.theta_arr[:, 0] = self.theta_arr[:, 0] / 2.5  # Scaling of first column
        self.theta_arr[:, 1] = -np.abs(self.theta_arr[:, 1]) * 2.5  # Negative scaling for price sensitivity

        # Randomly select which features have non-zero weights
        self.theta_arr[np.random.choice(np.arange(self.n_covariates), size=self.n_covariates - 30, replace=False), 0] = 0
        self.theta_arr[np.random.choice(np.arange(self.n_covariates), size=self.n_covariates - 30, replace=False), 1] = 0

        # Generate covariance matrix for features (Cholesky decomposition)
        self.chol_cov = np.diag(np.ones(self.n_covariates)) + np.tril(np.random.uniform(size=(self.n_covariates, self.n_covariates)))

    def sample(self, sample_size: int, seed: int = None) -> pd.DataFrame:
        # set seed
        if seed is not None:
            np.random.seed(seed)

        # Generate covariates
        X = self.sample_individuals(sample_size, seed)  # shape = (sample_size, n_covariates)

        # Compute true intercept (alpha_arr) and price sensitivity (beta_arr)
        alpha_arr = X @ self.theta_arr[:, 0]  # shape = (sample_size,)
        beta_arr = X @ self.theta_arr[:, 1]  # shape = (sample_size,)

        # Generate prices
        price_arr = np.random.uniform(size=sample_size) / 2  # shape = (sample_size,)

        # Calculate the probability of choice
        choice_prob_arr = expit(beta_arr * price_arr + alpha_arr)  # shape = (sample_size,)

        # Simulate consumer choices using a multinomial draw
        choice_arr = np.zeros((sample_size, 1))
        for i in range(sample_size):
            choice_arr[i] = np.random.multinomial(1, [choice_prob_arr[i], 1 - choice_prob_arr[i]])[0]

        data = pd.DataFrame(
            np.hstack([choice_arr, price_arr.reshape(-1, 1), X]), 
            columns=['choice', 'price', *[f'cov_{i}' for i in range(self.n_covariates)]]
        )

        return data

    def sample_individuals(self, sample_size: int, seed: int = None) -> np.ndarray:
        # set seed
        if seed is not None:
            np.random.seed(seed)
        
        # Generate covariates
        X = np.abs(np.random.normal(size=(sample_size, self.n_covariates)) @ self.chol_cov)
        X = X / X.max()
        return X
    
    def predict_proba(self, customers: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        Predict the purchasing probability for a given set of customers.
        
        Params:
        -------
        customers: np.ndarray, shape (sample_size, n_features)
            Customer data with covariates and prices.

        prices: np.ndarray, shape (sample_size,)
            Prices of the products.
        
        Returns:
        --------
        proba: np.ndarray, shape (sample_size, )
            Purchasing probability for each customer.
        """
        assert customers.shape[1] == self.n_covariates, f"Number of covariates should be {self.n_covariates}."
        assert prices.shape[0] == customers.shape[0], f"Number of prices should be equal to the number of customers."

        # Compute true intercept (alpha_arr) and price sensitivity (beta_arr)
        alpha_arr = customers @ self.theta_arr[:, 0]  # shape = (sample_size, )
        beta_arr = customers @ self.theta_arr[:, 1]  # shape = (sample_size, )

        # Calculate the purchasing probability
        choice_prob_arr = expit(beta_arr * prices + alpha_arr).reshape(-1, 1)  # shape = (sample_size, 1)


        return np.hstack([1 - choice_prob_arr, choice_prob_arr])
