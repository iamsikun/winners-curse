import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd

from scipy.special import expit

from core.variables import RandomVariable, ContinuousRandomVariable

class DataGenerationProcess(object):
    def sample(self) -> pd.DataFrame:
        raise NotImplementedError

    def sample_individuals(self) -> np.ndarray:
        raise NotImplementedError
    

class RCTs(DataGenerationProcess):
    def __init__(
        self, 
        n_experiments: int, 
        base_effects: list[RandomVariable], 
        noise_vars: list[ContinuousRandomVariable] = None, 
        response_type: str = 'continuous', 
        dgp_seed: int = 0, 
    ):
        # parameters check 
        if response_type == 'continuous':
            assert len(base_effects) == len(noise_vars), "Length of base effects and noise variables must be the same"
        assert all([isinstance(base_effect, RandomVariable) for base_effect in base_effects]), "All base effects must be RandomVariable instances"
        assert response_type in ['continuous', 'bernoulli'], "Response type must be either 'continuous' or 'bernoulli'"

        # store attributes
        self.n_arms = len(base_effects)
        self.n_experiments = n_experiments
        self.base_effects = base_effects
        self.noise_vars = noise_vars
        self.response_type = response_type 
        self.dgp_seed = dgp_seed

        # draw prior
        self.treatment_effects = self.draw_prior_mean(seed=dgp_seed)  # (n_arms, n_experiments)

        if self.response_type == 'bernoulli': 
            # check if all treatment effects are within (0, 1)
            assert all([0 < treatment_effect < 1 for treatment_effect in self.treatment_effects.flatten()]), "Treatment effects must be within (0, 1) for Bernoulli response type. Please change the base effects"

    def draw_prior_mean(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        return np.concatenate([
            prior_dstn.sample(self.n_experiments)[np.newaxis, :]  # shape = (1, n_experiments)
            for prior_dstn in self.base_effects
        ], axis=0)  # shape = (n_arms, n_experiments)

    def sample(self, sample_size: int, seed: int = None) -> tuple:
        """ 
        Sample from the data generation process, fixing the prior

        Params: 
        -------
        sample_size: int
            Number of samples to draw from each arm
        seed: int
            Random seed for reproducibility
        
        Returns:
        --------
        samples: list[np.ndarray]
            List of samples for each arm, where each sample is of shape (sample_size, n_experiments)
        """
        if seed is not None:
            np.random.seed(seed)

        # draw treatment and control groups 
        if self.response_type == 'continuous':
            samples = [
                np.repeat(self.treatment_effects[arm_id, :].reshape(1, -1), sample_size, axis=0) + self.noise_vars[arm_id].sample((sample_size, self.n_experiments))
                for arm_id in range(self.n_arms)
            ]
        elif self.response_type == 'bernoulli':
            samples = [
                np.random.binomial(1, self.treatment_effects[arm_id, :], size=(sample_size, self.n_experiments))  # shape = (sample_size, n_experiments)
                for arm_id in range(self.n_arms)
            ]

        return samples


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


class SingleSegmentTreatmentSelection(DataGenerationProcess):
    valid_response_types = ['continuous', 'bernoulli', 'logit']
    def __init__(
        self, te_arr: np.ndarray, noise_std: float = None, response_type: str = 'continuous'
    ):
        """ 
        Data generation process for single segment with multiple treatments. 

        Params:
        -------
        te_arr: np.ndarray
            Array of treatment effects for each treatment value.
        noise_std: float
            Standard deviation of the noise.
        response_type: str
            Type of response variable. Can be 'continuous', 'bernoulli', or 'logit'.
        """
        assert te_arr.shape[0] > 1, "Number of treatments should be greater than 1."
        assert response_type in self.valid_response_types, "Response type should be continuous, bernoulli, or logit."
        if response_type == 'continuous':
            assert noise_std is not None, "Noise standard deviation should be provided for continuous response type."

        self.te_arr = te_arr
        self.treatment_index_space = np.arange(te_arr.shape[0])
        self.noise_std = noise_std
        self.response_type = response_type
        

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
        treatments = np.random.choice(self.treatment_index_space, size=sample_size)  # shape = (sample_size, )

        # calculate outcomes
        outcomes = self.outcome_generation(treatments, seed)  # shape = (sample_size, )

        return treatments, outcomes 
    
    def sample_individuals(self, sample_size: int) -> np.ndarray:
        return np.ones(shape=(sample_size, ))  # shape = (sample_size, )
    
    def outcome_generation(self, treatments: np.ndarray, seed: int = None) -> np.ndarray:
        """ 
        Generate outcomes for the given sample size. 

        Params:
        -------
        treatments: np.ndarray, shape = (sample_size, )
            Array of treatment values.
        seed: int
            Random seed for reproducibility.

        Returns:
        --------
        np.ndarray, shape = (sample_size, )
            Array of outcomes.
        """
        if seed is not None:
            np.random.seed(seed)

        treatment_effect_arr = self.te_arr[treatments]  # shape = (sample_size, )

        if self.response_type == 'continuous':
            noises = np.random.normal(loc=0, scale=self.noise_std, size=(treatments.shape[0], ))  # shape = (sample_size, )
            return treatment_effect_arr + noises  # shape = (sample_size, )
        elif self.response_type == 'bernoulli':  # bernoulli response
            return np.random.binomial(n=1, p=treatment_effect_arr)  # shape = (sample_size, )
        elif self.response_type == 'logit':
            proba_arr =  1 - expit(-treatment_effect_arr)  # shape = (sample_size, )
            return np.random.binomial(n=1, p=proba_arr)  # shape = (sample_size, )
        else: # invalid response types
            raise ValueError("Invalid response type. Choose from {}.".format(self.valid_response_types))
        
class MultipleSegments(DataGenerationProcess):
    def __init__(
        self, segment_te_arr: np.ndarray,
        treatment_space: np.ndarray, noise_std: float = 1.0, 
    ):
        self.segment_arr = np.arange(segment_te_arr.shape[0])
        self.segment_te_arr = segment_te_arr
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
    

class MultipleSegmentsTreatmentSelection:
    valid_response_types = ['continuous', 'bernoulli', 'logit']
    def __init__(
        self, base_te_arr: np.ndarray, n_segments: int, 
        noise_std: float = None, response_type: str = 'continuous', 
        dgp_seed: int = 0
    ):
        """
        Data generation process for multiple segments with multiple treatments.

        Params:
        -------
        base_te_arr: np.ndarray, shape = (n_treatments, )
            Array of treatment effects for each treatment value.
        noise_std: float
            Standard deviation of the noise.
        response_type: str
            Type of response variable. Can be 'continuous', 'bernoulli', or 'logit'.
        """
        if dgp_seed is not None:
            np.random.seed(dgp_seed)

        assert response_type in self.valid_response_types, "Response type should be continuous, bernoulli, or logit."
        if response_type == 'continuous':
            assert noise_std is not None, "Noise standard deviation should be provided for continuous response type."
        
        self.base_te_arr = base_te_arr  # shape = (n_segments, n_treatments)
        self.n_segments = n_segments 
        self.n_treatments = base_te_arr.shape[0]

        self.te_arr =  np.random.normal(base_te_arr, 0.1, size=(self.n_segments, self.n_treatments))

        self.segment_idx_arr = np.arange(self.n_segments)
        self.treatment_idx_arr = np.arange(self.n_treatments)
        self.noise_std = noise_std
        self.response_type = response_type

    def sample_individuals(self, sample_size: int, seed: int = None) -> np.ndarray:
        """ 
        Sample individuals from the DGP. 
        """
        if seed is not None:
            np.random.seed(seed)

        return np.random.choice(self.segment_idx_arr, size=sample_size)

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
        tuple: 
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
        segment_arr = self.sample_individuals(sample_size)

        # sample treatments
        treatment_arr = np.random.choice(self.treatment_idx_arr, size=sample_size)

        # compute outcomes
        outcome_arr = self.outcome_generation(segment_arr, treatment_arr, seed)

        return segment_arr, treatment_arr, outcome_arr
    
    def outcome_generation(self, segments: np.ndarray, treatments: np.ndarray, seed: int = None) -> np.ndarray:
        """ 
        Generate outcomes for the given sample size. 

        Params:
        -------
        segments: np.ndarray, shape = (sample_size, )
            Array of segment values.
        treatments: np.ndarray, shape = (sample_size, )
            Array of treatment values.
        seed: int
            Random seed for reproducibility.

        Returns:
        --------
        np.ndarray, shape = (sample_size, )
            Array of outcomes.
        """
        if seed is not None:
            np.random.seed(seed)

        treatment_effect_arr = self.te_arr[segments, treatments]  # shape = (sample_size, )

        if self.response_type == 'continuous':
            noises = np.random.normal(loc=0, scale=self.noise_std, size=(segments.shape[0], ))
            return treatment_effect_arr + noises
        elif self.response_type == 'bernoulli':  # bernoulli response
            return np.random.binomial(n=1, p=treatment_effect_arr)
        elif self.response_type == 'logit':
            proba_arr = 1 - expit(-treatment_effect_arr)
            return np.random.binomial(n=1, p=proba_arr)
        else: # invalid response types
            raise ValueError("Invalid response type. Choose from {}.".format(self.valid_response_types))


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


class ContinuousSegments(DataGenerationProcess):
    valid_response_types = ['continuous', 'bernoulli', 'logit']
    def __init__(
        self, te_arr: np.ndarray, beta: float, const: float, noise_std: float, 
        response_type: str = 'continuous'
    ):
        super(ContinuousSegments, self).__init__()

        # attributes
        self.treatment_space = np.arange(te_arr.shape[0])
        self.noise_std = noise_std

        # generate coefficients
        self.te_arr = te_arr
        self.beta = beta 
        self.const = const
        self.response_type = response_type

    def sample_individuals(self, sample_size: int, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        return np.random.normal(0, 1, (sample_size, 1))
    
    def sample(self, sample_size: int, seed: int = None) -> tuple:
        if seed is not None:
            np.random.seed(seed)

        features = self.sample_individuals(sample_size)  # shape=(sample_size, n_features)
        treatments = np.random.choice(self.treatment_space, sample_size)  # shape=(sample_size,)
        
        treatment_effect_arr =  features.flatten() * self.te_arr[treatments] + self.const + features.flatten() * self.beta  # shape = (sample_size, )

        if self.response_type == 'continuous':
            noises = np.random.normal(loc=0, scale=self.noise_std, size=(sample_size, ))  # shape = (sample_size, )
            outcomes = treatment_effect_arr + noises
        elif self.response_type == 'bernoulli':
            outcomes = np.random.binomial(n=1, p=treatment_effect_arr)  # shape = (sample_size, )
        elif self.response_type == 'logit':
            proba_arr = 1 - expit(-treatment_effect_arr)
            outcomes = np.random.binomial(n=1, p=proba_arr)  # shape = (sample_size, )
        else:
            raise ValueError("Invalid response type. Choose from {}.".format(self.valid_response_types))

        return features, treatments, outcomes
    
    def predict(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        treatment_effect_arr = X.flatten() * self.te_arr[T] + self.const + X.flatten() * self.beta
        if self.response_type == 'continuous':
            return treatment_effect_arr
        elif self.response_type == 'bernoulli':
            return np.random.binomial(n=1, p=treatment_effect_arr)
        elif self.response_type == 'logit':
            proba_arr = 1 - expit(-treatment_effect_arr)
            return np.random.binomial(n=1, p=proba_arr)
            
