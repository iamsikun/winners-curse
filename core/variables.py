import numpy as np
from scipy.stats import norm


class RandomVariable(object):
    def sample(self, size: int):
        raise NotImplementedError()
    
    def cdf(self, x: float):
        """ 
        Cumulative distribution function
        """
        raise NotImplementedError()

class ContinuousRandomVariable(RandomVariable): 
    def pdf(self, x: float):
        """
        Probability density function (might not exist for some distributions)
        """
        raise NotImplementedError()
    
class DiscreteRandomVariable(RandomVariable): 
    def pmf(self, x: float):
        """ 
        Probability mass function
        """
        raise NotImplementedError()
    
class UnivariateGaussian(ContinuousRandomVariable):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def sample(self, size: int):
        return np.random.normal(self.mean, self.std, size)
    
    def pdf(self, x: float):
        return norm.pdf(x, self.mean, self.std)

    def cdf(self, x: float):
        return norm.cdf(x, self.mean, self.std)
    
class UnivariateExponential(ContinuousRandomVariable):
    def __init__(self, rate: float):
        self.rate = rate
    
    def sample(self, size: int):
        return np.random.exponential(1 / self.rate, size)
    
    def pdf(self, x: float):
        return self.rate * np.exp(-self.rate * x)
    
    def cdf(self, x: float):
        return 1 - np.exp(-self.rate * x)
    
    @property
    def mean(self):
        return 1 / self.rate
    

class Bernoulli(DiscreteRandomVariable):
    def __init__(self, p: float):
        self.p = p
    
    def sample(self, size: int):
        return np.random.binomial(1, self.p, size)
    
    def pmf(self, x: float):        
        if x == 1:
            return self.p
        elif x == 0:
            return 1 - self.p
        else:
            return 0
    
    def cdf(self, x: float):
        if x < 0:
            return 0
        elif x == 0:
            return 1 - self.p
        elif x < 1:
            return 1 - self.p
        else: 
            return 1