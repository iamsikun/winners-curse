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
    

# a point mass at a given value, i.e., a degenerate distribution
class PointMass(DiscreteRandomVariable):
    def __init__(self, value: float):
        self.value = value
    
    def sample(self, size: int):
        return np.full(size, self.value)
    
    def pmf(self, x: float):
        return 1 if x == self.value else 0
    
    def cdf(self, x: float):
        return 1 if x >= self.value else 0
    
    @property
    def mean(self):
        return self.value
    
    @property
    def variance(self):
        return 0


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
    
    @property
    def std(self):
        return 1 / self.rate
    

class StudentT(ContinuousRandomVariable):
    def __init__(self, df: float):
        self.df = df
    
    def sample(self, size: int):
        return np.random.standard_t(self.df, size)
    
    def pdf(self, x: float):
        return (np.math.gamma((self.df + 1) / 2) / 
                (np.sqrt(self.df * np.pi) * np.math.gamma(self.df / 2))) * (1 + x**2 / self.df)**(-(self.df + 1) / 2)
    
    def cdf(self, x: float):
        return norm.cdf(x)  # Using normal approximation for simplicity
    
    @property 
    def mean(self):
        return 0 if self.df > 1 else np.nan
    
    @property
    def std(self):
        return np.sqrt(self.df / (self.df - 2)) if self.df > 2 else np.nan
    

class Laplace(ContinuousRandomVariable):
    def __init__(self, mu: float, b: float):
        self.mu = mu
        self.b = b
    
    def sample(self, size: int):
        return np.random.laplace(self.mu, self.b, size)
    
    def pdf(self, x: float):
        return (1 / (2 * self.b)) * np.exp(-np.abs(x - self.mu) / self.b)
    
    def cdf(self, x: float):
        return np.where(x < self.mu, 
                        0.5 * np.exp((x - self.mu) / self.b), 
                        1 - 0.5 * np.exp(-(x - self.mu) / self.b))
    
    @property
    def mean(self):
        return self.mu
    
    @property
    def std(self):
        return np.sqrt(2) * self.b
    

class Logistic(ContinuousRandomVariable):
    def __init__(self, mu: float, s: float):
        self.mu = mu
        self.s = s
    
    def sample(self, size: int):
        return np.random.logistic(self.mu, self.s, size)
    
    def pdf(self, x: float):
        return (np.exp(-(x - self.mu) / self.s) / 
                (self.s * (1 + np.exp(-(x - self.mu) / self.s)**2)))
    
    def cdf(self, x: float):
        return 1 / (1 + np.exp(-(x - self.mu) / self.s))
    
    @property
    def mean(self):
        return self.mu
    
    @property
    def std(self):
        return self.s * np.pi / np.sqrt(3)
    
class Uniform(ContinuousRandomVariable):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
    
    def sample(self, size: int):
        return np.random.uniform(self.a, self.b, size)
    
    def pdf(self, x: float):
        return np.where((x >= self.a) & (x <= self.b), 1 / (self.b - self.a), 0)
    
    def cdf(self, x: float):
        return np.where(
            x < self.a, 
            0,  
            np.where(x > self.b, 1, (x - self.a) / (self.b - self.a))
        )
    
    @property
    def mean(self):
        return (self.a + self.b) / 2
    
    @property
    def std(self):
        return (self.b - self.a) / np.sqrt(12)
    

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
        
    @property
    def mean(self):
        return self.p
    
    @property
    def variance(self):
        return self.p * (1 - self.p)