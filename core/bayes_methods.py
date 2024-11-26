import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
import numpy as np 
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline

class EmpiricalBayes(object):
    def __init__(self, n_degree: int = 5, n_knots: int = 5, bin_width: float = 0.2, smoothing: float = 0.5):
        self.n_degree = n_degree
        self.n_knots = n_knots
        self.bin_width = bin_width
        self.smoothing = smoothing

        # initialize attributes
        self.spline = None
        self.std_est = None 

    def fit(self, x: np.ndarray):
        # Step 0: estimate sigma
        self.std_est = np.std(x)

        # Step 1: Bin data for Poisson regression
        bins = np.arange(min(x), max(x) + self.bin_width, self.bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        counts, _ = np.histogram(x, bins=bins)

        # Step 2: Poisson Regression for Log-density Estimation
        def create_polynomials(x, degree):
            return np.array([x ** i for i in range(1, degree + 1)]).T
        exog = sm.add_constant(create_polynomials(bin_centers, degree=self.n_degree))  # Add intercept
        model = sm.GLM(counts, exog, family=sm.families.Poisson()).fit()
        log_density_estimates = np.log(model.predict(exog))

        # Step 3: Smooth Log-density with Splines
        self.spline = UnivariateSpline(bin_centers, log_density_estimates, k=self.n_knots, s=self.smoothing)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Apply Tweedie's Formula
        log_density_derivative = self.spline.derivative()(x)
        bayes_correction = self.std_est**2 * log_density_derivative

        return x + bayes_correction
    
    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).predict(x)