import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
import numpy as np 
import statsmodels.api as sm
from scipy.interpolate import CubicSpline
from patsy import dmatrix

class EmpiricalBayes(object):
    def __init__(self, dof: int = 5, bin_width: float = 0.2, sigma: float = None):
        self.dof = dof
        self.bin_width = bin_width

        # initialize attributes
        self.spline = None
        self.std_est = sigma

    def fit(self, x: np.ndarray):
        # Step 0: estimate sigma
        self.std_est = np.std(x) if self.std_est is None else self.std_est

        # Step 1: Bin data for Poisson regression
        bins = np.arange(min(x), max(x) + self.bin_width, self.bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        counts, _ = np.histogram(x, bins=bins)

        # Step 2: Poisson Regression for Log-density Estimation
        exog = dmatrix(f"cr(x, df={self.dof})", {'x': bin_centers}, return_type='dataframe')
        model = sm.GLM(counts, exog, family=sm.families.Poisson()).fit()
        log_counts_est = np.log(model.predict(exog))
        # Step 3:
        self.spline = CubicSpline(bin_centers, log_counts_est, bc_type='natural')

        return self

    def predict(self, x: np.ndarray):
        # Apply Tweedie's Formula
        log_density_derivative = self.spline.derivative()(x)
        bayes_correction = self.std_est**2 * log_density_derivative

        return x + bayes_correction