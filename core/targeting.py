import os 
import sys 
import pickle 
sys.path.insert(0, os.path.abspath('.'))

from joblib import Parallel, delayed
from typing import Union

import numpy as np
from scipy.stats import ttest_ind

from scipy.optimize import fsolve 
from scipy.stats import truncnorm
import econml.grf as grf
from scipy.special import expit

# visualization parameters
import matplotlib.pyplot as plt
import seaborn as sns
tick_label_size = 12
legend_label_size = 12
axis_label_size = 14
title_size = 18
plt.rcParams['font.family'] = 'serif'

from core.dgp import Targeting
from core.bayes_methods import *


##########
# Visualization
##########
def plot_incremental_hte(
    dgp: Targeting, 
    model = None, 
    ci: bool = False, 
    x_lb=-1, x_ub=1
):
    assert dgp.n_treatments == 2, "Currently only supports 2 treatments"

    fig, axes = plt.subplots(1, dgp.n_treatments - 1, figsize=(10 * (dgp.n_treatments - 1), 6.18))
    if dgp.n_treatments == 2:
        axes = [axes]

    # plot the true HTE
    x = np.linspace(x_lb, x_ub, 1000).reshape(-1, 1)
    true_inc_te_arr = dgp.predict_incremental_effect(x)
    est_inc_te_arr = model.predict_incremental_effect(x)

    for treatment_id in range(dgp.n_treatments - 1):
        axes[treatment_id].plot(
            x, true_inc_te_arr[:, treatment_id], label='True', color='black', linewidth=2
        )

        # plot the estimated HTE
        axes[treatment_id].plot(
            x, est_inc_te_arr[:, treatment_id], label='Estimated', color='red', linewidth=2
        )

        if ci:
            # plot the confidence intervals
            ci_lb, ci_ub = model.predict_incremental_effect_interval(x)
            axes[treatment_id].fill_between(
                x.flatten(), 
                ci_lb[:, treatment_id], 
                ci_ub[:, treatment_id], 
                alpha=0.2, 
                color='red'
            )

        axes[treatment_id].set_xlabel('Customer Feature', fontsize=axis_label_size)
        axes[treatment_id].set_ylabel('Incremental Heterogeneous Treatment Effect', fontsize=axis_label_size)
        axes[treatment_id].set_title(f'Incremental HTE of Treatment {treatment_id + 2} Against Treatment 1', fontsize=title_size)
        axes[treatment_id].legend(fontsize=legend_label_size)
        axes[treatment_id].tick_params(axis='both', which='major', labelsize=tick_label_size)
        axes[treatment_id].grid()

    plt.tight_layout()
    plt.show()



##########
# Estimation
##########

class CausalForest(object):
    def __init__(self, n_estimators: int, max_depth: int, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        self.model = grf.CausalForest(
            n_estimators=n_estimators, max_depth=max_depth, **kwargs
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        self.model.fit(X=X, T=T, y=Y)
        return self
    
    def predict_incremental_effect(self, X: np.ndarray):
        return self.model.predict(X=X)  # shape = (sample_size, n_treatments)
    
    def predict_incremental_effect_interval(self, X: np.ndarray, alpha: float = 0.05):
        return self.model.predict_interval(X=X, alpha=alpha)


##########
# Selection
##########

def obj_func(
    selection: np.ndarray, 
    cust_feautres: np.ndarray, 
    demand_model, 
    treatment_effects: np.ndarray = None
) -> float:
    """
    Objective function to be maximized by the targeting algorithm. 
    This is the expected value of the treatment effect for the selected customers. 
    """
    if treatment_effects is None:
        treatment_effects = demand_model.predict_incremental_effect(cust_feautres)  # shape = (sample_size, n_treatments - 1)
    else:
        assert treatment_effects.shape[0] == selection.shape[0], "Treatment effects must be the same length as the selection vector."

    # calculate the expected value of the treatment effect for the selected customers
    return treatment_effects[np.arange(selection.shape[0]), selection].mean()


def optimize(demand_model, cust_features: np.ndarray) -> tuple:
    """ 
    Optimize targeting decision

    Params:
    -------
    demand_model: object
        demand model

    cust_features: np.ndarray, shape=(n_customers, n_features)
        features of the individuals

    treatment_space: np.ndarray, shape=(n_treatments, )
        treatment space

    Returns:
    --------
    tuple
        optimal decision, optimal value
    """
    # calculate objective values under each treatment
    te_arr = demand_model.predict_incremental_effect(cust_features)  # shape = (sample_size, n_treatments - 1)

    # add the control group to the treatment effects
    te_arr = np.hstack((np.zeros((te_arr.shape[0], 1)), te_arr))  # shape = (sample_size, n_treatments)

    # choose the better treatment for each customer 
    opt_decision = np.argmax(te_arr, axis=1)  # shape = (sample_size, )

    # calculate the optimal value
    opt_value = obj_func(
        selection=opt_decision, 
        cust_feautres=cust_features, 
        demand_model=demand_model,
        treatment_effects=te_arr
    )

    return opt_decision, opt_value