import os 
import sys 
import pickle 
sys.path.insert(0, os.path.abspath('.'))

from joblib import Parallel, delayed
from typing import Union
from itertools import product

import warnings 

import numpy as np

from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning

from winners_curse.bayes_methods import *


def compute_purchase_probabilities(fixed_effects: np.ndarray, assortment: np.ndarray) -> np.ndarray:
    """
    Given the fixed effects and an assortment of product indices,
    compute the purchase probabilities of each product in the assortment
    using the MNL model.
    
    Params:
    -------
    fixed_effects: np.ndarray, shape = (n_products, )
        The fixed effects of the customer.
    assortment: np.ndarray, shape = (n_products, )
        The indices of the products in the assortment.

    Returns:
    --------
    np.ndarray
        The purchase probabilities of each product in the assortment.
    """
    # check that assortment is a binary matrix 
    assert np.all(np.isin(assortment, [0, 1])), "Assortment must be a binary matrix"

    # replace zeros in the assortment with negative infinity
    # to ensure that the product is not chosen
    assortment = assortment.astype(float)
    neg_inf_mask = assortment == 0  # mask for zeros in the assortment

    # compute the utilities for each assortment
    exponent = fixed_effects * assortment
    exponent[neg_inf_mask] = -np.inf
    exp_utilities = np.exp(exponent)

    # Compute probabilities: outside option probability + product probabilities
    denom = 1 + np.sum(exp_utilities)
    prob_outside = 1 / denom
    prob_products = exp_utilities / denom

    return np.concatenate((prob_products, [prob_outside]))


def simulate_purchase(fixed_effects: np.ndarray, assortment: np.ndarray):
    """
    Given the fixed effects and an assortment of product indices,
    simulate a single customer purchase using the MNL model.
    The outside option has utility 0.

    Params:
    -------
    fixed_effects: np.ndarray
        The fixed effects of the customer.
    assortment: np.ndarray
        The indices of the products in the assortment.

    Returns:
    --------
    str
        The product index that the customer chose, or "outside" if the customer chose the outside option.
    """    
    # check that assortment is a binary matrix
    assert np.all(np.isin(assortment, [0, 1])), "Assortment must be a binary matrix"

    # Compute purchase probabilities, shape = (n_products + 1, )
    purchase_probs = compute_purchase_probabilities(fixed_effects, assortment)

    # Draw a choice (0 => outside, 1,... => corresponding product in assortment)
    choice= np.random.choice(len(purchase_probs), p=purchase_probs)

    if choice == len(purchase_probs) - 1:
        return "outside"
    else:
        return choice