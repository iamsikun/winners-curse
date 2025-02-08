import numpy as np
from scipy.stats import ks_2samp


from typing import Iterable


def choose_best_m(distribution_list: Iterable[np.ndarray]) -> int:
    """ 
    Given a list of bootstrap distributions coming from different bootstrap sample sizes (m), 
    choose the best m based on the discrepancy between the distributions. 

    This function follows the rule in Bickel and Sakov (2008, Statistica Sinica)

    Params:
    -------
    distribution_list: Iterable[np.ndarray]
        a list of bootstrap distributions, each element is an array of shape (n_bootstraps, )

    Returns:
    --------
    int: the index of the best m in the distribution_list
    """
    # calculate pairwise discrepancies
    discp_list = [None] * (len(distribution_list) - 1)  # list of pairwise discrepancies
    for idx, (prev_dstn, current_dstn) in enumerate(zip(distribution_list[:-1], distribution_list[1:])):
        ks_stat, _ = ks_2samp(prev_dstn, current_dstn)
        discp_list[idx] = ks_stat

    # choose the m with the smallest discrepancy
    min_discp_idx = np.argmin(discp_list)

    return min_discp_idx

