import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

from winners_curse.utils import *


def significance_level(mean: float, se: float) -> str: 
    """
    Returns the significance level of a mean given its standard error. 
    """
    if mean > 0:
        if mean - 2.58 * se > 0:
            return '***'
        elif mean - 1.96 * se > 0:
            return '**'
        elif mean - 1.645 * se > 0: 
            return '*'
        else:
            return ''
    else:  # mean < 0
        if mean + 2.58 * se < 0:
            return '***'
        elif mean + 1.96 * se < 0:
            return '**'
        elif mean + 1.645 * se < 0: 
            return '*'
        else:
            return ''
        

def significance_level_with_p_val(p_val) -> str:
    """
    Returns the significance level of a p-value. 
    """
    if p_val < 0.01:
        return '***'
    elif p_val < 0.05:
        return '**'
    elif p_val < 0.1: 
        return '*'
    else:
        return ''