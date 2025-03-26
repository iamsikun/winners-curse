import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))


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