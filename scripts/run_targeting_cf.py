import os
import pickle
import multiprocessing

# Fix for Windows multiprocessing + OpenBLAS issues
# Set OpenBLAS to use only 1 thread per process to avoid access violations
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso

from winners_curse.targeting import *
from winners_curse.variables import *

optimization_params = {
    'rank_and_select': {
        'optimizer': optimize, 'params': {}, 'display_name': 'Rank and Select'
    }
}

dgp_params = {
    'base_effect_vars': [None, None], 
    'cust_feat_var': UnivariateGaussian(0, 1), 
    'char_func': lambda x: x**2 + x, 
    'noise_var': UnivariateGaussian(0, 1), 
    'response_type': 'continuous', 
    'dgp_seed': 0
}

data_params = {'sample_size': 2500, 'targ_sample_size': 1000} 

experiment_params = {
    'n_repeats': 500, 
    'estimator': {
        'estimator': CausalForestDML, 
        'params': {
            'n_treatments': len(dgp_params['base_effect_vars']), 
            'model_t': LogisticRegression(C=0.01), 
            'model_y': Lasso(alpha=0.01), 
            'n_estimators': 100, 
            'max_depth': 5, 
            'n_jobs': 1,  # Set to 1 since we're already parallelizing at experiment level
            'discrete_treatment': True
        }
    }
}

estimators_dict = {
    'standard_bootstrap': {
        'estimator': bootstrap_correction_estimate, 
        'params': {'n_bootstraps': 100, 'n_jobs': 1},  # Set to 1 since we're already parallelizing at experiment level
        'display_name': 'Standard Bootstrap',
        'latex_name': 'Standard Bootstrap'
    },
    'mn_bootstrap': {
        'estimator': bootstrap_correction_estimate, 
        'params': {'bootstrap_method': 'm_out_of_n', 'n_bootstraps': 100, 'power': 0.95, 'n_jobs': 1},  # Set to 1 since we're already parallelizing at experiment level
        'display_name': 'm-out-of-n Bootstrap',
        'latex_name': 'm-out-of-n Bootstrap'
    },
    'sample_splitting': {
        'estimator': sample_splitting_estimate, 
        'params': {'estimation_split': 0.5, }, 
        'display_name': 'Sample Splitting', 
        'latex_name': 'Sample Splitting'
    }, 
    'eb_normal': {
        'estimator': empirical_bayes_estimate, 
        'params': {'prior': 'normal', 'dof': 3, 'bin_width': 0.2}, 
        'display_name': 'Empirical Bayes',
        'latex_name': 'Empirical Bayes'
    }, 
    'hybrid_si': {
        'estimator': selective_inference_estimate,
        'params': {'method': 'hybrid', 'quantile': 0.5, 'n_jobs': 1},  # Set to 1 since we're already parallelizing at experiment level
        'display_name': 'Hybrid SI',
        'latex_name': '\\begin{tabular}[c]{@{}c@{}}Hybrid\\\\ Selective Inference\\end{tabular}'
    }, 
}

tau_list = [
    (1, 1.005), (1, 1.01), (1, 1.02), 
]

def run_experiment():
    tau_result_dict = {tau: None for tau in tau_list}

    for te_tuple in tau_list:
        print(f'\n(tau_1, tau_2)= {te_tuple}')
        
        dgp_params['base_effect_vars'] = [PointMass(te_tuple[0]), PointMass(te_tuple[1])]

        # Use fewer parallel jobs on Windows to avoid access violations
        # With OpenBLAS_NUM_THREADS=1, we can use more processes safely
        max_jobs = min(24, multiprocessing.cpu_count())  # Limit to 8 or CPU count, whichever is smaller
        
        result_records = repeated_experiment(
            optimization_params=optimization_params, 
            dgp_params=dgp_params, 
            data_params=data_params, 
            experiment_params=experiment_params, 
            estimators_dict=estimators_dict, 
            verbose=True, 
            n_jobs=max_jobs
        )

        tau_result_dict[te_tuple] = calculate_winners_curse_measures(
            result_records=result_records, 
            optimization_params=optimization_params, 
            estimators_dict=estimators_dict, 
            data_params=data_params
        )

    return tau_result_dict


if __name__ == '__main__':
    tau_result_dict = run_experiment()

    with open('results/targeting_delta_tau.pkl', 'wb') as f:
        pickle.dump(tau_result_dict, f)