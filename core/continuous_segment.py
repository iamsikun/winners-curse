import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
from tqdm import tqdm
from joblib import Parallel, delayed

import pandas as pd 
import numpy as np

from sklearn.linear_model import Lasso
from statsmodels.regression.linear_model import OLS

from core.dgp import ContinuousSegments


class CorrectLinearRegression(object):
    def __init__(self, treatment_space: np.ndarray):
        self.treatment_space = treatment_space  # shape=(n_treatments, )
        self.model = None 

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        self.model = OLS(Y, self.transform(X, T)).fit()

        return self
    
    @staticmethod
    def transform(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        return np.concatenate([X * T[:, np.newaxis], X, np.ones((X.shape[0], 1))], axis=1)
    
    def predict(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        return self.transform(X, T) @ self.model.params

class SegmentModel(object):
    def __init__(self, treatment_space: np.ndarray, threshold: float = 0):
        self.treatment_space = treatment_space  # shape=(n_treatments, )
        self.threshold = threshold  # float
        
        # initialize attributes 
        self.segment_outcome_df = None   # shape=(2, n_treatments)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        # create segments 
        segments_arr = self.get_segment(X)

        # estimate treatment effects for each segment
        self.segment_outcome_df = pd.DataFrame({
            'segment': segments_arr, 'treatment': T, 'outcome': Y
        }).groupby(['segment', 'treatment']).mean().reset_index().pivot(
            index='segment', columns='treatment', values='outcome'
        )

        return self

    def get_segment(self, X: np.ndarray) -> np.ndarray:
        """ 
        Put customers into two segments based on their features
        """
        return (X > self.threshold).astype(int).flatten()
    
    def predict(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """ 
        Predict outcome for each customer given their features and treatment
        """
        # get segments
        segments_arr = self.get_segment(X)

        # get treatment effects
        treatment_idx_dict = {treatment: self.segment_outcome_df.columns.get_loc(treatment) for treatment in self.treatment_space}

        T_idx_arr = np.array([treatment_idx_dict[treatment] for treatment in T])

        te_est_arr = self.segment_outcome_df.values[segments_arr, T_idx_arr]

        return te_est_arr

    

demand_model_dict = {
    'correct_model': CorrectLinearRegression,
    'segment_model': SegmentModel,
}


def obj_func(demand_model, X: np.ndarray, T: np.ndarray) -> float:
    """ 
    Calculate the objective function of the model

    Params:
    -------
    demand_model: object
        model object
    X: np.ndarray
        features of the individuals
    T: np.ndarray
        treatment assignment

    Returns:
    --------
    float
        objective function value
    """
    assert X.shape[0] == T.shape[0], 'X and T must have the same number of rows'

    return demand_model.predict(X, T).mean()


def optimize_with_counterfactual(
    counterfactual_arr: np.ndarray, treatment_space: np.ndarray
) -> tuple:
    """ 
    Optimize targeting decision with counterfactual predictions

    Params:
    -------
    counterfactual_arr: np.ndarray, shape=(n_treatments, n_customers)
        counterfactual predictions
    treatment_space: np.ndarray, shape=(n_treatments, )
        treatment space

    Returns:
    --------
    tuple
        optimal decision, optimal value
    """
    opt_decision = treatment_space[np.argmax(counterfactual_arr, axis=0)]
    opt_val = np.max(counterfactual_arr, axis=0).mean()

    return opt_decision, opt_val


def optimize(demand_model, X: np.ndarray, treatment_space: np.ndarray) -> tuple:
    """ 
    Optimize targeting decision

    Params:
    -------
    demand_model: object
        demand model

    X: np.ndarray, shape=(n_customers, n_features)
        features of the individuals

    treatment_space: np.ndarray, shape=(n_treatments, )
        treatment space

    Returns:
    --------
    tuple
        optimal decision, optimal value
    """
    # initialize placeholders, shape=(n_treatments, n_customers)
    counterfactual_val = np.array([
        demand_model.predict(X, t * np.ones((X.shape[0], ), dtype=int)) for t in treatment_space
    ])

    return optimize_with_counterfactual(counterfactual_val, treatment_space)


def repeated_experiments(
    dgp_params: dict, 
    data_params: dict, 
    experiment_params: dict, 
    estimators_dict: dict, 
    n_jobs: float = -1, 
    verbose: bool = False, 
) -> list:
    # extract parameters
    n_experiments = experiment_params['n_experiments']
    demand_model = experiment_params['demand_model']
    sample_size = data_params['sample_size']
    n_customers = data_params['n_customers']
    segment_threshold = experiment_params['segment_threshold']

    # create data generation process
    dgp = ContinuousSegments(**dgp_params)

    # placeholders for results
    result_list = [None] * n_experiments

    def run_single_experiment(experiment_id: int) -> dict:
        # generate data
        X, T, Y = dgp.sample(sample_size=sample_size, seed=experiment_id)
        targ_customers = dgp.sample_individuals(sample_size=n_customers, seed=n_experiments + experiment_id)

        # initialize and fit demand model
        if demand_model == 'segment_model':
            dm = demand_model_dict[demand_model](dgp.treatment_space, segment_threshold).fit(X, T, Y)
        else:
            dm = demand_model_dict[demand_model](dgp.treatment_space).fit(X, T, Y)

        # optimize targeting decision
        plugin_decision, plugin_val_est = optimize(
            demand_model=dm, 
            X=targ_customers, 
            treatment_space=dgp.treatment_space
        )

        # calculate the true value of the plugin decision
        true_plugin_val = obj_func(
            demand_model=dgp, 
            X=targ_customers, T=plugin_decision
        )

        # bookkeeping
        result_dict = {
            'plugin_val_est': plugin_val_est,
            'true_plugin_val': true_plugin_val,
            'plugin_wc': plugin_val_est - true_plugin_val, 
            'plugin_wc_pct': (plugin_val_est - true_plugin_val) / true_plugin_val,
        }

        # fit remaining estimators
        for name in estimators_dict.keys():
            targ_val_est = estimators_dict[name]['estimator'](
                X=X, T=T, Y=Y, 
                targ_customers=targ_customers,
                treatment_space=dgp.treatment_space, 
                demand_model=demand_model, 
                **estimators_dict[name]['params'], 
            )
            result_dict.update({
                f'{name}_val_est': targ_val_est,
                f'{name}_wc': targ_val_est - true_plugin_val, 
                f'{name}_wc_pct': (targ_val_est - true_plugin_val) / true_plugin_val,
            }) 

        return result_dict
    
    # run experiments
    if verbose: 
        print(f'Running {n_experiments} experiments...')

    result_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_experiment)(i) for i in range(n_experiments)
    )

    return result_list


def get_wc_boot_dstn(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    targ_customers: np.ndarray, treatment_space: np.ndarray,
    n_bootstraps: int, demand_model: str, 
    n_jobs: int = -1, verbose: bool = False
) -> np.ndarray:
    # attributes
    sample_size = X.shape[0]

    # initialize placeholders for the bootstrap distribution
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))

    # get demand model fitted using all data
    emp_dm = demand_model_dict[demand_model](treatment_space).fit(X, T, Y)

    def fit_single_bootstrap():
        # create bootstrap sample
        boot_idx = np.random.choice(np.arange(sample_size), sample_size, replace=True)

        # fit demand model using bootstrap sample
        boot_dm = demand_model_dict[demand_model](treatment_space).fit(X[boot_idx], T[boot_idx], Y[boot_idx])

        # get plugin decision and value using bootstrap demand model
        boot_decision, boot_val_est = optimize(
            demand_model=boot_dm, 
            X=targ_customers, 
            treatment_space=treatment_space, 
        )

        # evaluate bootstrap decision using empirical demand model
        emp_boot_val_est = obj_func(
            demand_model=emp_dm, 
            X=targ_customers, 
            T=boot_decision
        )

        return boot_val_est - emp_boot_val_est
    
    # run bootstrap
    boot_wc_dstn_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_single_bootstrap)() for _ in range(n_bootstraps)
    ))

    return boot_wc_dstn_arr


def get_wc_m_out_of_n_boot_dstn(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    targ_customers: np.ndarray, treatment_space: np.ndarray,
    demand_model: str, 
    n_bootstraps: int, power: float = 0.9, 
    n_jobs: int = -1, verbose: bool = False
) -> np.ndarray:
    # parameter checks
    assert 0 < power < 1, 'Power should be between 0 and 1.' 

    # attributes
    sample_size = X.shape[0]
    m = int(sample_size ** power)

    # initialize placeholders for the bootstrap distribution
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))

    # get demand model fitted using all data
    emp_dm = demand_model_dict[demand_model](treatment_space).fit(X, T, Y)

    def fit_single_bootstrap():
        # create bootstrap sample
        boot_idx = np.random.choice(np.arange(sample_size), m, replace=True)

        # fit demand model using bootstrap sample
        boot_dm = demand_model_dict[demand_model](treatment_space).fit(X[boot_idx], T[boot_idx], Y[boot_idx])

        # get plugin decision and value using bootstrap demand model
        boot_decision, boot_val_est = optimize(
            demand_model=boot_dm, 
            X=targ_customers, 
            treatment_space=treatment_space, 
        )

        # evaluate bootstrap decision using empirical demand model
        emp_boot_val_est = obj_func(
            demand_model=emp_dm, 
            X=targ_customers, 
            T=boot_decision
        )

        return boot_val_est - emp_boot_val_est
    
    # run bootstrap
    boot_wc_dstn_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_single_bootstrap)() for _ in range(n_bootstraps)
    ))

    return boot_wc_dstn_arr


def get_wc_num_boot_dstn(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    targ_customers: np.ndarray, treatment_space: np.ndarray,
    demand_model: str, 
    n_bootstraps: int, power: float = -0.45, 
    n_jobs: int = -1, verbose: bool = False
) -> np.ndarray:
    # parameter checks
    assert 0 > power >= -0.5, 'Power should be between 0 and -0.5.'

    # attributes
    sample_size = X.shape[0]
    epsilon_n = sample_size ** power

    # initialize placeholders for the bootstrap distribution
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))

    # get treatment effect estimate using all data (empirical estimate)
    emp_dm = demand_model_dict[demand_model](treatment_space).fit(X, T, Y)

    def fit_single_bootstrap():
        # create bootstrap sample
        boot_idx = np.random.choice(np.arange(sample_size), sample_size, replace=True)

        # fit demand model using bootstrap sample
        boot_dm = demand_model_dict[demand_model](treatment_space).fit(X[boot_idx], T[boot_idx], Y[boot_idx])

        # optimize targeting decision with counterfactual predictions
        boot_decision, _ = optimize(
            demand_model=boot_dm, 
            X=targ_customers, 
            treatment_space=treatment_space, 
        )

        # evaluate boot_decision with perturbed prediction
        emp_pred = emp_dm.predict(targ_customers, boot_decision) 
        boot_pred = boot_dm.predict(targ_customers, boot_decision)
        norm_error = np.sqrt(sample_size) * (boot_pred - emp_pred)  # prediction error
        perturb_pred = emp_pred + epsilon_n * norm_error
        perturb_val_est = perturb_pred.mean()

        # evaluate boot_decision with empirical prediction
        emp_boot_val_est = emp_pred.mean()

        return perturb_val_est - emp_boot_val_est
    
    # run bootstrap
    boot_wc_dstn_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_single_bootstrap)() for _ in range(n_bootstraps)
    ))

    return boot_wc_dstn_arr


def bootstrap_correction_estimate(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    targ_customers: np.ndarray, treatment_space: np.ndarray,
    demand_model: str, 
    bootstrap_method: str = 'standard', 
    n_jobs: int = -1, verbose: bool = False, **kwargs
) -> float:
    # estiamte empirical treatment effects
    emp_dm = demand_model_dict[demand_model](treatment_space).fit(X, T, Y)

    # get the empirical targeting policy
    plugin_decision, plugin_val_est = optimize(
        demand_model=emp_dm,
        X=targ_customers,
        treatment_space=treatment_space
    )

    # dictionary of available bootstrap methods
    boot_methods_dict = {
        'standard': get_wc_boot_dstn, 
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn, 
        'numerical': get_wc_num_boot_dstn, 
    }

    # get the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = boot_methods_dict[bootstrap_method](
        X=X, T=T, Y=Y, 
        targ_customers=targ_customers, 
        treatment_space=treatment_space, 
        demand_model=demand_model, 
        n_jobs=n_jobs, verbose=verbose, 
        **kwargs
    )

    return plugin_val_est - np.nanmean(boot_wc_dstn_arr)
