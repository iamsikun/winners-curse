import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Iterable

import pandas as pd 
import numpy as np

from sklearn.linear_model import Lasso
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LogisticRegression
import econml.grf as grf
from scipy.special import expit

from winners_curse.dgp import ContinuousSegments
from winners_curse.m_out_of_n_bootstrap import choose_best_m


class CorrectLinearRegression(object):
    def __init__(
        self, treatment_space: np.ndarray, fit_intercept: bool = True, 
        fit_x: bool = False, 
    ):
        self.treatment_space = treatment_space  # shape=(n_treatments, )
        self.fit_intercept = fit_intercept  # bool
        self.fit_x = fit_x  # bool
        self.model = None 

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        self.model = OLS(Y, self.transform(X, T)).fit()

        return self
    
    def transform(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        # turn T into one hot encoding with two columns
        T = np.eye(self.treatment_space.shape[0])[T.astype(int)]

        variables_list= [X * T]

        if self.fit_x:
            variables_list.append(X)  # shape = (sample_size, 1)
        
        if self.fit_intercept:
            variables_list.append(np.ones((X.shape[0], 1)))

        return np.concatenate(variables_list, axis=1)
    
    def predict(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        return self.transform(X, T) @ self.model.params
    
    def predict_effect(self, X: np.ndarray) -> np.ndarray:
        return X @ self.model.params[:self.treatment_space.shape[0]]  # shape = (sample_size, n_treatments)
    

class CorrectLogisticRegression(object):
    def __init__(self, treatment_space: np.ndarray, fit_intercept: bool = True):
        self.treatment_space = treatment_space
        self.fit_intercept = fit_intercept
        self.model = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        self.model = LogisticRegression(penalty=None, fit_intercept=self.fit_intercept).fit(
            self.transform(X, T), Y
        )

        return self

    def transform(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        # turn T into one hot encoding with two columns
        T = np.eye(self.treatment_space.shape[0])[T.astype(int)]

        variables_list= [X * T]

        return np.concatenate(variables_list, axis=1)
    
    def predict(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        purchase_proba = self.model.predict_proba(self.transform(X, T))

        return (purchase_proba[:, 1] > purchase_proba[:, 0]).astype(int)  # shape = (sample_size, n_treatments)
        
    
    def predict_effect(self, X: np.ndarray) -> np.ndarray:
        return X @ self.model.coef_  # shape = (sample_size, n_treatments)


class SegmentModel(object):
    def __init__(self, treatment_space: np.ndarray, response_type: str, threshold: float = 0):
        self.treatment_space = treatment_space  # shape=(n_treatments, )
        self.threshold = threshold  # float
        self.response_type = response_type  # str, either 'continuous' or 'logit'
        assert response_type in ['continuous', 'logit'], 'response_type must be either "continuous" or "logit"'
        
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

        if self.response_type == 'continuous':
            return te_est_arr
        elif self.response_type == 'logit':
            return (1 - expit(- te_est_arr) >= 0.5).astype(int)
        else:
            raise ValueError('response_type must be either "continuous" or "logit"')

    def predict_effect(self, X: np.ndarray) -> np.ndarray:
        """ 
        Predict treatment effect for each customer given their features
        """
        incremental_effect = (self.segment_outcome_df.iloc[:, 1] - self.segment_outcome_df.iloc[:, 0]).values  # shape=(n_segments, )
        segments_arr = self.get_segment(X)
        return incremental_effect[segments_arr]


class CausalForest(object):
    def __init__(self, n_estimators: int, max_depth: int, response_type: str, **kwargs):
        assert response_type in ['continuous', 'logit'], 'response_type must be either "continuous" or "logit"'
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.response_type = response_type
        
        self.model = grf.CausalForest(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray):
        self.model.fit(X=X, T=T, y=y)
        return self
    
    def predict(self, X: np.ndarray, T: np.ndarray):
        if self.response_type == 'continuous':
            return self.model.predict(X=X).flatten() * T
        elif self.response_type == 'logit':
            return (1 - expit(- self.model.predict(X=X).flatten() * T) >= 0.5).astype(int)
        else:
            raise ValueError('response_type must be either "continuous" or "logit"')
    
    def predict_effect(self, X: np.ndarray):
        return self.model.predict(X=X).flatten()


demand_model_dict = {
    'correct_model': CorrectLinearRegression,
    'segment_model': SegmentModel,
    'causal_forest': CausalForest,
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
    demand_model = experiment_params['dm_info']['model']
    dm_params = experiment_params['dm_info']['params']
    sample_size = data_params['sample_size']
    n_customers = data_params['n_customers']

    # create data generation process
    dgp = ContinuousSegments(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        # generate data
        X, T, Y = dgp.sample(sample_size=sample_size, seed=experiment_id)
        targ_customers = dgp.sample_individuals(sample_size=n_customers, seed=n_experiments + experiment_id)

        # initialize and fit demand model
        dm = demand_model(treatment_space=dgp.treatment_space, **dm_params).fit(X, T, Y)

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

        # solve clairvoyant optimization
        clairvoyant_decision, clairvoyant_val = optimize(
            demand_model=dgp, 
            X=targ_customers, 
            treatment_space=dgp.treatment_space
        )

        # bookkeeping
        result_dict = {
            'plugin_decision': plugin_decision,
            'clairvoyant_decision': clairvoyant_decision,
            'clairvoyant_val': clairvoyant_val,
            'plugin_val_est': plugin_val_est,
            'true_plugin_val': true_plugin_val,
        }

        # fit remaining estimators
        for name in estimators_dict.keys():
            temp_decision, temp_val_est = estimators_dict[name]['estimator'](
                X=X, T=T, Y=Y, 
                targ_customers=targ_customers,
                treatment_space=dgp.treatment_space, 
                demand_model=demand_model,
                dm_params=dm_params, 
                **estimators_dict[name]['params'], 
            )
            temp_decision = temp_decision if temp_decision is not None else plugin_decision
            temp_val_true = obj_func(
                demand_model=dgp, 
                X=targ_customers, T=temp_decision
            )
            result_dict.update({
                f'{name}_val_est': temp_val_est,
                f'{name}_wc': temp_val_est - true_plugin_val, 
                f'{name}_decision': temp_decision,
                f'{name}_val_true': temp_val_true,
            }) 

        return result_dict
    
    # run experiments
    if verbose: 
        print(f'Running {n_experiments} experiments...')

    result_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_experiment)(i) for i in range(n_experiments)
    )

    return result_list


def calculate_winners_curse_measures(
    result_records: list, estimators_dict: dict, data_params: dict
) -> dict:
    # initialize dictionary
    wc_measure_dict = {}

    est_val_arr = np.array([record['plugin_val_est'] for record in result_records])
    true_val_arr = np.array([record['true_plugin_val'] for record in result_records])
    correct_decision_arr = np.array([record['plugin_decision'] == record['clairvoyant_decision'] for record in result_records])

    wc_measure_dict.update({
        'nc_val_est_avg': np.mean(est_val_arr), 'nc_val_est_se': np.std(est_val_arr) / np.sqrt(data_params['sample_size']), 
        'nc_val_true_avg': np.mean(true_val_arr), 'nc_val_true_se': np.std(true_val_arr) / np.sqrt(data_params['sample_size']),
        'nc_correct_decision_rate': np.mean(correct_decision_arr),
    })

    # no correction
    nc_wc_arr = est_val_arr - true_val_arr  # winner's curse
    nc_wc_pct_arr = nc_wc_arr / np.abs(true_val_arr)  # winner's curse percentage of true value

    wc_measure_dict.update({
        'nc_wc_arr': nc_wc_arr, 'nc_wc_pct_arr': nc_wc_pct_arr,
        'nc_wc_avg': np.mean(nc_wc_arr), 'nc_wc_se': np.std(nc_wc_arr) / np.sqrt(data_params['sample_size']),
        'nc_wc_pct_avg': np.mean(nc_wc_arr / np.abs(true_val_arr)), 'nc_wc_pct_se': np.std(nc_wc_arr / np.abs(true_val_arr)) / np.sqrt(data_params['sample_size']),
        'nc_wc_rmse': np.sqrt(np.mean(np.square(nc_wc_arr))),
    })

    # for each estimator
    for estimator in estimators_dict.keys():
        est_val_arr = np.array([record[f'{estimator}_val_est'] for record in result_records])
        true_val_arr = np.array([record[f'{estimator}_val_true'] for record in result_records])

        correct_decision_arr = np.array([record[f'{estimator}_decision'] == record['clairvoyant_decision'] for record in result_records])

        wc_arr = est_val_arr - true_val_arr
        wc_pct_arr = wc_arr / np.abs(true_val_arr)  # winner's curse percentage of true value

        wc_measure_dict.update({
            f'{estimator}_correct_decision_rate': np.mean(correct_decision_arr),
            f'{estimator}_val_est_avg': np.nanmean(est_val_arr), f'{estimator}_val_est_se': np.nanstd(est_val_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_val_true_avg': np.mean(true_val_arr), f'{estimator}_val_true_se': np.std(true_val_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_wc_arr': wc_arr, f'{estimator}_wc_pct_arr': wc_pct_arr,
            f'{estimator}_wc_avg': np.nanmean(wc_arr), f'{estimator}_wc_se': np.nanstd(wc_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_wc_pct_avg': np.nanmean(wc_pct_arr), f'{estimator}_wc_pct_se': np.nanstd(wc_pct_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_wc_rmse': np.sqrt(np.nanmean(np.square(wc_arr))),
        })

    return wc_measure_dict


def get_wc_boot_dstn(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    targ_customers: np.ndarray, treatment_space: np.ndarray,
    n_bootstraps: int, demand_model: callable, dm_params: dict, 
    n_jobs: int = -1, verbose: bool = False
) -> np.ndarray:
    # attributes
    sample_size = X.shape[0]

    # initialize placeholders for the bootstrap distribution
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))

    # get demand model fitted using all data
    emp_dm = demand_model(treatment_space=treatment_space, **dm_params).fit(X, T, Y)

    def fit_single_bootstrap():
        # create bootstrap sample
        boot_idx = np.random.choice(np.arange(sample_size), sample_size, replace=True)

        # fit demand model using bootstrap sample
        boot_dm = demand_model(treatment_space=treatment_space, **dm_params).fit(X[boot_idx], T[boot_idx], Y[boot_idx])

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
    demand_model: callable, dm_params: dict,
    n_bootstraps: int, power: float = 0.9, 
    candidate_powers: Iterable[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99],
    n_jobs: int = -1, verbose: bool = False
) -> np.ndarray:
    # parameter checks
    assert power == 'auto' or 0 < power < 1, 'Power must be between 0 and 1 or "auto"'

    # attributes
    sample_size = X.shape[0]

    # initialize placeholders for the bootstrap distribution
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))

    # get demand model fitted using all data
    emp_dm = demand_model(treatment_space=treatment_space, **dm_params).fit(X, T, Y)

    def fit_single_bootstrap(boot_sample_size):
        # create bootstrap sample
        boot_idx = np.random.choice(np.arange(sample_size), boot_sample_size, replace=True)

        # fit demand model using bootstrap sample
        boot_dm = demand_model(treatment_space=treatment_space, **dm_params).fit(X[boot_idx], T[boot_idx], Y[boot_idx])

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
    
    if power == 'auto':
        wc_dstn_list = [
            np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(fit_single_bootstrap)(boot_sample_size=int(sample_size ** cand_power)) for _ in range(n_bootstraps)
        )) for cand_power in candidate_powers
        ]
        best_m_idx = choose_best_m(wc_dstn_list)
        boot_wc_dstn_arr = wc_dstn_list[best_m_idx]
    else:
        boot_wc_dstn_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(fit_single_bootstrap)(boot_sample_size=int(sample_size ** power)) for _ in range(n_bootstraps)
        ))

    return boot_wc_dstn_arr


def get_wc_num_boot_dstn(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    targ_customers: np.ndarray, treatment_space: np.ndarray,
    demand_model: callable, dm_params: dict,
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
    emp_dm = demand_model(treatment_space=treatment_space, **dm_params).fit(X, T, Y)

    def fit_single_bootstrap():
        # create bootstrap sample
        boot_idx = np.random.choice(np.arange(sample_size), sample_size, replace=True)

        # fit demand model using bootstrap sample
        boot_dm = demand_model(treatment_space=treatment_space, **dm_params).fit(X[boot_idx], T[boot_idx], Y[boot_idx])

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
    demand_model: callable, dm_params: dict,
    bootstrap_method: str = 'standard', 
    n_jobs: int = -1, verbose: bool = False, **kwargs
) -> tuple:
    # estiamte empirical treatment effects
    emp_dm = demand_model(treatment_space=treatment_space, **dm_params).fit(X, T, Y)

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
        dm_params=dm_params,
        n_jobs=n_jobs, verbose=verbose, 
        **kwargs
    )

    return plugin_decision, plugin_val_est - np.nanmean(boot_wc_dstn_arr)
