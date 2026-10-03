"""Protection and repeat-alignment checks for the in-place EB rerun."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/rerun_targeting_eb_inplace.py'
spec = importlib.util.spec_from_file_location('rerun_targeting_eb_inplace', SCRIPT)
rerun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rerun)


def example():
    saved = {(1, 1.01): {
        'nc_wc_arr': np.array([.1, .2, .3]),
        'nc_val_true_arr': np.array([1., 2., 3.]),
        'nc_val_est_arr': np.array([1.1, 2.2, 3.3]),
        'sample_splitting_val_est_arr': np.array([1., np.nan, 3.]),
        'eb_normal_wc_arr': np.array([.05, .1, .2]),
        'eb_normal_val_est_arr': np.array([1.05, 2.1, 3.2]),
    }}
    rows = {i: {
        'naive': saved[(1, 1.01)]['nc_val_est_arr'][i],
        'truth': float(i + 1), 'normal': [1.05, 2.1, 3.2][i],
        'value': [1.02, np.nan, 20.][i],
        'failure': 'fit failed' if i == 1 else None,
    } for i in range(3)}
    return saved, {(1, 1.01): rows}


def test_replacement_preserves_other_estimators_and_full_repeat_alignment():
    saved, records = example()
    before = rerun.non_eb_digest(saved)
    updated = rerun.replace_eb_arrays(saved, records)
    values = updated[(1, 1.01)]
    assert rerun.non_eb_digest(updated) == before
    assert 'eb_normal_wc_arr' not in values
    assert 'eb_normal_wc_arr' in saved[(1, 1.01)]
    np.testing.assert_allclose(values['eb_tweedies_wc_arr'], [.02])
    np.testing.assert_allclose(values['eb_tweedies_wc_full_arr'], [.02, np.nan, 17.])
    np.testing.assert_array_equal(values['eb_tweedies_valid_mask_arr'], [True, False, False])
    np.testing.assert_array_equal(values['eb_tweedies_fit_failed_arr'], [False, True, False])


def test_validation_rejects_incomplete_or_different_reconstruction():
    saved, records = example()
    assert rerun.validate_records(saved, records) == {'naive': 0., 'truth': 0., 'normal': 0.}
    records[(1, 1.01)][0]['naive'] += .1
    with pytest.raises(AssertionError):
        rerun.validate_records(saved, records)
    del records[(1, 1.01)][0]
    with pytest.raises(AssertionError, match='Incomplete repetitions'):
        rerun.validate_records(saved, records)


def test_selection_validation_rejects_changed_policy():
    saved, records = example()
    saved[(1, 1.01)]['nc_selection_arr'] = np.array([[0, 1], [1, 0], [0, 0]])
    for i, selection in enumerate(saved[(1, 1.01)]['nc_selection_arr']):
        records[(1, 1.01)][i]['selection_sha256'] = rerun.selection_digest(selection)
    rerun.validate_records(saved, records, check_selections=True)
    records[(1, 1.01)][1]['selection_sha256'] = rerun.selection_digest([0, 1])
    with pytest.raises(AssertionError, match='Selection mismatch'):
        rerun.validate_records(saved, records, check_selections=True)


def test_config_update_changes_only_empirical_bayes_selector():
    text = '''# keep this comment
estimators_dict:
  eb_normal:
    estimator: empirical_bayes_estimate
    params:
      prior: normal
      dof: 3
      bin_width: 0.2
  another_method:
    params:
      prior: normal
'''
    actual = rerun.update_config_text(text)
    parsed = yaml.safe_load(actual)
    assert actual.startswith('# keep this comment\n')
    assert parsed['estimators_dict']['eb_tweedies']['params']['prior'] == 'tweedies'
    assert parsed['estimators_dict']['another_method']['params']['prior'] == 'normal'
    assert rerun.update_config_text(actual) == actual
