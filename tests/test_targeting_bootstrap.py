import numpy as np
import pytest

from winners_curse.targeting import _draw_cv_valid_bootstrap_indices


def test_draw_cv_valid_bootstrap_indices_preserves_size_and_class_support():
    treatments = np.array([0, 0, 0, 1, 1, 2])
    outcomes = np.array([0, 0, 0, 0, 1, 1])

    np.random.seed(7)
    indices = _draw_cv_valid_bootstrap_indices(
        treatments=treatments,
        outcomes=outcomes,
        sample_size=12,
        discrete_treatment=True,
        discrete_outcome=True,
        n_splits=2,
    )

    assert indices.shape == (12,)
    sampled_strata = np.column_stack((treatments[indices], outcomes[indices]))
    assert np.all(np.unique(sampled_strata, axis=0, return_counts=True)[1] >= 2)


def test_draw_cv_valid_bootstrap_indices_fails_when_support_is_impossible():
    treatments = np.array([0, 1])
    outcomes = np.array([0, 1])

    with pytest.raises(RuntimeError, match="Could not draw a bootstrap sample"):
        _draw_cv_valid_bootstrap_indices(
            treatments=treatments,
            outcomes=outcomes,
            sample_size=3,
            discrete_treatment=True,
            discrete_outcome=True,
            n_splits=2,
            max_attempts=10,
        )
