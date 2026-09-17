import numpy as np
import pytest

from winners_curse import analysis


@pytest.mark.parametrize('kind', ['snr', 'n_treatments', 'bernoulli', 'functional_form', 'noise_dist', 'targeting_model_comparison'])
def test_latex_tables_use_raw_configured_names_with_display_name_fallback(kind):
    latex_name = r'\begin{tabular}[c]{@{}c@{}}Hybrid\\ Selective Inference\end{tabular}'
    config = {
        'dgp_params': {
            'noise_vars': [{'std': 1.0}],
            'base_effects': [{'value': 0.1}, {'value': 0.2}],
        },
        'estimators_dict': {
            'hybrid_si': {'display_name': 'Hybrid SI', 'latex_name': latex_name},
            'sample_splitting': {'display_name': 'Sample Splitting'},
        },
    }
    values = {
        'nc_wc_arr': np.array([0.03, 0.05]),
        'hybrid_si_wc_arr': np.array([0.01, 0.03]),
        'sample_splitting_wc_arr': np.array([0.02, 0.04]),
    }
    key = {'functional_form': 'linear', 'noise_dist': ('Gaussian',)}.get(kind, (0.1, 0.2))
    results = {key: values}
    kwargs = dict(normalize=False, stats='mean', include_std=False)
    if kind == 'targeting_model_comparison':
        latex = analysis.generate_targeting_model_comparison_latex_table(
            {2: values}, results, config, **kwargs
        )
    else:
        name = f'generate_{kind}_sum_stats_latex_table'
        if kind == 'functional_form':
            name = 'generate_functional_form_latex_table'
        latex = getattr(analysis, name)(results, config, **kwargs)

    assert latex_name + ' & 0.02' in latex
    assert 'Hybrid SI' not in latex
    assert 'Sample Splitting & 0.03' in latex
    assert latex.index('No Correction') < latex.index(latex_name) < latex.index('Sample Splitting')
