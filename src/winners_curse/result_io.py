"""Load saved experiments without requiring the large policy-selection arrays."""

import json
import pickle
import re
from pathlib import Path


RESULT_FILENAMES = ('results_slim.pkl', 'results_with_new_moon.pkl', 'results.pkl')


def find_latest_result_dir(results_root, experiment_names):
    """Find the newest timestamped run with both configuration and results.

    Params:
    -------
    results_root: str or Path
        Directory containing experiment runs.
    experiment_names: str or iterable of str
        Exact experiment prefixes, including any historical aliases. Estimator
        subsets and other similarly named sweeps are not matched implicitly.

    Returns:
    --------
    Path
        Latest usable run. Checkpoints may contain only part of a sweep; this
        function does not merge runs or infer completion from their timestamps.
    """
    if isinstance(experiment_names, str):
        experiment_names = [experiment_names]
    pattern = re.compile(
        r'(?:' + '|'.join(re.escape(name) for name in experiment_names)
        + r')_(\d{8}_\d{6})'
    )
    candidates = []
    for directory in Path(results_root).glob('*'):
        match = pattern.fullmatch(directory.name)
        if (match and (directory / 'config.json').is_file()
                and any((directory / name).is_file() for name in RESULT_FILENAMES)):
            candidates.append((match[1], directory.name, directory))
    if not candidates:
        raise FileNotFoundError(
            f'No saved run for {", ".join(experiment_names)} in {results_root}'
        )
    return max(candidates)[2]


def load_saved_results(result_dir):
    """Load a trusted local run, preferring its compact results.

    Params:
    -------
    result_dir: str or Path
        Run directory with config.json and a supported results pickle. Legacy
        amended results take precedence over the original raw results.

    Returns:
    --------
    tuple[dict, dict, Path]
        Results, configuration, and the exact file loaded for provenance.
    """
    result_dir = Path(result_dir)
    result_path = next(
        (result_dir / name for name in RESULT_FILENAMES
         if (result_dir / name).is_file()),
        None,
    )
    if result_path is None:
        raise FileNotFoundError(f'No supported results pickle in {result_dir}')
    config = json.loads((result_dir / 'config.json').read_text())
    with result_path.open('rb') as handle:
        results = pickle.load(handle)
    return results, config, result_path
