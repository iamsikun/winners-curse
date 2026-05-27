"""
Utility for merging partial experiment results into a base results pickle.

Typical workflow: rerun a single estimator (via ``--estimators`` on the
experiment scripts) to produce a "delta" pickle that only contains arrays for
that estimator, then merge those arrays into a full paper-results pickle
without rerunning every other estimator.

The results pickle layout is a dict keyed by parameter combo (e.g., a tau
tuple, or a ``(depth, tau)`` tuple), with inner dicts containing per-estimator
arrays named ``{estimator_key}_{wc|val_est|val_true|selection}_arr`` plus
shared arrays like ``effect_est_arr`` and the ``nc_*`` no-correction arrays.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Iterable


_ARRAY_SUFFIX = re.compile(r'_(wc|val_est|val_true|selection)_arr$')

_SHARED_KEYS = {
    'effect_est_arr',
    'nc_wc_arr',
    'nc_val_est_arr',
    'nc_val_true_arr',
    'nc_selection_arr',
}


def _estimator_key(inner_key: str) -> str | None:
    """Extract the estimator prefix for a per-estimator array key, or None."""
    m = _ARRAY_SUFFIX.search(inner_key)
    if not m or inner_key in _SHARED_KEYS:
        return None
    return inner_key[: m.start()]


def merge_results(
    base_path: str | Path,
    delta_path: str | Path,
    out_path: str | Path,
    drop_estimators: Iterable[str] = (),
    allow_partial_keys: bool = False,
) -> dict:
    """
    Merge per-estimator arrays from ``delta_path`` into ``base_path`` and
    write the combined pickle to ``out_path``.

    Behavior:
    - Requires the parameter-combo keysets to match (unless
      ``allow_partial_keys=True``, in which case only shared combos are
      merged and delta-only combos are ignored with a printed warning).
    - For each shared param combo: any estimator whose arrays appear in
      ``delta`` is copied into ``base``, overwriting existing arrays for
      that estimator. Shared no-correction arrays and arrays for estimators
      that appear only in ``base`` are left untouched.
    - If ``drop_estimators`` is given, arrays for those estimator prefixes
      are removed from ``base`` before merging. Use this to retire stale
      estimator keys (e.g., ``m_out_of_n_bootstrap``) that were renamed in
      newer configs.

    Parameters
    ----------
    base_path: path to the original results pickle.
    delta_path: path to the partial-rerun results pickle.
    out_path: path to write the merged pickle. Must differ from base_path.
    drop_estimators: iterable of estimator prefixes to strip from base.
    allow_partial_keys: if True, permit delta to cover a subset of base's
        parameter-combo keys instead of requiring an exact match.

    Returns
    -------
    The merged results dict (also written to ``out_path``).
    """
    base_path = Path(base_path)
    delta_path = Path(delta_path)
    out_path = Path(out_path)

    if out_path.resolve() == base_path.resolve():
        raise ValueError(
            f"Refusing to overwrite base pickle in place. "
            f"out_path ({out_path}) must differ from base_path ({base_path})."
        )

    with open(base_path, 'rb') as f:
        base = pickle.load(f)
    with open(delta_path, 'rb') as f:
        delta = pickle.load(f)

    if not isinstance(base, dict) or not isinstance(delta, dict):
        raise TypeError(
            "Both pickles must contain a dict keyed by parameter combo."
        )

    base_keys = set(base.keys())
    delta_keys = set(delta.keys())

    missing_in_base = delta_keys - base_keys
    if missing_in_base:
        raise KeyError(
            f"delta has parameter combos not present in base: {sorted(missing_in_base)}"
        )

    missing_in_delta = base_keys - delta_keys
    if missing_in_delta and not allow_partial_keys:
        raise KeyError(
            f"base has parameter combos not present in delta: "
            f"{sorted(missing_in_delta)}. Pass allow_partial_keys=True to "
            f"proceed anyway (those combos will be left untouched)."
        )

    drop_estimators = tuple(drop_estimators)
    shared_combos = base_keys & delta_keys

    for combo in shared_combos:
        base_inner = base[combo]
        delta_inner = delta[combo]
        if not isinstance(base_inner, dict) or not isinstance(delta_inner, dict):
            raise TypeError(
                f"Inner value at {combo!r} must be a dict; got "
                f"base={type(base_inner).__name__}, delta={type(delta_inner).__name__}"
            )

        # Drop stale estimator arrays from base if requested.
        for est in drop_estimators:
            for k in [k for k in base_inner if _estimator_key(k) == est]:
                del base_inner[k]

        # Copy every per-estimator array that appears in delta into base.
        for k, v in delta_inner.items():
            est = _estimator_key(k)
            if est is None:
                # Skip shared arrays like effect_est_arr, nc_*; base already has them.
                continue
            base_inner[k] = v

    with open(out_path, 'wb') as f:
        pickle.dump(base, f)

    return base


def _parse_drop_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [name.strip() for name in raw.split(',') if name.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-estimator arrays from a partial-rerun results pickle "
            "(`delta`) into a full results pickle (`base`), writing the "
            "combined result to `out`."
        )
    )
    parser.add_argument('--base', required=True, help='Path to full base results.pkl')
    parser.add_argument('--delta', required=True, help='Path to partial delta results.pkl')
    parser.add_argument('--out', required=True, help='Path to write merged results.pkl')
    parser.add_argument(
        '--drop',
        default='',
        help=(
            'Comma-separated list of estimator prefixes to strip from base '
            'before merging (e.g., "m_out_of_n_bootstrap"). Useful when the '
            'estimator was renamed between runs.'
        ),
    )
    parser.add_argument(
        '--allow-partial-keys',
        action='store_true',
        help=(
            'Permit delta to cover a subset of base parameter-combo keys. '
            'Combos not in delta are left untouched.'
        ),
    )
    args = parser.parse_args(argv)

    merge_results(
        base_path=args.base,
        delta_path=args.delta,
        out_path=args.out,
        drop_estimators=_parse_drop_list(args.drop),
        allow_partial_keys=args.allow_partial_keys,
    )
    print(f"Wrote merged pickle to {args.out}")


if __name__ == '__main__':
    main()
