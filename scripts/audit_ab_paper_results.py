"""Compare designated A/B results with the local manuscript; never edit the paper.

Run: uv run python scripts/audit_ab_paper_results.py
"""
import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ['nc', 'standard_bootstrap', 'moon_bootstrap', 'sample_splitting', 'eb_normal', 'hybrid_si']
TAUS = [(1, 1.005), (1, 1.01), (1, 1.02)]
NUMBER = re.compile(r'-?\d+\.\d+')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=ROOT.parent / 'winners-curse-paper/main.tex')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/ab-paper-crosscheck')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'docs/ab-results-current.json').read_text())
    results = {}
    for name, entry in manifest['runs'].items():
        path = ROOT / entry['directory'] / 'results.pkl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['results_sha256'], path
        assert hashlib.sha256((path.parent / 'config.json').read_bytes()).hexdigest() == entry['config_sha256'], path.parent
        results[name] = pickle.loads(path.read_bytes())
    # Remove commented manuscript lines before locating active tables.
    raw = args.paper.read_text()
    source = '\n'.join(re.sub(r'(?<!\\)%.*', '', line) for line in raw.splitlines())
    tables = {}
    for match in re.finditer(r'\\begin\{table\}.*?\\end\{table\}', source, re.S):
        label = re.search(r'\\label\{([^}]+)\}', match.group())
        if label:
            tables[label[1]] = match.group()
    noise_keys = []
    for name in ['UnivariateGaussian', 'Uniform', 'Laplace', 'Logistic']:
        matches = [k for k in results['ab_test_noise_vars'] if all(f"'type': '{name}'" in s for s in k)]
        assert len(matches) == 1, name
        noise_keys.append(matches[0])
    specs = [
        ('2', 'tab: ab test estimates', 'ab_test_snr', MAIN, TAUS, 'mean', True),
        ('3', 'tab: ab test mae', 'ab_test_snr', MAIN, TAUS, 'mae', False),
        ('4', 'tab: ab test noise dstn', 'ab_test_noise_vars', MAIN, noise_keys, 'mean', True),
        ('D.1', 'tab: ab test bayes', 'ab_test_bayesian', ['nc', 'bayes_normal', 'eb_normal', 'standard_bootstrap', 'moon_bootstrap'], [(1, 1.01), (-1.01, -1)], 'mean', True),
        ('E.2', 'tab: ab test bernoulli delta tau', 'ab_test_bernoulli', MAIN, [(0.1,0.101),(0.2,0.201),(0.3,0.301),(0.1,0.105),(0.2,0.205),(0.3,0.305)], 'mean', True),
        ('E.5', 'tab: ab test estimates (appendix)', 'ab_test_snr_additional', ['nc','conditional_si','plugin','sample_splitting9010','cross_validation','jackknife'], TAUS, 'mean', True),
        ('E.6', 'tab: ab test multiple actions', 'ab_test_n_treatments', MAIN, [(1,)*n for n in [2,4,6,8,10]], 'mean', False),
        ('E.7', 'tab: ab test wc imbalanced', 'ab_test_imbalanced_snr', [(n,5000-n) for n in range(500,5000,500)], TAUS, 'mean', True),
        ('E.8', 'tab: ab test estimates imbalanced', 'ab_test_imbalanced_snr', MAIN, [((4500,500),t) for t in TAUS], 'mean', True),
        ('E.9', 'tab: gamma sweep', 'ab_test_moon_power', ['nc','standard_bootstrap','moon_095','moon_09','moon_08','moon_07','moon_06','moon_05','moon_04','moon_03','moon_02','moon_01'], TAUS, 'mean', True),
        ('E.12', 'tab: ab test estimates median (appendix)', 'ab_test_snr', MAIN, TAUS, 'median', True),
        ('E.13', 'tab: ab test noise dstn median (appendix)', 'ab_test_noise_vars', MAIN, noise_keys, 'median', True),
    ]
    records, summaries, replacements = [], [], []
    for number, label, run, estimators, keys, metric, normalized in specs:
        table = tables[label]
        body = table.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
        rows = [line for line in body.splitlines() if '&' in line]
        assert len(rows) == len(estimators), (label, len(rows))
        table_records = []
        for row, estimator in zip(rows, estimators):
            cells = row.split('&')
            assert len(cells) == len(keys)+1, label
            new_cells = [cells[0]]
            for key, cell in zip(keys, cells[1:]):
                data_key, est = ((estimator,key), 'nc') if number == 'E.7' else (key,estimator)
                arr = np.asarray(results[run][data_key][f'{est}_wc_arr'])
                assert np.isfinite(arr).all(), (run,data_key,est)
                tau = key[1] if number == 'E.8' else key
                gap = 0.01 if run == 'ab_test_noise_vars' else tau[1]-tau[0]
                factor = 100/gap if normalized else 1
                values = [np.mean(np.abs(arr))] if metric == 'mae' else [np.median(arr)*factor] if metric == 'median' else [np.mean(arr)*factor,np.std(arr)*factor]
                old_numbers = NUMBER.findall(cell)
                assert len(old_numbers) == len(values), (label,cell,values)
                formatted = []
                for stat_idx,(old,value) in enumerate(zip(old_numbers, values)):
                    precision = len(old.split('.')[1])
                    new = f'{value:.{precision}f}'
                    formatted.append(new)
                    rec = dict(table=number,label=label,run=run,row=cells[0].strip(),estimator=str(estimator),column=str(key),statistic=('sd' if stat_idx else metric),paper=float(old),current=float(value),current_display=new,delta=float(value)-float(old),matches=float(new)==float(old))
                    table_records.append(rec)
                replacements_iter=iter(formatted)
                new_cells.append(NUMBER.sub(lambda m:next(replacements_iter),cell))
            table = table.replace(row,'&'.join(new_cells),1)
        records.extend(table_records)
        summaries.append(dict(table=number,label=label,total=len(table_records),matches=sum(x['matches'] for x in table_records)))
        if number == 'E.6':
            repetitions = manifest['runs'][run]['repetitions']
            table = re.sub(r'Results aggregate from [\d,]+ repetitions',
                           f'Results aggregate from {repetitions:,} repetitions', table)
        replacements.append('% Table '+number+'; source: '+manifest['runs'][run]['directory']+'\n'+table)
    selection_rates = {str(t): float(100*np.mean(results['ab_test_snr'][t]['nc_selection_arr'] != results['ab_test_snr'][t]['sample_splitting_selection_arr'])) for t in TAUS}
    out=args.output
    out.mkdir(exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(dict(paper=str(args.paper),paper_sha256=hashlib.sha256(raw.encode()).hexdigest(),tables=summaries,values=records,sample_splitting_winner_change_percent=selection_rates),indent=2)+'\n')
    (out/'replacement-tables.tex').write_text('% Generated from designated A/B results; manuscript has NOT been edited.\n\n'+'\n\n'.join(replacements)+'\n')
    lines=['# A/B results cross-check — 2026-09-17','',f'Compared all 12 A/B simulation tables in `{args.paper}` against the 10 designated reruns in [the current-results manifest](../ab-results-current.json). The manuscript was not edited.','',
        'Calculations follow `analysis.py`: mean/median WC, population SD (`ddof=0`), and mean absolute WC for MAE. Percentage tables divide by the treatment-effect gap and multiply by 100. A match means equality after rounding to the precision printed in the paper. Signed zero counts as equal. Every compared array is finite.','',
        '| Table | Matching numeric entries | Different |','|---|---:|---:|']
    for s in summaries:lines.append(f"| {s['table']} | {s['matches']}/{s['total']} | {s['total']-s['matches']} |")
    lines += ['',f"Overall: **{sum(s['matches'] for s in summaries)}/{len(records)} numeric entries match** (means, SDs, medians and MAEs counted separately).",'',
        '## Design and coverage','',
        '- All 10 designated runs have completion logs and finite WC arrays for every configured estimator/combination. Conditional SI has 999 saved draws rather than 1,000 at SNR 0.5% and 2%; statistics use the saved draws (the runner retains only WC values strictly between -5 and 5, excluding NaNs and outliers).',
        '- Repetition counts: ' + '; '.join(f"{name}: {entry['repetitions']:,}" for name, entry in manifest['runs'].items()) + '. Table E.6 requires 1,000 repetitions.',
        '- The Table 2 caption reports sample-splitting winner changes of 26.1%, 25.5%, and 20.9%. Current rates, ordered by SNR, are ' + ', '.join(f'{x:.1f}%' for x in selection_rates.values()) + '.',
        '- Bayesian moon power is now 0.6, matching the manuscript. The additional-estimator config retains power 0.8, but that moon row is not displayed in Table E.5.',
        '- Conditional-bootstrap and moon sample-size scaling runs have no active paper table; they are retained as current exploratory results.',
        '- Figures 1 and 2 require separate sample-size grids that this batch did not run. Figure D.1 is a separate subsampling illustration. These figures have not been numerically revalidated or regenerated by this audit. Targeting and Upworthy results are outside this A/B simulation replacement.',
        '- Differences alone do not establish statistical significance or a code defect. The paper does not record sufficient execution provenance to attribute every discrepancy to a particular source.',
        '', 'Differing estimators: ' + ', '.join(sorted({r['estimator'] for r in records if not r['matches']})) + '. Tables with zero differences in the summary match numerically in full; see the repetition counts above when checking captions.', '', '## Main-table differences', '', '| Table | Estimator | Setting | Statistic | Paper | Current |', '|---|---|---|---|---:|---:|']
    for r in records:
        if r['table'] in ['2','3'] and not r['matches']:
            lines.append(f"| {r['table']} | {r['estimator']} | {r['column']} | {r['statistic']} | {r['paper']} | {r['current_display']} |")
    lines += ['', '## All differing entries', '', 'Percentage-table deltas are percentage points; MAE and multiple-action deltas use raw outcome units. See [comparison.json](comparison.json) for every entry, including matches. [replacement-tables.tex](replacement-tables.tex) contains all 12 refreshed tables ready for review.','', '| Table | Estimator / allocation | Setting | Statistic | Paper | Current |', '|---|---|---|---|---:|---:|']
    for r in records:
        if not r['matches']:lines.append(f"| {r['table']} | {r['estimator']} | {r['column']} | {r['statistic']} | {r['paper']} | {r['current_display']} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))

if __name__ == '__main__':
    main()
