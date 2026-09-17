import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=[('run_all_ab_tests.sh', 'ab_test'),
                       ('run_all_targeting_tests.sh', 'targeting')])
def batch(tmp_path, request):
    script_name, prefix = request.param
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    script = scripts / script_name
    shutil.copyfile(ROOT / 'scripts' / script_name, script)
    configs = tmp_path / 'configs'
    configs.mkdir()
    for suffix in ['a', 'b', 'c']:
        (configs / f'{prefix}_{suffix}.yaml').write_text('{}\n')
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_uv = fake_bin / 'uv'
    fake_uv.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$BATCH_CALL_LOG"\n'
        'printf "live experiment output\\n"\n'
        'exit "${BATCH_TEST_STATUS:-0}"\n'
    )
    fake_uv.chmod(0o755)
    log = tmp_path / 'calls.log'
    env = {**os.environ, 'PATH': str(fake_bin) + os.pathsep + os.environ['PATH'],
           'BATCH_CALL_LOG': str(log)}
    return script, env, log


def test_batch_reports_progress_and_summary_without_capturing_live_output(batch):
    script, env, log = batch
    result = subprocess.run(['bash', str(script)], env=env, cwd='/tmp',
                            text=True, capture_output=True, check=True)
    assert '[batch 1/3 | 0% complete] START' in result.stdout
    assert '[batch 3/3 | 100% complete] DONE' in result.stdout
    assert 'Batch summary: 3/3 completed; elapsed' in result.stdout
    assert result.stdout.count('live experiment output') == 3
    calls = log.read_text().splitlines()
    assert len(calls) == 3
    assert all(call.startswith('run python -u ') for call in calls)


@pytest.mark.parametrize('exit_code, label', [(23, 'FAILED'), (130, 'INTERRUPTED')])
def test_batch_reports_failure_and_stops(batch, exit_code, label):
    script, env, log = batch
    env['BATCH_TEST_STATUS'] = str(exit_code)
    result = subprocess.run(['bash', str(script)], env=env, cwd='/tmp',
                            text=True, capture_output=True)
    assert result.returncode == exit_code
    assert 'Batch summary: 0/3 completed' in result.stdout
    assert label in result.stdout
    assert '2 experiment(s) not started' in result.stdout
    assert len(log.read_text().splitlines()) == 1


def test_batch_dry_run_prints_commands_without_starting_experiments(batch):
    script, env, log = batch
    result = subprocess.run(['bash', str(script), '--dry-run'], env=env, cwd='/tmp',
                            text=True, capture_output=True, check=True)
    assert len(result.stdout.splitlines()) == 3
    assert 'Batch summary' not in result.stdout
    assert not log.exists()
