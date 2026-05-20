"""Backfill new-scheme experiment results from existing DeCart configurations."""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / 'experiments' / 'results'

DECART_FOLDER = 'our_decart'
TARGET_SCHEMES = {
    'secpq': PROJECT_ROOT / 'experiments' / 'secpq' / 'runner.py',
    'naive_ccs23': PROJECT_ROOT / 'experiments' / 'naive_ccs23' / 'runner.py',
}


def _load_config(path: Path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle).get('config', {})


def _has_nonempty_results(path: Path):
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    models = payload.get('models', {})
    if not isinstance(models, dict):
        return False

    for metrics in models.values():
        if not isinstance(metrics, dict):
            continue

        runs = metrics.get('runs')
        if isinstance(runs, list) and runs:
            return True

        for value in metrics.values():
            if isinstance(value, list) and value:
                return True

    summary = payload.get('summary', {})
    if isinstance(summary, dict):
        for value in summary.values():
            if isinstance(value, dict) and value:
                return True

    return False


def collect_decart_square_configs():
    configs = set()
    for path_str in glob.glob(str(RESULTS_ROOT / DECART_FOLDER / '*.json')):
        path = Path(path_str)
        try:
            config = _load_config(path)
        except Exception:
            continue

        num_records = config.get('num_records')
        record_dim = config.get('record_dim')
        policy_size = config.get('policy_size')
        num_runs = config.get('num_runs')
        if num_records is None or record_dim is None:
            continue
        if num_records != record_dim:
            continue
        if policy_size is None or num_runs is None:
            continue
        configs.add((num_records, record_dim, policy_size, num_runs))
    return sorted(configs)


def has_result(result_folder: str, config_key):
    num_records, record_dim, policy_size, num_runs = config_key
    for path_str in glob.glob(str(RESULTS_ROOT / result_folder / '*.json')):
        path = Path(path_str)
        try:
            config = _load_config(path)
            has_data = _has_nonempty_results(path)
        except Exception:
            continue
        if (
            config.get('num_records') == num_records
            and config.get('record_dim') == record_dim
            and config.get('policy_size') == policy_size
            and config.get('num_runs') == num_runs
            and has_data
        ):
            return True
    return False


def missing_configs_for_scheme(scheme: str, decart_configs):
    return [config for config in decart_configs if not has_result(scheme, config)]


def run_one(runner: Path, config_key, python_executable: str, dry_run: bool):
    num_records, record_dim, policy_size, num_runs = config_key
    cmd = [
        python_executable,
        str(runner),
        '--num-records', str(num_records),
        '--record-dim', str(record_dim),
        '--policy-size', str(policy_size),
        '--num-runs', str(num_runs),
    ]
    print('RUN:', ' '.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Backfill SecPQ and Naive CCS-2023 using DeCart configs.')
    parser.add_argument(
        '--schemes',
        nargs='+',
        choices=sorted(TARGET_SCHEMES.keys()),
        default=sorted(TARGET_SCHEMES.keys()),
        help='Target schemes to backfill.',
    )
    parser.add_argument(
        '--max-configs',
        type=int,
        default=None,
        help='Only run the first N missing configs per scheme after sorting.',
    )
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=None,
        help='Optional filter on num_records=record_dim sizes.',
    )
    parser.add_argument(
        '--policy-sizes',
        type=int,
        nargs='+',
        default=None,
        help='Optional filter on policy_size.',
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        nargs='+',
        default=None,
        help='Optional filter on num_runs.',
    )
    parser.add_argument('--dry-run', action='store_true', help='Print missing configs without running them.')
    return parser.parse_args()


def main():
    args = parse_args()
    decart_configs = collect_decart_square_configs()
    if args.sizes is not None:
        sizes = set(args.sizes)
        decart_configs = [config for config in decart_configs if config[0] in sizes]
    if args.policy_sizes is not None:
        policy_sizes = set(args.policy_sizes)
        decart_configs = [config for config in decart_configs if config[2] in policy_sizes]
    if args.num_runs is not None:
        num_runs = set(args.num_runs)
        decart_configs = [config for config in decart_configs if config[3] in num_runs]

    print('DeCart square configs:')
    for config in decart_configs:
        print('  ', config)

    for scheme in args.schemes:
        missing = missing_configs_for_scheme(scheme, decart_configs)
        if args.max_configs is not None:
            missing = missing[:args.max_configs]

        print(f'\nMissing for {scheme}: {len(missing)}')
        for config in missing:
            print('  ', config)

        for config in missing:
            run_one(TARGET_SCHEMES[scheme], config, sys.executable, args.dry_run)


if __name__ == '__main__':
    main()