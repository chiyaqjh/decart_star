"""Helpers for dataset-specific experiment result directories."""

import os
from typing import Optional


def resolve_results_dir(dataset: str, default_dir: str, scheme_name: str, results_dir: Optional[str] = None) -> str:
    """Return the effective results directory, preserving explicit overrides."""
    if results_dir:
        return results_dir
    if dataset == 'uci_har':
        return os.path.join('experiments', 'results', 'result_UCI', scheme_name)
    if dataset == 'mnist':
        return os.path.join('experiments', 'results', 'result_MNIST', scheme_name)
    return default_dir
