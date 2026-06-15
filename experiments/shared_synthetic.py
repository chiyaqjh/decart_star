import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np


def _seed_from_parts(*parts: object) -> int:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


def _normalize_vector(values: np.ndarray) -> List[float]:
    max_val = float(np.max(np.abs(values))) if values.size else 0.0
    if max_val > 0:
        values = values / max_val
    return values.astype(float).tolist()


def generate_synthetic_records(num_records: int, record_dim: int, run_id: int) -> Tuple[List[List[float]], List[int]]:
    rng = np.random.default_rng(_seed_from_parts("records", run_id, num_records, record_dim))
    data: List[List[float]] = []
    for _ in range(num_records):
        record = rng.standard_normal(record_dim)
        data.append(_normalize_vector(record))
    return data, []


def generate_synthetic_dot_model(record_dim: int, run_id: int) -> List[float]:
    rng = np.random.default_rng(_seed_from_parts("dot", run_id, record_dim))
    model = rng.standard_normal(record_dim)
    return _normalize_vector(model)


def generate_synthetic_decision_tree() -> Dict[str, Any]:
    return {
        'type': 'decision_tree',
        'root': 0,
        'nodes': [
            {'id': 0, 'feature': 0, 'threshold': 0.5, 'left': 1, 'right': 2},
            {'id': 1, 'value': 0.0},
            {'id': 2, 'value': 1.0},
        ],
    }


def generate_synthetic_shallow_mlp(record_dim: int, run_id: int, hidden_dim: int = 16, output_dim: int = 10) -> Dict[str, Any]:
    rng = np.random.default_rng(_seed_from_parts("neural_network", run_id, record_dim, hidden_dim, output_dim))
    weight_scale = min(0.1, 1.0 / np.sqrt(max(1, record_dim)))
    hidden_weights = rng.standard_normal((hidden_dim, record_dim)) * weight_scale
    hidden_bias = rng.standard_normal(hidden_dim) * weight_scale
    output_weights = rng.standard_normal((output_dim, hidden_dim)) * weight_scale
    output_bias = rng.standard_normal(output_dim) * weight_scale
    return {
        'type': 'neural_network',
        'format': 'mlp_1hidden',
        'input_dim': record_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'layers': [
            {
                'layer_idx': 0,
                'layer_type': 'linear',
                'activation': 'square',
                'weights': hidden_weights.flatten().astype(float).tolist(),
                'bias': hidden_bias.astype(float).tolist(),
                'weights_shape': (hidden_dim, record_dim),
                'bias_shape': (hidden_dim,),
            },
            {
                'layer_idx': 1,
                'layer_type': 'linear',
                'activation': 'linear',
                'weights': output_weights.flatten().astype(float).tolist(),
                'bias': output_bias.astype(float).tolist(),
                'weights_shape': (output_dim, hidden_dim),
                'bias_shape': (output_dim,),
            },
        ],
    }