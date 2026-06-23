"""Collect pap_id-vs-n data for DeCart and DeCart*.

This script only runs setup, keygen, and register. It stores per-user pap_id
payloads together with n-specific timing and size statistics under each scheme's
data_new/other folder.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from entities.key_curator import KeyCurator
from schemes.decart import DeCartParams
from schemes.decart_star import DeCartStarParams


DEFAULT_N_VALUES = [16, 32, 64, 128, 256, 512]
DEFAULT_N = Config.MAX_USERS
DEFAULT_NUM_RECORDS = 10
DEFAULT_RECORD_DIM = 10
DEFAULT_POLICY_SIZE = 32
DEFAULT_OWNER_ID = 5
DEFAULT_QUERIER_OFFSET = 1


def build_policy(user_count: int, policy_size: int, owner_id: int, active_querier_id: int) -> List[int]:
    """Build the same policy-shaped registration set used by the experiment runners."""
    base_policy = list(range(min(policy_size, user_count - 2)))
    base_policy.append(owner_id)
    base_policy.append(active_querier_id)
    return sorted(set(base_policy))


def encode_pap_id(pap_id: List[Any]) -> Dict[str, Any]:
    """Serialize pap_id so the raw data can be restored later."""
    raw_bytes = pickle.dumps(pap_id, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "pap_id_len": len(pap_id),
        "pap_id_size_bytes": len(raw_bytes),
        "pap_id_size_kb": len(raw_bytes) / 1024.0,
        "pap_id_pickle_b64": base64.b64encode(raw_bytes).decode("ascii"),
    }


def make_scheme_curator(scheme: str, n_value: int, max_users: int) -> KeyCurator:
    """Create a scheme-specific curator with the requested n."""
    if scheme == "decart":
        params = DeCartParams(N=max_users, n=n_value)
    elif scheme == "decart_star":
        params = DeCartStarParams(N=max_users, n=n_value)
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")
    return KeyCurator(scheme=scheme, params=params)


def run_single_setting(scheme: str, n_value: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run setup, keygen, and register for one scheme/n combination."""
    curator = make_scheme_curator(scheme, n_value, config["N"])

    setup_start = time.perf_counter()
    curator.setup()
    setup_time = time.perf_counter() - setup_start

    owner_id = config["owner_id"]
    active_querier_id = owner_id + config["querier_offset"]
    policy = build_policy(config["N"], config["policy_size"], owner_id, active_querier_id)

    keygen_times: List[float] = []
    register_times: List[float] = []
    pap_records: List[Dict[str, Any]] = []

    for user_id in policy:
        keygen_start = time.perf_counter()
        _, pk_id, pap_id = curator.generate_user_key(user_id)
        keygen_time = time.perf_counter() - keygen_start

        register_start = time.perf_counter()
        registered = curator.register(user_id, pk_id, pap_id)
        register_time = time.perf_counter() - register_start

        keygen_times.append(keygen_time)
        register_times.append(register_time)

        pap_record = {
            "user_id": user_id,
            "registered": bool(registered),
            "u_id_prime": curator.user_id_prime.get(user_id),
        }
        pap_record.update(encode_pap_id(pap_id))
        pap_records.append(pap_record)

    registered_pap_sizes = [record["pap_id_size_bytes"] for record in pap_records if record["registered"]]
    pap_avg_size_bytes = float(sum(registered_pap_sizes) / len(registered_pap_sizes)) if registered_pap_sizes else 0.0
    pap_std_size_bytes = (
        float((sum((size - pap_avg_size_bytes) ** 2 for size in registered_pap_sizes) / len(registered_pap_sizes)) ** 0.5)
        if registered_pap_sizes
        else 0.0
    )

    return {
        "scheme": scheme,
        "n": n_value,
        "config": {
            "N": config["N"],
            "n": n_value,
            "num_records": config["num_records"],
            "record_dim": config["record_dim"],
            "policy_size": config["policy_size"],
            "owner_id": owner_id,
            "active_querier_id": active_querier_id,
            "policy": policy,
        },
        "summary": {
            "setup_time": setup_time,
            "keygen_time": float(sum(keygen_times)),
            "register_time": float(sum(register_times)),
            "avg_pap_id_size_bytes": pap_avg_size_bytes,
            "std_pap_id_size_bytes": pap_std_size_bytes,
            "avg_pap_id_size_kb": pap_avg_size_bytes / 1024.0,
            "std_pap_id_size_kb": pap_std_size_bytes / 1024.0,
        },
        "runs": [
            {
                "run_id": 0,
                "setup_time": setup_time,
                "keygen_times": keygen_times,
                "register_times": register_times,
                "pap_records": pap_records,
                "success": True,
                "policy_size": len(policy),
            }
        ],
    }


def results_dir_for_scheme(scheme: str) -> Path:
    """Return the destination data_new/other folder for one scheme."""
    scheme_dir = "our_decart" if scheme == "decart" else "our_decart_star"
    return PROJECT_ROOT / "experiments" / "results" / "data_new" / scheme_dir / "other"


def save_results(scheme: str, payload: Dict[str, Any]) -> Path:
    """Persist one scheme's results as JSON."""
    output_dir = results_dir_for_scheme(scheme)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = "decart" if scheme == "decart" else "decart_star"
    output_path = output_dir / f"{prefix}_pap_id_n_data_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect pap_id and n data for DeCart schemes.")
    parser.add_argument("--schemes", nargs="+", default=["decart", "decart_star"], choices=["decart", "decart_star"])
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--n-values", nargs="+", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--num-records", type=int, default=DEFAULT_NUM_RECORDS)
    parser.add_argument("--record-dim", type=int, default=DEFAULT_RECORD_DIM)
    parser.add_argument("--policy-size", type=int, default=DEFAULT_POLICY_SIZE)
    parser.add_argument("--owner-id", type=int, default=DEFAULT_OWNER_ID)
    parser.add_argument("--querier-offset", type=int, default=DEFAULT_QUERIER_OFFSET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = {
        "N": args.N,
        "num_records": args.num_records,
        "record_dim": args.record_dim,
        "policy_size": args.policy_size,
        "owner_id": args.owner_id,
        "querier_offset": args.querier_offset,
    }

    print("Starting pap_id/n collection")
    print(f"  N={args.N}, n-values={args.n_values}, policy_size={args.policy_size}")

    for scheme in args.schemes:
        scheme_payload = {
            "experiment": "pap_id_n_setup_keygen_register",
            "scheme": scheme,
            "config": {
                "N": args.N,
                "n_values": args.n_values,
                "num_records": args.num_records,
                "record_dim": args.record_dim,
                "policy_size": args.policy_size,
                "owner_id": args.owner_id,
                "querier_offset": args.querier_offset,
            },
            "results": [],
        }

        for n_value in args.n_values:
            print(f"\n[{scheme}] Running n={n_value}")
            scheme_payload["results"].append(run_single_setting(scheme, n_value, experiment_config))

        output_path = save_results(scheme, scheme_payload)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()