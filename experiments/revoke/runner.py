
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from experiments.datasets import get_dataset_spec, load_experiment_records
from experiments.models.model_loader import load_trained_experiment_model
from experiments.result_paths import resolve_results_dir
from experiments.our_decart.wrapper import DeCartExperimentWrapper
from experiments.our_decart_star.wrapper import DeCartStarExperimentWrapper


@dataclass
class RevokeExperimentConfig:

    scheme: str = "decart"
    N: int = Config.MAX_USERS
    n: int = Config.BLOCK_SIZE
    num_records: int = Config.EXPERIMENT_NUM_RECORDS
    record_dim: int = Config.EXPERIMENT_RECORD_DIM
    dataset: str = "synthetic"
    mnist_data_dir: str = "data"
    model_source: str = "synthetic"
    trained_models_dir: str = "experiments/models/trained"
    policy_size: int = Config.EXPERIMENT_POLICY_SIZE
    model_type: str = "dot"
    num_runs: int = Config.EXPERIMENT_NUM_RUNS
    save_results: bool = True
    results_dir: Optional[str] = None

    def __post_init__(self):
        self.scheme = self.scheme.lower()
        self.model_type = self.model_type.lower()

        if self.scheme not in {"decart", "decart_star"}:
            raise ValueError("scheme must be 'decart' or 'decart_star'")

        if self.model_type not in {"dot", "decision_tree", "neural_network"}:
            raise ValueError("model_type must be dot, decision_tree, or neural_network")

        if self.N < 3:
            raise ValueError("N must be at least 3 for revoke experiments")

        if self.dataset not in {"synthetic", "mnist", "uci_har"}:
            raise ValueError("dataset must be 'synthetic', 'mnist', or 'uci_har'")
        dataset_spec = get_dataset_spec(self.dataset)
        if dataset_spec is not None and self.record_dim != dataset_spec['input_dim']:
            raise ValueError(f"{self.dataset} experiments require record_dim={dataset_spec['input_dim']}")
        if self.model_source not in {'synthetic', 'trained'}:
            raise ValueError("model_source must be 'synthetic' or 'trained'")

        minimum_policy_size = 3
        if self.policy_size < minimum_policy_size:
            self.policy_size = minimum_policy_size

        if self.results_dir is None:
            self.results_dir = resolve_results_dir(self.dataset, os.path.join("experiments", "results", "revoke", self.scheme), os.path.join("revoke", self.scheme), self.results_dir)


class RevokeExperimentRunner:
    WRAPPER_BY_SCHEME = {
        "decart": DeCartExperimentWrapper,
        "decart_star": DeCartStarExperimentWrapper,
    }

    def __init__(self, config: RevokeExperimentConfig):
        self.config = config
        self.wrapper_cls = self.WRAPPER_BY_SCHEME[config.scheme]
        self.results: Dict[str, Any] = {
            "config": asdict(config),
            "runs": [],
            "summary": {},
        }

        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)

    def generate_test_data(self) -> List[List[float]]:
        if self.config.dataset != 'synthetic':
            print(f"\nLoading {self.config.dataset} samples from {self.config.mnist_data_dir}...")
            data, _ = load_experiment_records(self.config.dataset, self.config.num_records, data_dir=self.config.mnist_data_dir)
            return data

        data = []
        for _ in range(self.config.num_records):
            record = np.random.randn(self.config.record_dim).tolist()
            max_val = max(abs(min(record)), abs(max(record))) or 1.0
            data.append([value / max_val for value in record])
        return data

    def generate_model(self) -> Any:
        if self.config.model_source == 'trained':
            print(f"   Loading trained {self.config.model_type} model...")
            return load_trained_experiment_model(self.config.model_type, self.config.trained_models_dir, dataset_name=self.config.dataset)

        if self.config.model_type == "dot":
            model = np.random.randn(self.config.record_dim).tolist()
            max_val = max(abs(min(model)), abs(max(model))) or 1.0
            return [value / max_val for value in model]

        if self.config.model_type == "decision_tree":
            try:
                from schemes.ai_model import DecisionTreeHE, DecisionTreeNode

                tree = DecisionTreeHE()
                root = DecisionTreeNode(0)
                root.feature_idx = 0
                root.threshold = 0.5
                root.left_child = 1
                root.right_child = 2
                tree.add_node(root)

                left = DecisionTreeNode(1, is_leaf=True)
                left.value = 0.0
                tree.add_node(left)

                right = DecisionTreeNode(2, is_leaf=True)
                right.value = 1.0
                tree.add_node(right)

                tree.set_root(0)
                return tree
            except ImportError:
                return {
                    "type": "decision_tree",
                    "format": "dict",
                    "root": 0,
                    "nodes": [
                        {"id": 0, "feature": 0, "threshold": 0.5, "left": 1, "right": 2, "is_leaf": False},
                        {"id": 1, "value": 0.0, "is_leaf": True},
                        {"id": 2, "value": 1.0, "is_leaf": True},
                    ],
                }

        output_dim = 10
        hidden_dim = 16
        input_dim = self.config.record_dim
        weight_scale = min(0.1, 1.0 / np.sqrt(max(1, input_dim)))
        hidden_weights = np.random.randn(hidden_dim, input_dim) * weight_scale
        hidden_bias = np.random.randn(hidden_dim) * weight_scale
        output_weights = np.random.randn(output_dim, hidden_dim) * weight_scale
        output_bias = np.random.randn(output_dim) * weight_scale

        return {
            "type": "neural_network",
            "format": "mlp_1hidden",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "layers": [
                {
                    "layer_idx": 0,
                    "layer_type": "linear",
                    "activation": "square",
                    "weights": hidden_weights.flatten().tolist(),
                    "bias": hidden_bias.tolist(),
                    "weights_shape": (hidden_dim, input_dim),
                    "bias_shape": (hidden_dim,),
                },
                {
                    "layer_idx": 1,
                    "layer_type": "linear",
                    "activation": "linear",
                    "weights": output_weights.flatten().tolist(),
                    "bias": output_bias.tolist(),
                    "weights_shape": (output_dim, hidden_dim),
                    "bias_shape": (output_dim,),
                },
            ],
        }

    def build_user_layout(self) -> Dict[str, Any]:
        if self.config.N >= 8:
            owner_id = 5
            survivor_id = 6
            revoked_id = 7
        else:
            owner_id = 0
            survivor_id = 1
            revoked_id = 2

        reserved = [owner_id, survivor_id, revoked_id]
        policy: List[int] = []
        for user_id in range(self.config.N):
            if len(policy) >= self.config.policy_size:
                break
            if user_id not in policy:
                policy.append(user_id)

        for user_id in reserved:
            if user_id not in policy:
                if len(policy) < self.config.policy_size:
                    policy.append(user_id)
                else:
                    policy[-1] = user_id

        deduped_policy = []
        for user_id in policy:
            if user_id not in deduped_policy:
                deduped_policy.append(user_id)

        for user_id in reserved:
            if user_id not in deduped_policy:
                deduped_policy.append(user_id)

        return {
            "owner_id": owner_id,
            "survivor_id": survivor_id,
            "revoked_id": revoked_id,
            "policy": sorted(set(deduped_policy)),
        }

    def register_policy_users(self, wrapper: Any, policy: List[int]) -> Dict[str, float]:
        for user_id in policy:
            wrapper.register_user(user_id)

        keygen_total = float(np.sum(wrapper.metrics["keygen_times"])) if wrapper.metrics["keygen_times"] else 0.0
        register_total = float(np.sum(wrapper.metrics["register_times"])) if wrapper.metrics["register_times"] else 0.0
        return {
            "keygen_time": keygen_total,
            "register_time": register_total,
        }

    def prepare_query_package(
        self,
        wrapper: Any,
        querier_id: int,
        owner_id: int,
        dataset_id: str,
        model: Any,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Prepare an encrypted query package without executing the server query."""
        metrics = {
            "check_time": 0.0,
            "encrypt_time": 0.0,
            "check_size": 0,
            "check_request_size": 0,
            "check_response_size": 0,
            "query_request_size": 0,
        }

        querier = wrapper.create_querier(querier_id)
        C_m, _ = wrapper.db_server.get_dataset(owner_id, dataset_id)
        if C_m is None:
            return None, metrics

        check_start = time.perf_counter()
        C_M = querier.check_access(C_m)
        metrics["check_time"] = time.perf_counter() - check_start

        check_req_payload = {
            "querier_id": querier_id,
            "owner_id": owner_id,
            "dataset_id": dataset_id,
        }
        metrics["check_request_size"] = wrapper._safe_obj_size(check_req_payload, fallback=32)
        metrics["check_response_size"] = wrapper._safe_obj_size(C_M) if C_M is not None else 0
        metrics["check_size"] = metrics["check_request_size"] + metrics["check_response_size"]

        if C_M is None:
            return None, metrics

        encrypt_start = time.perf_counter()
        if isinstance(model, list) or (isinstance(model, dict) and model.get("type") == "neural_network"):
            C_M = querier.encrypt_ai_model(model, C_M)
        elif isinstance(model, dict) and model.get("type") == "decision_tree":
            pk_h = wrapper.curator.system.he.public_key
            if hasattr(wrapper.curator.system, "encrypt_decision_tree"):
                encrypted_model = wrapper.curator.system.encrypt_decision_tree(model, pk_h)
            else:
                encrypted_model = {
                    "type": "decision_tree",
                    "encrypted": True,
                    "nodes": model.get("nodes", []),
                }
            C_M["encrypted_model"] = encrypted_model
            C_M["model_type"] = "decision_tree"
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        metrics["encrypt_time"] = time.perf_counter() - encrypt_start

        query_req_payload = {
            "querier_id": querier_id,
            "owner_id": owner_id,
            "dataset_id": dataset_id,
            "encrypted_model": C_M.get("encrypted_model"),
            "model_type": C_M.get("model_type"),
        }
        metrics["query_request_size"] = wrapper._safe_obj_size(query_req_payload)
        return C_M, metrics

    def execute_prepared_query(
        self,
        wrapper: Any,
        querier_id: int,
        owner_id: int,
        dataset_id: str,
        C_M: Dict[str, Any],
        prep_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = {
            **prep_metrics,
            "server_query_time": 0.0,
            "decrypt_time": 0.0,
            "response_size": 0,
            "success": False,
            "num_results": 0,
            "blocked_revoked_queries": wrapper.db_server.stats.get("blocked_revoked_queries", 0),
        }

        query_start = time.perf_counter()
        ER = wrapper.db_server.execute_query(querier_id, owner_id, dataset_id, C_M)
        result["server_query_time"] = time.perf_counter() - query_start
        result["blocked_revoked_queries"] = wrapper.db_server.stats.get("blocked_revoked_queries", 0)

        if ER is None:
            return result

        result["response_size"] = wrapper._safe_obj_size(ER)

        decrypt_start = time.perf_counter()
        query_results = wrapper.curator.system.decrypt(C_M["sk_h_u"], ER)
        result["decrypt_time"] = time.perf_counter() - decrypt_start
        result["success"] = query_results is not None
        result["num_results"] = len(query_results) if query_results is not None else 0
        return result

    def run_pipeline_query(
        self,
        wrapper: Any,
        querier_id: int,
        owner_id: int,
        dataset_id: str,
        model: Any,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        prepared_C_M, prep_metrics = self.prepare_query_package(
            wrapper,
            querier_id,
            owner_id,
            dataset_id,
            model,
        )
        if prepared_C_M is None:
            return {
                **prep_metrics,
                "server_query_time": 0.0,
                "decrypt_time": 0.0,
                "response_size": 0,
                "success": False,
                "num_results": 0,
                "blocked_revoked_queries": wrapper.db_server.stats.get("blocked_revoked_queries", 0),
                "blocked_at": "client",
            }, None

        result = self.execute_prepared_query(
            wrapper,
            querier_id,
            owner_id,
            dataset_id,
            prepared_C_M,
            prep_metrics,
        )
        result["blocked_at"] = None if result["success"] else "server"
        return result, prepared_C_M

    def run_single_experiment(self, run_id: int) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print(f"Revoke run {run_id + 1}/{self.config.num_runs} - {self.config.scheme} - {self.config.model_type}")
        print("=" * 80)

        wrapper = self.wrapper_cls(N=self.config.N, n=self.config.n)

        setup_start = time.perf_counter()
        wrapper.setup()
        setup_time = time.perf_counter() - setup_start
        setup_auxiliary_sizes = wrapper.get_auxiliary_sizes()
        wrapper.reset_metrics()

        layout = self.build_user_layout()
        owner_id = layout["owner_id"]
        survivor_id = layout["survivor_id"]
        revoked_id = layout["revoked_id"]
        policy = layout["policy"]

        print(f"Policy users: {policy}")
        registration_metrics = self.register_policy_users(wrapper, policy)
        register_auxiliary_sizes = wrapper.get_auxiliary_sizes()

        wrapper.curator.add_trust(survivor_id, owner_id)
        wrapper.curator.add_trust(revoked_id, owner_id)

        owner = wrapper.create_owner(owner_id)
        wrapper.create_querier(survivor_id)
        wrapper.create_querier(revoked_id)

        data = self.generate_test_data()
        model = self.generate_model()

        encrypt_start = time.perf_counter()
        C_m, sk_h_s, dataset_id = wrapper.encrypt_dataset(owner_id, data, policy)
        dataset_encrypt_time = time.perf_counter() - encrypt_start
        wrapper.store_dataset(owner_id, dataset_id, C_m, sk_h_s)

        pre_revoke_query, stale_query_package = self.run_pipeline_query(
            wrapper,
            revoked_id,
            owner_id,
            dataset_id,
            model,
        )

        pre_policy = owner.access_policies[dataset_id].copy()
        db_version_before = wrapper.db_server.datasets[owner_id][dataset_id].get("version", 1)

        revoke_start = time.perf_counter()
        revoke_success = wrapper.curator.revoke_user(revoked_id)
        revoke_time = time.perf_counter() - revoke_start

        update_start = time.perf_counter()
        updated_C_m = owner.update_dataset_after_revoke(dataset_id, [revoked_id])
        owner_update_time = time.perf_counter() - update_start

        db_sync_success = False
        db_sync_time = 0.0
        if updated_C_m is not None:
            updated_dataset = owner.encrypted_datasets[dataset_id]
            db_sync_start = time.perf_counter()
            db_sync_success = wrapper.db_server.update_dataset(
                owner_id,
                dataset_id,
                updated_dataset["C_m"],
                updated_dataset.get("sk_h_s"),
                metadata={
                    "updated_after_revoke": True,
                    "revoked_users": [revoked_id],
                },
            )
            db_sync_time = time.perf_counter() - db_sync_start
        elif owner.dataset_metadata.get(dataset_id, {}).get("invalid"):
            db_sync_start = time.perf_counter()
            db_sync_success = wrapper.db_server.mark_dataset_invalid(
                owner_id,
                dataset_id,
                owner.dataset_metadata[dataset_id].get("invalid_reason", "all_users_revoked"),
            )
            db_sync_time = time.perf_counter() - db_sync_start

        post_policy = owner.access_policies.get(dataset_id, []).copy()
        db_version_after = wrapper.db_server.datasets[owner_id][dataset_id].get("version", db_version_before)

        stale_revoked_query = {
            "success": False,
            "blocked_revoked_queries": wrapper.db_server.stats.get("blocked_revoked_queries", 0),
            "server_query_time": 0.0,
            "decrypt_time": 0.0,
            "response_size": 0,
        }
        if stale_query_package is not None:
            stale_revoked_query = self.execute_prepared_query(
                wrapper,
                revoked_id,
                owner_id,
                dataset_id,
                stale_query_package,
                {
                    "check_time": 0.0,
                    "encrypt_time": 0.0,
                    "check_size": 0,
                    "check_request_size": 0,
                    "check_response_size": 0,
                    "query_request_size": 0,
                },
            )

        fresh_revoked_query, _ = self.run_pipeline_query(
            wrapper,
            revoked_id,
            owner_id,
            dataset_id,
            model,
        )
        surviving_query, _ = self.run_pipeline_query(
            wrapper,
            survivor_id,
            owner_id,
            dataset_id,
            model,
        )

        final_auxiliary_sizes = wrapper.get_auxiliary_sizes()
        overall_success = (
            pre_revoke_query["success"]
            and revoke_success
            and db_sync_success
            and not stale_revoked_query["success"]
            and not fresh_revoked_query["success"]
            and surviving_query["success"]
            and revoked_id not in post_policy
        )

        return {
            "run_id": run_id,
            "scheme": self.config.scheme,
            "model_type": self.config.model_type,
            "owner_id": owner_id,
            "survivor_id": survivor_id,
            "revoked_id": revoked_id,
            "policy_before": pre_policy,
            "policy_after": post_policy,
            "removed_users": [user_id for user_id in pre_policy if user_id not in post_policy],
            "setup_time": setup_time,
            "keygen_time": registration_metrics["keygen_time"],
            "register_time": registration_metrics["register_time"],
            "dataset_encrypt_time": dataset_encrypt_time,
            "revoke_time": revoke_time,
            "owner_update_time": owner_update_time,
            "db_sync_time": db_sync_time,
            "revoke_success": revoke_success,
            "db_sync_success": db_sync_success,
            "db_version_before": db_version_before,
            "db_version_after": db_version_after,
            "pre_revoke_query": pre_revoke_query,
            "stale_revoked_query": stale_revoked_query,
            "fresh_revoked_query": fresh_revoked_query,
            "surviving_query": surviving_query,
            "setup_auxiliary_sizes": setup_auxiliary_sizes,
            "register_auxiliary_sizes": register_auxiliary_sizes,
            "final_auxiliary_sizes": final_auxiliary_sizes,
            "overall_success": overall_success,
        }

    def compute_summary(self):
        runs = self.results["runs"]
        if not runs:
            self.results["summary"] = {}
            return

        def mean(values: List[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        summary = {
            "avg_setup_time": mean([run["setup_time"] for run in runs]),
            "avg_keygen_time": mean([run["keygen_time"] for run in runs]),
            "avg_register_time": mean([run["register_time"] for run in runs]),
            "avg_dataset_encrypt_time": mean([run["dataset_encrypt_time"] for run in runs]),
            "avg_revoke_time": mean([run["revoke_time"] for run in runs]),
            "avg_owner_update_time": mean([run["owner_update_time"] for run in runs]),
            "avg_db_sync_time": mean([run["db_sync_time"] for run in runs]),
            "avg_pre_revoke_query_time": mean([run["pre_revoke_query"]["server_query_time"] for run in runs]),
            "avg_surviving_query_time": mean([run["surviving_query"]["server_query_time"] for run in runs]),
            "overall_success_rate": mean([1.0 if run["overall_success"] else 0.0 for run in runs]),
            "revoke_success_rate": mean([1.0 if run["revoke_success"] else 0.0 for run in runs]),
            "db_sync_success_rate": mean([1.0 if run["db_sync_success"] else 0.0 for run in runs]),
            "stale_revoked_block_rate": mean([1.0 if not run["stale_revoked_query"]["success"] else 0.0 for run in runs]),
            "fresh_revoked_block_rate": mean([1.0 if not run["fresh_revoked_query"]["success"] else 0.0 for run in runs]),
            "surviving_query_success_rate": mean([1.0 if run["surviving_query"]["success"] else 0.0 for run in runs]),
            "avg_removed_users": mean([len(run["removed_users"]) for run in runs]),
            "avg_db_version_increment": mean([run["db_version_after"] - run["db_version_before"] for run in runs]),
            "avg_blocked_revoked_queries": mean([
                run["stale_revoked_query"].get("blocked_revoked_queries", 0)
                for run in runs
            ]),
        }
        self.results["summary"] = summary

    def save_results(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.scheme}_revoke_{self.config.model_type}_{timestamp}.json"
        output_path = os.path.join(self.config.results_dir, filename)

        def convert_to_json(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {key: convert_to_json(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [convert_to_json(value) for value in obj]
            return obj

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(convert_to_json(self.results), file, indent=2)

        print(f"Saved revoke results: {output_path}")

    def run(self) -> Dict[str, Any]:

        for key, value in asdict(self.config).items():
            print(f"{key}: {value}")

        for run_id in range(self.config.num_runs):
            run_result = self.run_single_experiment(run_id)
            self.results["runs"].append(run_result)

            print("\nRun summary")
            print(f"overall_success: {run_result['overall_success']}")
            print(f"revoke_time: {run_result['revoke_time'] * 1000:.2f} ms")
            print(f"owner_update_time: {run_result['owner_update_time'] * 1000:.2f} ms")
            print(f"db_sync_time: {run_result['db_sync_time'] * 1000:.2f} ms")
            print(f"stale_revoked_blocked: {not run_result['stale_revoked_query']['success']}")
            print(f"fresh_revoked_blocked: {not run_result['fresh_revoked_query']['success']}")
            print(f"surviving_query_success: {run_result['surviving_query']['success']}")

        self.compute_summary()

        if self.config.save_results:
            self.save_results()

        return self.results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a standalone revoke experiment.")
    parser.add_argument("--scheme", choices=["decart", "decart_star"], default="decart", help="Scheme to evaluate")
    parser.add_argument("--N", type=int, default=Config.MAX_USERS, help="Maximum number of users")
    parser.add_argument("--n", type=int, default=Config.BLOCK_SIZE, help="Block size")
    parser.add_argument("--num-records", type=int, default=Config.EXPERIMENT_NUM_RECORDS, help="Number of records")
    parser.add_argument("--record-dim", type=int, default=Config.EXPERIMENT_RECORD_DIM, help="Record dimension")
    parser.add_argument("--dataset", choices=["synthetic", "mnist", "uci_har"], default="synthetic", help="Dataset source")
    parser.add_argument("--mnist-data-dir", type=str, default="data", help="Directory used to cache MNIST data")
    parser.add_argument("--model-source", choices=["synthetic", "trained"], default="synthetic", help="Model source")
    parser.add_argument("--trained-models-dir", type=str, default="experiments/models/trained", help="Directory containing trained model pickle files")
    parser.add_argument("--policy-size", type=int, default=Config.EXPERIMENT_POLICY_SIZE, help="Access policy size")
    parser.add_argument(
        "--model-type",
        choices=["dot", "decision_tree", "neural_network"],
        default="dot",
        help="Model used in the revoke experiment",
    )
    parser.add_argument("--num-runs", type=int, default=Config.EXPERIMENT_NUM_RUNS, help="Number of repeated runs")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory for JSON results")
    parser.add_argument("--no-save", action="store_true", help="Skip saving JSON results")

    args = parser.parse_args()
    config = RevokeExperimentConfig(
        scheme=args.scheme,
        N=args.N,
        n=args.n,
        num_records=args.num_records,
        record_dim=args.record_dim,
        dataset=args.dataset,
        mnist_data_dir=args.mnist_data_dir,
        model_source=args.model_source,
        trained_models_dir=args.trained_models_dir,
        policy_size=args.policy_size,
        model_type=args.model_type,
        num_runs=args.num_runs,
        save_results=not args.no_save,
        results_dir=args.results_dir,
    )
    runner = RevokeExperimentRunner(config)
    runner.run()
