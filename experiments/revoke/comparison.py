"""Multi-setting revoke comparison runner shared by DeCart and DeCart*."""

import json
import os
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np

from config import Config
from experiments.revoke.runner import RevokeExperimentConfig, RevokeExperimentRunner


@dataclass
class RevokeComparisonConfig:
    """Configuration for comparing revoke behavior across settings."""

    scheme: str
    N_values: List[int]
    n_values: List[int]
    num_records_values: List[int]
    record_dim_values: List[int]
    policy_size_values: List[int]
    model_types: List[str]
    num_runs: int = Config.EXPERIMENT_NUM_RUNS
    save_results: bool = True
    results_dir: Optional[str] = None

    def __post_init__(self):
        self.scheme = self.scheme.lower()
        if self.scheme not in {"decart", "decart_star"}:
            raise ValueError("scheme must be 'decart' or 'decart_star'")

        self.N_values = self._normalize_int_list(self.N_values, Config.MAX_USERS)
        self.n_values = self._normalize_int_list(self.n_values, Config.BLOCK_SIZE)
        self.num_records_values = self._normalize_int_list(self.num_records_values, Config.EXPERIMENT_NUM_RECORDS)
        self.record_dim_values = self._normalize_int_list(self.record_dim_values, Config.EXPERIMENT_RECORD_DIM)
        self.policy_size_values = self._normalize_int_list(self.policy_size_values, Config.EXPERIMENT_POLICY_SIZE)
        self.model_types = self._normalize_model_types(self.model_types)

        if self.results_dir is None:
            scheme_dir = "our_decart" if self.scheme == "decart" else "our_decart_star"
            self.results_dir = os.path.join("experiments", "results", scheme_dir)

    @staticmethod
    def _normalize_int_list(values: Optional[List[int]], default: int) -> List[int]:
        normalized = values or [default]
        return [int(value) for value in normalized]

    @staticmethod
    def _normalize_model_types(model_types: Optional[List[str]]) -> List[str]:
        normalized = model_types or ["dot"]
        valid_model_types = {"dot", "decision_tree", "neural_network"}
        result = []
        for model_type in normalized:
            lowered = model_type.lower()
            if lowered not in valid_model_types:
                raise ValueError(f"Unsupported model type: {model_type}")
            result.append(lowered)
        return result


class RevokeComparisonRunner:
    """Runs revoke experiments across multiple settings for one scheme."""

    def __init__(self, config: RevokeComparisonConfig):
        self.config = config
        self.results: Dict[str, Any] = {
            "scheme": config.scheme,
            "config": asdict(config),
            "settings": [],
            "comparison_summary": [],
        }

        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)

    def iter_settings(self):
        """Yield every requested setting combination."""
        for N, n, num_records, record_dim, policy_size, model_type in product(
            self.config.N_values,
            self.config.n_values,
            self.config.num_records_values,
            self.config.record_dim_values,
            self.config.policy_size_values,
            self.config.model_types,
        ):
            yield {
                "N": N,
                "n": n,
                "num_records": num_records,
                "record_dim": record_dim,
                "policy_size": policy_size,
                "model_type": model_type,
            }

    @staticmethod
    def build_setting_id(setting: Dict[str, Any]) -> str:
        """Create a stable label for one setting."""
        return (
            f"N{setting['N']}_n{setting['n']}_records{setting['num_records']}_"
            f"dim{setting['record_dim']}_policy{setting['policy_size']}_{setting['model_type']}"
        )

    def run_setting(self, setting_index: int, total_settings: int, setting: Dict[str, Any]) -> Dict[str, Any]:
        """Run one comparison point."""
        setting_id = self.build_setting_id(setting)
        print("\n" + "#" * 80)
        print(f"Setting {setting_index + 1}/{total_settings}: {setting_id}")
        print("#" * 80)

        setting_config = RevokeExperimentConfig(
            scheme=self.config.scheme,
            N=setting["N"],
            n=setting["n"],
            num_records=setting["num_records"],
            record_dim=setting["record_dim"],
            policy_size=setting["policy_size"],
            model_type=setting["model_type"],
            num_runs=self.config.num_runs,
            save_results=False,
            results_dir=self.config.results_dir,
        )
        runner = RevokeExperimentRunner(setting_config)
        setting_result = runner.run()

        return {
            "setting_id": setting_id,
            "setting": setting,
            "runs": setting_result.get("runs", []),
            "summary": setting_result.get("summary", {}),
        }

    def build_comparison_summary(self):
        """Flatten per-setting summaries for later plotting and analysis."""
        comparison_summary = []
        for setting_result in self.results["settings"]:
            setting = setting_result["setting"]
            summary = setting_result.get("summary", {})
            comparison_summary.append({
                "setting_id": setting_result["setting_id"],
                "N": setting["N"],
                "n": setting["n"],
                "num_records": setting["num_records"],
                "record_dim": setting["record_dim"],
                "policy_size": setting["policy_size"],
                "model_type": setting["model_type"],
                "avg_revoke_time": summary.get("avg_revoke_time", 0.0),
                "avg_owner_update_time": summary.get("avg_owner_update_time", 0.0),
                "avg_db_sync_time": summary.get("avg_db_sync_time", 0.0),
                "avg_pre_revoke_query_time": summary.get("avg_pre_revoke_query_time", 0.0),
                "avg_surviving_query_time": summary.get("avg_surviving_query_time", 0.0),
                "overall_success_rate": summary.get("overall_success_rate", 0.0),
                "stale_revoked_block_rate": summary.get("stale_revoked_block_rate", 0.0),
                "fresh_revoked_block_rate": summary.get("fresh_revoked_block_rate", 0.0),
                "surviving_query_success_rate": summary.get("surviving_query_success_rate", 0.0),
                "avg_db_version_increment": summary.get("avg_db_version_increment", 0.0),
            })
        self.results["comparison_summary"] = comparison_summary

    def save_results(self):
        """Persist the comparison output into the scheme's original results folder."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_prefix = "decart" if self.config.scheme == "decart" else "decart_star"
        output_path = os.path.join(self.config.results_dir, f"{file_prefix}_revoke_comparison_{timestamp}.json")

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

        print(f"Saved revoke comparison results: {output_path}")

    def run(self) -> Dict[str, Any]:
        """Execute all requested setting comparisons."""
        settings = list(self.iter_settings())
        print("\n" + "=" * 80)
        print(f"Starting revoke comparison for {self.config.scheme}")
        print("=" * 80)
        print(f"Total settings: {len(settings)}")

        for setting_index, setting in enumerate(settings):
            setting_result = self.run_setting(setting_index, len(settings), setting)
            self.results["settings"].append(setting_result)

        self.build_comparison_summary()

        if self.config.save_results:
            self.save_results()

        return self.results
