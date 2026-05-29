"""DeCart revoke experiment runner across multiple settings."""

from config import Config
from experiments.revoke.comparison import RevokeComparisonConfig, RevokeComparisonRunner


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run DeCart revoke experiments across multiple settings.")
    parser.add_argument("--N-values", nargs="+", type=int, default=[Config.MAX_USERS], help="One or more N values")
    parser.add_argument("--n-values", nargs="+", type=int, default=[Config.BLOCK_SIZE], help="One or more n values")
    parser.add_argument(
        "--num-records-values",
        nargs="+",
        type=int,
        default=[Config.EXPERIMENT_NUM_RECORDS],
        help="One or more record-count values",
    )
    parser.add_argument(
        "--record-dim-values",
        nargs="+",
        type=int,
        default=[Config.EXPERIMENT_RECORD_DIM],
        help="One or more record-dimension values",
    )
    parser.add_argument(
        "--policy-size-values",
        nargs="+",
        type=int,
        default=[Config.EXPERIMENT_POLICY_SIZE],
        help="One or more policy-size values",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["dot"],
        help="Model types to compare: dot decision_tree neural_network",
    )
    parser.add_argument("--num-runs", type=int, default=Config.EXPERIMENT_NUM_RUNS, help="Number of runs per setting")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/our_decart",
        help="Directory for JSON results",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip saving JSON results")

    args = parser.parse_args()
    config = RevokeComparisonConfig(
        scheme="decart",
        N_values=args.N_values,
        n_values=args.n_values,
        num_records_values=args.num_records_values,
        record_dim_values=args.record_dim_values,
        policy_size_values=args.policy_size_values,
        model_types=args.model_types,
        num_runs=args.num_runs,
        save_results=not args.no_save,
        results_dir=args.results_dir,
    )
    runner = RevokeComparisonRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
