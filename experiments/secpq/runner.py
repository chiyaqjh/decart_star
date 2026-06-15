# decart/experiments/secpq/runner.py
"""
SecPQ 方案实验运行器
作为独立对比方法接入现有实验框架。
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from experiments.datasets import get_dataset_spec, load_experiment_records
from experiments.models.model_loader import load_trained_experiment_model
from experiments.shared_synthetic import generate_synthetic_decision_tree, generate_synthetic_records
from experiments.result_paths import resolve_results_dir
from experiments.secpq.wrapper import SecPQExperimentWrapper


SUPPORTED_MODEL_TYPES = ('decision_tree',)


@dataclass
class ExperimentConfig:
    """实验配置"""
    N: int = Config.MAX_USERS
    n: int = Config.BLOCK_SIZE
    num_records: int = Config.EXPERIMENT_NUM_RECORDS
    record_dim: int = Config.EXPERIMENT_RECORD_DIM
    dataset: str = 'synthetic'
    mnist_data_dir: str = 'data'
    model_source: str = 'synthetic'
    trained_models_dir: str = 'experiments/models/trained'
    model_types: List[str] = None
    policy_size: int = Config.EXPERIMENT_POLICY_SIZE
    num_queriers: int = 1
    num_runs: int = Config.EXPERIMENT_NUM_RUNS
    save_results: bool = True
    results_dir: Optional[str] = None

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = list(SUPPORTED_MODEL_TYPES)

        unsupported = sorted(set(self.model_types) - set(SUPPORTED_MODEL_TYPES))
        if unsupported:
            raise ValueError(f"SecPQ 仅支持决策树模型，收到不支持的模型类型: {unsupported}")
        if self.num_queriers < 1:
            raise ValueError('num_queriers 必须至少为 1')
        if self.dataset not in {'synthetic', 'mnist', 'uci_har'}:
            raise ValueError("dataset must be 'synthetic', 'mnist', or 'uci_har'")
        dataset_spec = get_dataset_spec(self.dataset)
        if dataset_spec is not None and self.record_dim != dataset_spec['input_dim']:
            raise ValueError(f"{self.dataset} experiments require record_dim={dataset_spec['input_dim']}")
        if self.model_source not in {'synthetic', 'trained'}:
            raise ValueError("model_source must be 'synthetic' or 'trained'")
        self.results_dir = resolve_results_dir(self.dataset, "experiments/results/secpq", "secpq", self.results_dir)


class ExperimentRunner:
    """SecPQ 方案实验运行器"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {
            'config': asdict(config),
            'models': {},
            'summary': {}
        }

        for model_type in config.model_types:
            self.results['models'][model_type] = {
                'setup_times': [],
                'keygen_times': [],
                'register_times': [],
                'check_times': [],
                'encrypt_times': [],
                'query_times': [],
                'decrypt_times': [],
                'communication_sizes': [],
                'comm_upload_sizes': [],
                'comm_check_sizes': [],
                'comm_query_sizes': [],
                'comm_decrypt_sizes': [],
                'setup_crs_sizes': [],
                'setup_pp_sizes': [],
                'setup_aux_sizes': [],
                'setup_total_auxiliary_sizes': [],
                'register_crs_sizes': [],
                'register_pp_sizes': [],
                'register_aux_sizes': [],
                'register_total_auxiliary_sizes': [],
                'final_crs_sizes': [],
                'final_pp_sizes': [],
                'final_aux_sizes': [],
                'final_total_auxiliary_sizes': [],
                'runs': []
            }

        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)

    @staticmethod
    def _print_progress(label: str, current: int, total: int, width: int = 28):
        total = max(total, 1)
        current = min(max(current, 0), total)
        filled = int(width * current / total)
        bar = '#' * filled + '-' * (width - filled)
        end = '\n' if current >= total else '\r'
        print(f"{label} [{bar}] {current}/{total}", end=end, flush=True)

    def generate_test_data(self, run_id: int = 0) -> Tuple[List[List[float]], List[int]]:
        if self.config.dataset != 'synthetic':
            print(f"\n加载 {self.config.dataset} 样本: {self.config.mnist_data_dir}")
            return load_experiment_records(self.config.dataset, self.config.num_records, data_dir=self.config.mnist_data_dir)

        return generate_synthetic_records(self.config.num_records, self.config.record_dim, run_id)

    def generate_model(self, model_type: str, run_id: int = 0) -> Any:
        if self.config.model_source == 'trained':
            print(f"   加载训练好的 {model_type} 模型...")
            return load_trained_experiment_model(model_type, self.config.trained_models_dir, dataset_name=self.config.dataset)

        if model_type == 'decision_tree':
            return generate_synthetic_decision_tree()

        raise ValueError(f"SecPQ 仅支持 decision_tree，收到模型类型: {model_type}")

    def register_all_users(self, wrapper: SecPQExperimentWrapper, policy: List[int]):
        print("\n 注册用户...")
        total = len(policy)
        for index, uid in enumerate(policy, start=1):
            wrapper.register_user(uid)
            self._print_progress("   注册进度", index, total)

    def run_single_experiment(self, run_id: int, model_type: str) -> Optional[Dict]:
        print(f"\n{'='*60}")
        print(f" [SecPQ方案] 运行实验 {run_id+1}/{self.config.num_runs} - 模型: {model_type}")
        print(f"{'='*60}")

        try:
            wrapper = SecPQExperimentWrapper(N=self.config.N, n=self.config.n)
            setup_time = wrapper.setup()
            setup_auxiliary_sizes = wrapper.get_auxiliary_sizes()
            wrapper.reset_metrics()

            owner_id = 5
            active_querier_id = owner_id + 1
            if active_querier_id >= self.config.N:
                raise ValueError(f"当前 N={self.config.N} 无法分配查询用户")
            query_repetitions = self.config.num_queriers

            policy = list(range(min(self.config.policy_size, self.config.N - 2)))
            policy.append(owner_id)
            policy.append(active_querier_id)
            policy = list(set(policy))

            self.register_all_users(wrapper, policy)
            register_auxiliary_sizes = wrapper.get_auxiliary_sizes()

            data, _ = self.generate_test_data(run_id)
            model = self.generate_model(model_type, run_id)

            print("\n 加密数据集...")
            _, _, ds_id = wrapper.encrypt_dataset(owner_id, data, policy)

            prepared_model = wrapper.prepare_query_model(active_querier_id, model) if hasattr(wrapper, 'prepare_query_model') else None

            print(f"\n 执行查询 ({query_repetitions} 次重复查询, querier={active_querier_id})...")
            results = None
            total_results = 0
            for index in range(1, query_repetitions + 1):
                current_results = wrapper.execute_query(active_querier_id, owner_id, ds_id, model, prepared_model=prepared_model)
                if current_results is not None:
                    results = current_results
                    total_results += len(current_results)
                self._print_progress("   查询进度", index, query_repetitions)

            keygen_times = wrapper.metrics['keygen_times']
            register_times = wrapper.metrics['register_times']
            check_times = wrapper.metrics['check_times']
            encrypt_times = wrapper.metrics['encrypt_times']
            query_times = wrapper.metrics['query_times']
            decrypt_times = wrapper.metrics['decrypt_times']
            communication_sizes = [
                s.get('size', 0) if isinstance(s, dict) else s
                for s in wrapper.metrics['communication_sizes']
            ]

            phase_comm = {'upload': 0, 'check': 0, 'query': 0, 'decrypt': 0}
            for c in wrapper.metrics['communication_sizes']:
                if isinstance(c, dict):
                    t = c.get('type')
                    s = c.get('size', 0)
                    if t == 'encrypt':
                        phase_comm['upload'] += s
                    elif t == 'check':
                        phase_comm['check'] += s
                    elif t == 'query':
                        phase_comm['query'] += s
                    elif t == 'decrypt':
                        phase_comm['decrypt'] += s

            final_auxiliary_sizes = wrapper.get_auxiliary_sizes()

            model_metrics = {
                'setup_time': setup_time,
                'keygen_times': keygen_times.copy(),
                'keygen_time': float(np.sum(keygen_times)) if keygen_times else 0,
                'register_times': register_times.copy(),
                'register_time': float(np.sum(register_times)) if register_times else 0,
                'check_time': float(np.sum(check_times)) if check_times else 0,
                'encrypt_times': encrypt_times.copy(),
                'encrypt_time': float(np.sum(encrypt_times)) if encrypt_times else 0,
                'query_time': float(np.sum(query_times)) if query_times else 0,
                'decrypt_time': float(np.sum(decrypt_times)) if decrypt_times else 0,
                'setup_auxiliary_sizes': setup_auxiliary_sizes.copy(),
                'register_auxiliary_sizes': register_auxiliary_sizes.copy(),
                'final_auxiliary_sizes': final_auxiliary_sizes.copy(),
                'communication_sizes': [s.copy() if isinstance(s, dict) else s for s in wrapper.metrics['communication_sizes']],
                'communication_size': float(np.sum(communication_sizes)) if communication_sizes else 0,
                'comm_upload_size': phase_comm['upload'],
                'comm_check_size': phase_comm['check'],
                'comm_query_size': phase_comm['query'],
                'comm_decrypt_size': phase_comm['decrypt'],
                'success': results is not None,
                'num_results': total_results
            }

            if results:
                print(f"\n 查询成功! 结果数量: {len(results)}")
                print(f"   结果示例: {results[:3]}")

            return model_metrics
        except Exception as e:
            print(f"\n   实验运行失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run(self) -> Dict:
        print("\n" + "="*80)
        print(" 开始 SecPQ 方案实验")
        print("="*80)
        print("\n 实验配置:")
        for key, value in asdict(self.config).items():
            print(f"   {key}: {value}")

        for model_type in self.config.model_types:
            print(f"\n{'#'*70}")
            print(f" 测试模型类型: {model_type}")
            print(f"{'#'*70}")

            model_results = self.results['models'][model_type]

            for i in range(self.config.num_runs):
                run_result = self.run_single_experiment(i, model_type)
                if run_result:
                    model_results['setup_times'].append(run_result.get('setup_time', 0))
                    model_results['keygen_times'].append(run_result.get('keygen_time', 0))
                    model_results['register_times'].append(run_result.get('register_time', 0))
                    model_results['check_times'].append(run_result.get('check_time', 0))
                    model_results['encrypt_times'].append(run_result.get('encrypt_time', 0))
                    model_results['query_times'].append(run_result['query_time'])
                    model_results['decrypt_times'].append(run_result['decrypt_time'])
                    model_results['communication_sizes'].append(run_result.get('communication_size', 0))
                    model_results['comm_upload_sizes'].append(run_result.get('comm_upload_size', 0))
                    model_results['comm_check_sizes'].append(run_result.get('comm_check_size', 0))
                    model_results['comm_query_sizes'].append(run_result.get('comm_query_size', 0))
                    model_results['comm_decrypt_sizes'].append(run_result.get('comm_decrypt_size', 0))

                    setup_auxiliary_sizes = run_result.get('setup_auxiliary_sizes', {})
                    register_auxiliary_sizes = run_result.get('register_auxiliary_sizes', {})
                    final_auxiliary_sizes = run_result.get('final_auxiliary_sizes', {})
                    model_results['setup_crs_sizes'].append(setup_auxiliary_sizes.get('crs_size_bytes', 0))
                    model_results['setup_pp_sizes'].append(setup_auxiliary_sizes.get('pp_size_bytes', 0))
                    model_results['setup_aux_sizes'].append(setup_auxiliary_sizes.get('aux_size_bytes', 0))
                    model_results['setup_total_auxiliary_sizes'].append(setup_auxiliary_sizes.get('total_auxiliary_size_bytes', 0))
                    model_results['register_crs_sizes'].append(register_auxiliary_sizes.get('crs_size_bytes', 0))
                    model_results['register_pp_sizes'].append(register_auxiliary_sizes.get('pp_size_bytes', 0))
                    model_results['register_aux_sizes'].append(register_auxiliary_sizes.get('aux_size_bytes', 0))
                    model_results['register_total_auxiliary_sizes'].append(register_auxiliary_sizes.get('total_auxiliary_size_bytes', 0))
                    model_results['final_crs_sizes'].append(final_auxiliary_sizes.get('crs_size_bytes', 0))
                    model_results['final_pp_sizes'].append(final_auxiliary_sizes.get('pp_size_bytes', 0))
                    model_results['final_aux_sizes'].append(final_auxiliary_sizes.get('aux_size_bytes', 0))
                    model_results['final_total_auxiliary_sizes'].append(final_auxiliary_sizes.get('total_auxiliary_size_bytes', 0))

                    model_results['runs'].append({
                        'run_id': i,
                        'keygen_times': run_result.get('keygen_times', []),
                        'keygen_time': run_result.get('keygen_time', 0),
                        'register_times': run_result.get('register_times', []),
                        'register_time': run_result.get('register_time', 0),
                        'check_time': run_result.get('check_time', 0),
                        'encrypt_times': run_result.get('encrypt_times', []),
                        'encrypt_time': run_result.get('encrypt_time', 0),
                        'query_time': run_result['query_time'],
                        'decrypt_time': run_result['decrypt_time'],
                        'communication_size': run_result.get('communication_size', 0),
                        'setup_auxiliary_sizes': setup_auxiliary_sizes,
                        'register_auxiliary_sizes': register_auxiliary_sizes,
                        'final_auxiliary_sizes': final_auxiliary_sizes,
                        'success': run_result['success'],
                        'num_results': run_result['num_results']
                    })
                    self._print_progress(f"   {model_type} 运行进度", i + 1, self.config.num_runs)
                    print(f"\n     运行 {i+1} 完成: 查询时间 {run_result['query_time']*1000:.2f} ms")
                else:
                    self._print_progress(f"   {model_type} 运行进度", i + 1, self.config.num_runs)
                    print(f"\n     运行 {i+1} 失败")

        self._compute_statistics()
        if self.config.save_results:
            self.save_results()
        return self.results

    def _compute_statistics(self):
        summary = {}

        for model_type, model_data in self.results['models'].items():
            stats = {}

            if model_data['setup_times']:
                times = model_data['setup_times']
                stats['avg_setup_time'] = float(np.mean(times))
                stats['std_setup_time'] = float(np.std(times))
                stats['min_setup_time'] = float(np.min(times))
                stats['max_setup_time'] = float(np.max(times))

            if model_data['keygen_times']:
                times = model_data['keygen_times']
                stats['avg_keygen_time'] = float(np.mean(times))
                stats['std_keygen_time'] = float(np.std(times))
                stats['min_keygen_time'] = float(np.min(times))
                stats['max_keygen_time'] = float(np.max(times))

            if model_data['register_times']:
                times = model_data['register_times']
                stats['avg_register_time'] = float(np.mean(times))
                stats['std_register_time'] = float(np.std(times))
                stats['min_register_time'] = float(np.min(times))
                stats['max_register_time'] = float(np.max(times))

            if model_data['check_times']:
                times = model_data['check_times']
                stats['avg_check_time'] = float(np.mean(times))
                stats['std_check_time'] = float(np.std(times))
                stats['min_check_time'] = float(np.min(times))
                stats['max_check_time'] = float(np.max(times))

            if model_data['encrypt_times']:
                times = model_data['encrypt_times']
                stats['avg_encrypt_time'] = float(np.mean(times))
                stats['std_encrypt_time'] = float(np.std(times))
                stats['min_encrypt_time'] = float(np.min(times))
                stats['max_encrypt_time'] = float(np.max(times))

            if model_data['query_times']:
                times = model_data['query_times']
                stats['avg_query_time'] = float(np.mean(times))
                stats['std_query_time'] = float(np.std(times))
                stats['min_query_time'] = float(np.min(times))
                stats['max_query_time'] = float(np.max(times))

            if model_data['decrypt_times']:
                times = model_data['decrypt_times']
                stats['avg_decrypt_time'] = float(np.mean(times))
                stats['std_decrypt_time'] = float(np.std(times))
                stats['min_decrypt_time'] = float(np.min(times))
                stats['max_decrypt_time'] = float(np.max(times))

            if model_data['communication_sizes']:
                sizes = model_data['communication_sizes']
                stats['avg_communication_size'] = float(np.mean(sizes)) / 1024
                stats['std_communication_size'] = float(np.std(sizes)) / 1024
                stats['total_communication'] = float(np.sum(sizes)) / 1024

            auxiliary_size_fields = {
                'setup_crs_sizes': 'avg_setup_crs_size_kb',
                'setup_pp_sizes': 'avg_setup_pp_size_kb',
                'setup_aux_sizes': 'avg_setup_aux_size_kb',
                'setup_total_auxiliary_sizes': 'avg_setup_total_auxiliary_size_kb',
                'register_crs_sizes': 'avg_register_crs_size_kb',
                'register_pp_sizes': 'avg_register_pp_size_kb',
                'register_aux_sizes': 'avg_register_aux_size_kb',
                'register_total_auxiliary_sizes': 'avg_register_total_auxiliary_size_kb',
                'final_crs_sizes': 'avg_final_crs_size_kb',
                'final_pp_sizes': 'avg_final_pp_size_kb',
                'final_aux_sizes': 'avg_final_aux_size_kb',
                'final_total_auxiliary_sizes': 'avg_final_total_auxiliary_size_kb'
            }
            for field_name, summary_name in auxiliary_size_fields.items():
                if model_data[field_name]:
                    stats[summary_name] = float(np.mean(model_data[field_name])) / 1024

            summary[model_type] = stats

        self.results['summary'] = summary

        print("\n" + "="*80)
        print(" SecPQ 方案实验结果")
        print("="*80)
        for model_type, stats in summary.items():
            print(f"\n  模型类型: {model_type}")
            for key, value in stats.items():
                if 'time' in key:
                    print(f"   {key}: {value*1000:.2f} ms")
                elif 'size' in key:
                    print(f"   {key}: {value:.2f} KB")
                else:
                    print(f"   {key}: {value}")

    def save_results(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_filename = f"secpq_exp_{timestamp}.json"
        json_path = os.path.join(self.config.results_dir, json_filename)

        def convert_to_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_json(v) for v in obj]
            return obj

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_json(self.results), f, indent=2)

        print("\n SecPQ 方案实验结果已保存:")
        print(f"   JSON: {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SecPQ 方案实验")
    parser.add_argument("--N", type=int, default=Config.MAX_USERS, help="最大用户数")
    parser.add_argument("--n", type=int, default=Config.BLOCK_SIZE, help="每块用户数")
    parser.add_argument("--num-records", type=int, default=Config.EXPERIMENT_NUM_RECORDS, help="数据记录数")
    parser.add_argument("--record-dim", type=int, default=Config.EXPERIMENT_RECORD_DIM, help="记录维度")
    parser.add_argument("--dataset", choices=["synthetic", "mnist", "uci_har"], default="synthetic", help="数据集来源")
    parser.add_argument("--mnist-data-dir", type=str, default="data", help="MNIST 缓存目录")
    parser.add_argument("--model-source", choices=["synthetic", "trained"], default="synthetic", help="模型来源")
    parser.add_argument("--trained-models-dir", type=str, default="experiments/models/trained", help="训练模型目录")
    parser.add_argument("--policy-size", type=int, default=Config.EXPERIMENT_POLICY_SIZE, help="策略大小")
    parser.add_argument("--num-queriers", type=int, default=1, help="真实查询者数量")
    parser.add_argument("--num-runs", type=int, default=Config.EXPERIMENT_NUM_RUNS, help="重复运行次数")
    parser.add_argument("--model-types", nargs="+", default=list(SUPPORTED_MODEL_TYPES), help="模型类型列表，仅支持 decision_tree")
    parser.add_argument("--results-dir", type=str, default=None, help="结果保存目录")
    parser.add_argument("--no-save", action="store_true", help="不保存结果")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("  启动 SecPQ 方案实验")
    print("="*80)
    print("\n 命令行参数:")
    print(f"   N: {args.N}")
    print(f"   n: {args.n}")
    print(f"   num-records: {args.num_records}")
    print(f"   record-dim: {args.record_dim}")
    print(f"   dataset: {args.dataset}")
    print(f"   mnist-data-dir: {args.mnist_data_dir}")
    print(f"   model-source: {args.model_source}")
    print(f"   trained-models-dir: {args.trained_models_dir}")
    print(f"   policy-size: {args.policy_size}")
    print(f"   num-queriers: {args.num_queriers}")
    print(f"   num-runs: {args.num_runs}")
    print(f"   model-types: {args.model_types}")
    print(f"   results-dir: {args.results_dir}")
    print(f"   save-results: {not args.no_save}")

    config = ExperimentConfig(
        N=args.N,
        n=args.n,
        num_records=args.num_records,
        record_dim=args.record_dim,
        dataset=args.dataset,
        mnist_data_dir=args.mnist_data_dir,
        model_source=args.model_source,
        trained_models_dir=args.trained_models_dir,
        policy_size=args.policy_size,
        num_queriers=args.num_queriers,
        model_types=args.model_types,
        num_runs=args.num_runs,
        save_results=not args.no_save,
        results_dir=args.results_dir
    )

    runner = ExperimentRunner(config)
    runner.run()

    print("\n" + "="*80)
    print("  SecPQ 方案实验完成")
    print("="*80)
