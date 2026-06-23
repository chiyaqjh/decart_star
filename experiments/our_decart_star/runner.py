# decart/experiments/our_decart_star/runner.py

import sys
import os
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  #  decart 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from experiments.datasets import get_dataset_spec, load_experiment_records
from experiments.models.model_loader import load_trained_experiment_model
from experiments.shared_synthetic import generate_synthetic_decision_tree, generate_synthetic_dot_model, generate_synthetic_records, generate_synthetic_shallow_mlp
from experiments.result_paths import resolve_results_dir
from experiments.our_decart_star.wrapper import DeCartStarExperimentWrapper


@dataclass
class ExperimentConfig:

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
            self.model_types = ['dot', 'decision_tree', 'neural_network']
        if self.num_queriers < 1:
            raise ValueError('num_queriers ?1')
        if self.dataset not in {'synthetic', 'mnist', 'uci_har'}:
            raise ValueError("dataset must be 'synthetic', 'mnist', or 'uci_har'")
        dataset_spec = get_dataset_spec(self.dataset)
        if dataset_spec is not None and self.record_dim != dataset_spec['input_dim']:
            raise ValueError(f"{self.dataset} experiments require record_dim={dataset_spec['input_dim']}")
        if self.model_source not in {'synthetic', 'trained'}:
            raise ValueError("model_source must be 'synthetic' or 'trained'")
        self.results_dir = resolve_results_dir(self.dataset, "experiments/results/our_decart_star", "our_decart_star", self.results_dir)


class ExperimentRunner:
    
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
                'comm_check_request_sizes': [],
                'comm_check_response_sizes': [],
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
    
    def generate_test_data(self, run_id: int = 0) -> Tuple[List[List[float]], List[int]]:
    
        if self.config.dataset != 'synthetic':
            print(f"\nLoading {self.config.dataset} test samples from {self.config.mnist_data_dir}...")
            data, labels = load_experiment_records(
                self.config.dataset,
                self.config.num_records,
                data_dir=self.config.mnist_data_dir,
                split='test',
            )
            print(f"   Loaded {len(data)} {self.config.dataset} records with dimension {len(data[0])}")
            return data, labels

        return generate_synthetic_records(self.config.num_records, self.config.record_dim, run_id)
    
    def generate_model(self, model_type: str, run_id: int = 0) -> Any:
    
        if self.config.model_source == 'trained':
            print(f"   Loading trained {model_type} model...")
            return load_trained_experiment_model(model_type, self.config.trained_models_dir, dataset_name=self.config.dataset)
        
        if model_type == 'dot':

            print(f"   ...")
            return generate_synthetic_dot_model(self.config.record_dim, run_id)
        
        elif model_type == 'decision_tree':
       
            print(f"   Generating decision tree model...")
            return generate_synthetic_decision_tree()
        
        elif model_type == 'neural_network':
    
            print(f"   ...")
            
            print("   NN random init scale: deterministic shared synthetic model")
            return generate_synthetic_shallow_mlp(self.config.record_dim, run_id)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def register_all_users(self, wrapper: DeCartStarExperimentWrapper, policy: List[int]):
        print(f"\nRegistering policy users...")
        for uid in policy:
            try:
                wrapper.register_user(uid)
                print(f"    Registered user {uid}")
            except Exception as e:
                print(f"    Failed to register user {uid}: {e}")
    
    def run_single_experiment(self, run_id: int, model_type: str) -> Optional[Dict]:
        print(f"\n{'='*60}")
        print(f"[DeCart*] Run {run_id + 1}/{self.config.num_runs} - model: {model_type}")
        print(f"{'='*60}")
        
        try:

            wrapper = DeCartStarExperimentWrapper(N=self.config.N, n=self.config.n)
            setup_time = wrapper.setup()
            setup_auxiliary_sizes = wrapper.get_auxiliary_sizes()
            wrapper.reset_metrics()
            
        
            owner_id = 5
            active_querier_id = owner_id + 1
            if active_querier_id >= self.config.N:
                raise ValueError(f" N={self.config.N} ")
            query_repetitions = self.config.num_queriers
            
            policy = list(range(min(self.config.policy_size, self.config.N - 2)))
            policy.append(owner_id)
            policy.append(active_querier_id)
            policy = list(set(policy)) 
            
            self.register_all_users(wrapper, policy)
            register_auxiliary_sizes = wrapper.get_auxiliary_sizes()

            wrapper.curator.add_trust(active_querier_id, owner_id)

            data, _ = self.generate_test_data(run_id)
 
            model = self.generate_model(model_type, run_id)
            
            print(f"\nEncrypting dataset...")
            C_m, sk_h_s, ds_id = wrapper.encrypt_dataset(owner_id, data, policy)
  
            wrapper.store_dataset(owner_id, ds_id, C_m, sk_h_s)

            prepared_model = wrapper.prepare_query_model(active_querier_id, model) if hasattr(wrapper, 'prepare_query_model') and model_type in {'dot', 'decision_tree', 'neural_network'} else None
   
            total_results = 0
            check_request_size_total = 0
            check_response_size_total = 0
            if model_type == 'dot':
    
                print(f"\nExecuting dot-product queries ({query_repetitions} repetitions, querier={active_querier_id})...")
                results = None
                for repetition_idx in range(query_repetitions):
                    query_index = repetition_idx + 1
                    print(f"   [Query {query_index}/{query_repetitions}] Dot-product query start, querier={active_querier_id}")
                    current_results = wrapper.execute_query(active_querier_id, owner_id, ds_id, model, prepared_model=prepared_model)
                    if current_results is not None:
                        results = current_results
                        total_results += len(current_results)
                        print(f"   [Query {query_index}/{query_repetitions}] Dot-product query done, results={len(current_results)}")
                    else:
                        print(f"   [Query {query_index}/{query_repetitions}] Dot-product query returned no results")
                
                query_time = float(np.sum(wrapper.metrics['query_times'])) if wrapper.metrics['query_times'] else 0
                decrypt_time = float(np.sum(wrapper.metrics['decrypt_times'])) if wrapper.metrics['decrypt_times'] else 0
                
            elif model_type == 'neural_network':
      
                print(f"\nExecuting neural-network queries ({query_repetitions} repetitions, querier={active_querier_id})...")
                results = None
                for repetition_idx in range(query_repetitions):
                    query_index = repetition_idx + 1
                    print(f"   [Query {query_index}/{query_repetitions}] Neural-network query start, querier={active_querier_id}")
                    current_results = wrapper.execute_query(active_querier_id, owner_id, ds_id, model, prepared_model=prepared_model)
                    if current_results is not None:
                        results = current_results
                        total_results += len(current_results)
                        print(f"   [Query {query_index}/{query_repetitions}] Neural-network query done, results={len(current_results)}")
                    else:
                        print(f"   [Query {query_index}/{query_repetitions}] Neural-network query returned no results")
                
                query_time = float(np.sum(wrapper.metrics['query_times'])) if wrapper.metrics['query_times'] else 0
                decrypt_time = float(np.sum(wrapper.metrics['decrypt_times'])) if wrapper.metrics['decrypt_times'] else 0
                
            elif model_type == 'decision_tree':
                print(f"\nExecuting decision-tree queries ({query_repetitions} repetitions, querier={active_querier_id})...")
                results = None
                for repetition_idx in range(query_repetitions):
                    query_index = repetition_idx + 1
                    print(f"   [Query {query_index}/{query_repetitions}] Decision-tree query start, querier={active_querier_id}")
                    current_results = wrapper.execute_query(active_querier_id, owner_id, ds_id, model, prepared_model=prepared_model)
                    if current_results is not None:
                        results = current_results
                        total_results += len(current_results)
                        print(f"   [Query {query_index}/{query_repetitions}] Decision-tree query done, results={len(current_results)}")
                    else:
                        print(f"   [Query {query_index}/{query_repetitions}] Decision-tree query returned no results")

                query_time = float(np.sum(wrapper.metrics['query_times'])) if wrapper.metrics['query_times'] else 0
                decrypt_time = float(np.sum(wrapper.metrics['decrypt_times'])) if wrapper.metrics['decrypt_times'] else 0
            
            else:
                print(f"      : {model_type}")
                return None

            phase_comm = {'upload': 0, 'check': 0, 'query': 0, 'decrypt': 0}
            for c in wrapper.metrics['communication_sizes']:
                if isinstance(c, dict):
                    t = c.get('type')
                    s = c.get('size', 0)
                    if t == 'encrypt':
                        phase_comm['upload'] += s
                    elif t == 'check':
                        phase_comm['check'] += s
                        check_request_size_total += c.get('request_size', 0)
                        check_response_size_total += c.get('response_size', 0)
                    elif t == 'query':
                        phase_comm['query'] += s
                    elif t == 'decrypt':
                        phase_comm['decrypt'] += s

            final_auxiliary_sizes = wrapper.get_auxiliary_sizes()
            keygen_times = wrapper.metrics['keygen_times']
            register_times = wrapper.metrics['register_times']
            check_times = wrapper.metrics['check_times']
            encrypt_times = wrapper.metrics['encrypt_times']
            communication_sizes = [
                s.get('size', 0) if isinstance(s, dict) else s
                for s in wrapper.metrics['communication_sizes']
            ]

            model_metrics = {
                'setup_time': setup_time,
                'keygen_times': keygen_times.copy(),
                'keygen_time': float(np.sum(keygen_times)) if keygen_times else 0,
                'register_times': register_times.copy(),
                'register_time': float(np.sum(register_times)) if register_times else 0,
                'check_time': float(np.sum(check_times)) if check_times else 0,
                'encrypt_times': encrypt_times.copy(),
                'encrypt_time': float(np.sum(encrypt_times)) if encrypt_times else 0,
                'query_time': query_time,
                'decrypt_time': decrypt_time,
                'setup_auxiliary_sizes': setup_auxiliary_sizes.copy(),
                'register_auxiliary_sizes': register_auxiliary_sizes.copy(),
                'final_auxiliary_sizes': final_auxiliary_sizes.copy(),
                'communication_sizes': [s.copy() if isinstance(s, dict) else s 
                                       for s in wrapper.metrics['communication_sizes']],
                'communication_size': float(np.sum(communication_sizes)) if communication_sizes else 0,
                'comm_upload_size': phase_comm['upload'],
                'comm_check_size': phase_comm['check'],
                'comm_check_request_size': check_request_size_total,
                'comm_check_response_size': check_response_size_total,
                'comm_query_size': phase_comm['query'],
                'comm_decrypt_size': phase_comm['decrypt'],
                'success': results is not None,
                'num_results': total_results
            }
            
            if results:
                print(f"\n   ! : {len(results)}")
            
            return model_metrics
            
        except Exception as e:
            print(f"\n   : {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self) -> Dict:
        print("\n" + "=" * 80)
        print("Starting DeCart* experiments")
        print("=" * 80)
        print(f"\nConfiguration:")
        for key, value in asdict(self.config).items():
            print(f"   {key}: {value}")
        
        for model_type in self.config.model_types:
            print(f"\n{'#' * 70}")
            print(f"Model type: {model_type}")
            print(f"{'#' * 70}")
            
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
                    model_results['comm_check_request_sizes'].append(run_result.get('comm_check_request_size', 0))
                    model_results['comm_check_response_sizes'].append(run_result.get('comm_check_response_size', 0))
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
                        'comm_check_size': run_result.get('comm_check_size', 0),
                        'comm_check_request_size': run_result.get('comm_check_request_size', 0),
                        'comm_check_response_size': run_result.get('comm_check_response_size', 0),
                        'setup_auxiliary_sizes': setup_auxiliary_sizes,
                        'register_auxiliary_sizes': register_auxiliary_sizes,
                        'final_auxiliary_sizes': final_auxiliary_sizes,
                        'success': run_result['success'],
                        'num_results': run_result['num_results']
                    })
                    
                    print(f"\n       {i+1} :  {run_result['query_time']*1000:.2f} ms")
                else:
                    print(f"\n       {i+1} ")
        
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
                stats['avg_communication_size'] = float(np.mean(sizes)) / 1024  # KB
                stats['std_communication_size'] = float(np.std(sizes)) / 1024
                stats['total_communication'] = float(np.sum(sizes)) / 1024  # KB
            if model_data['comm_check_request_sizes']:
                stats['avg_check_request_size_kb'] = float(np.mean(model_data['comm_check_request_sizes'])) / 1024
            if model_data['comm_check_response_sizes']:
                stats['avg_check_response_size_kb'] = float(np.mean(model_data['comm_check_response_sizes'])) / 1024
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
        
        print("DeCart* Result Summary")
        for model_type, stats in summary.items():
            print(f"\nModel: {model_type}")
            for key, value in stats.items():
                if 'time' in key:
                    print(f"   {key}: {value*1000:.2f} ms")
                elif 'size' in key:
                    print(f"   {key}: {value:.2f} KB")
                else:
                    print(f"   {key}: {value}")
    
    def save_results(self):
        """Persist the collected experiment results as JSON."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_filename = f"decart_star_multimodel_exp_{timestamp}.json"
        json_path = os.path.join(self.config.results_dir, json_filename)
        
        def convert_to_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json(v) for v in obj]
            else:
                return obj
        
        json_results = convert_to_json(self.results)
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nSaved results:")
        print(f"   JSON: {json_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the DeCart* experiment runner.")
    parser.add_argument("--N", type=int, default=Config.MAX_USERS, help="Maximum number of users")
    parser.add_argument("--n", type=int, default=Config.BLOCK_SIZE, help="Block size")
    parser.add_argument("--num-records", type=int, default=Config.EXPERIMENT_NUM_RECORDS, help="Number of records")
    parser.add_argument("--record-dim", type=int, default=Config.EXPERIMENT_RECORD_DIM, help="Record dimension")
    parser.add_argument("--dataset", choices=["synthetic", "mnist", "uci_har"], default="synthetic", help="Dataset source")
    parser.add_argument("--mnist-data-dir", type=str, default="data", help="Directory used to cache MNIST data")
    parser.add_argument("--model-source", choices=["synthetic", "trained"], default="synthetic", help="Model source")
    parser.add_argument("--trained-models-dir", type=str, default="experiments/models/trained", help="Directory containing trained model pickle files")
    parser.add_argument("--policy-size", type=int, default=Config.EXPERIMENT_POLICY_SIZE, help="Access policy size")
    parser.add_argument("--num-queriers", type=int, default=1, help="Number of queriers")
    parser.add_argument("--num-runs", type=int, default=Config.EXPERIMENT_NUM_RUNS, help="Number of repeated runs")
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=['dot', 'decision_tree', 'neural_network'],
        help="Model types to evaluate",
    )
    parser.add_argument("--results-dir", type=str, default=None, help="Directory for JSON results")
    parser.add_argument("--no-save", action="store_true", help="Skip saving JSON results")

    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(f"DeCart* Experiment (N={args.N}, n={args.n})")
    print("=" * 80)
    print("\nConfiguration")
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
