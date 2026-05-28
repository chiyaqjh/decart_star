# decart/experiments/our_decart/runner.py
"""DeCart experiment runner."""

import sys
import os
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  #  decart 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#  wrapper __init__
from config import Config
from experiments.our_decart.wrapper import DeCartExperimentWrapper


@dataclass
class ExperimentConfig:
    """Configuration for a DeCart experiment run."""
    # 
    N: int = Config.MAX_USERS  # 
    n: int = Config.BLOCK_SIZE  # ?
    
    # 
    num_records: int = Config.EXPERIMENT_NUM_RECORDS  # ?
    record_dim: int = Config.EXPERIMENT_RECORD_DIM   # 
    
    # 
    model_types: List[str] = None  # 
    
    # 
    policy_size: int = Config.EXPERIMENT_POLICY_SIZE  # ?
    num_queriers: int = 1
    
    # 
    num_runs: int = Config.EXPERIMENT_NUM_RUNS      # 
    save_results: bool = True
    results_dir: str = "experiments/results/our_decart"
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['dot', 'decision_tree', 'neural_network']
        if self.num_queriers < 1:
            raise ValueError('num_queriers ?1')


class ExperimentRunner:
    """DeCart ()"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {
            'config': asdict(config),
            'models': {},  # ?
            'summary': {}
        }
        
        # ?
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
        
        # 
        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)
    
    def generate_test_data(self) -> Tuple[List[List[float]], List[int]]:
        """Generate normalized random records and an empty policy placeholder."""
        data = []
        for _ in range(self.config.num_records):
            record = np.random.randn(self.config.record_dim).tolist()
            #  [-1, 1] ?
            max_val = max(abs(min(record)), abs(max(record)))
            if max_val > 0:
                record = [x / max_val for x in record]
            data.append(record)
        
        return data, []
    
    def generate_model(self, model_type: str) -> Any:
        """Generate a synthetic model for the requested model type."""
        
        if model_type == 'dot':
            #  - ?
            print(f"   ...")
            model = np.random.randn(self.config.record_dim).tolist()
            # ?
            max_val = max(abs(min(model)), abs(max(model))) or 1
            model = [x / max_val for x in model]
            return model
        
        elif model_type == 'decision_tree':
            # ?- 
            print(f"   Generating decision tree model...")
            try:
                from schemes.ai_model import DecisionTreeHE, DecisionTreeNode
                tree = DecisionTreeHE()
                
                # 
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
                
                print(f"     Decision tree nodes: {len(tree.nodes)}")
                return tree
            except ImportError as e:
                print(f"     Falling back to dict decision tree format: {e}")
                # ?
                return {
                    'type': 'decision_tree',
                    'format': 'dict',
                    'root': 0,
                    'nodes': [
                        {'id': 0, 'feature': 0, 'threshold': 0.5, 'left': 1, 'right': 2, 'is_leaf': False},
                        {'id': 1, 'value': 0.0, 'is_leaf': True},
                        {'id': 2, 'value': 1.0, 'is_leaf': True}
                    ]
                }
        
        elif model_type == 'neural_network':
            #  - 
            print(f"   ...")
            
            # ?
            output_dim = 10
            input_dim = self.config.record_dim
            
            #  (output_dim x input_dim)
            weights_matrix = np.random.randn(output_dim, input_dim) * 0.1
            bias = np.random.randn(output_dim) * 0.1
            
            # 
            weights = weights_matrix.flatten().tolist()
            
            # 
            return {
                'type': 'neural_network',
                'format': 'single_layer',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'weights': weights,
                'bias': bias.tolist(),
                'weights_shape': (output_dim, input_dim),
                'bias_shape': (output_dim,)
            }
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def register_all_users(self, wrapper: DeCartExperimentWrapper, policy: List[int]):
        """Register every user in the supplied policy."""
        print(f"\nRegistering policy users...")
        for uid in policy:
            try:
                wrapper.register_user(uid)
                print(f"    Registered user {uid}")
            except Exception as e:
                print(f"    Failed to register user {uid}: {e}")
    
    def run_single_experiment(self, run_id: int, model_type: str) -> Optional[Dict]:
        print(f"\n{'='*60}")
        print(f"Run {run_id + 1}/{self.config.num_runs} - model: {model_type}")
        print(f"{'='*60}")
        
        try:
            # ?
            wrapper = DeCartExperimentWrapper(N=self.config.N, n=self.config.n)
            setup_time = wrapper.setup()
            setup_auxiliary_sizes = wrapper.get_auxiliary_sizes()
            wrapper.reset_metrics()
            
            # ID
            owner_id = 5
            active_querier_id = owner_id + 1
            if active_querier_id >= self.config.N:
                raise ValueError(f" N={self.config.N} ")
            query_repetitions = self.config.num_queriers
            
            # 
            policy = list(range(min(self.config.policy_size, self.config.N - 2)))
            policy.append(owner_id)
            policy.append(active_querier_id)
            policy = list(set(policy))  # 
            
            # ?
            self.register_all_users(wrapper, policy)
            register_auxiliary_sizes = wrapper.get_auxiliary_sizes()
            
            # 
            wrapper.curator.add_trust(active_querier_id, owner_id)
            
            # 
            data, _ = self.generate_test_data()
            
            # 
            model = self.generate_model(model_type)
            
            # ?
            print(f"\nEncrypting dataset...")
            C_m, sk_h_s, ds_id = wrapper.encrypt_dataset(owner_id, data, policy)
            
            # ?
            wrapper.store_dataset(owner_id, ds_id, C_m, sk_h_s)
            
            # 
            total_results = 0
            check_request_size_total = 0
            check_response_size_total = 0
            if model_type == 'dot':
                # 
                print(f"\nExecuting dot-product queries ({query_repetitions} repetitions, querier={active_querier_id})...")
                results = None
                for repetition_idx in range(query_repetitions):
                    print(f"   Dot-product query {repetition_idx + 1}/{query_repetitions}, querier={active_querier_id}")
                    current_results = wrapper.execute_query(active_querier_id, owner_id, ds_id, model)
                    if current_results is not None:
                        results = current_results
                        total_results += len(current_results)
                
                query_time = float(np.sum(wrapper.metrics['query_times'])) if wrapper.metrics['query_times'] else 0
                decrypt_time = float(np.sum(wrapper.metrics['decrypt_times'])) if wrapper.metrics['decrypt_times'] else 0
                
            elif model_type == 'neural_network':
                # 
                print(f"\nExecuting neural-network queries ({query_repetitions} repetitions, querier={active_querier_id})...")
                results = None
                for repetition_idx in range(query_repetitions):
                    print(f"   Neural-network query {repetition_idx + 1}/{query_repetitions}, querier={active_querier_id}")
                    current_results = wrapper.execute_query(active_querier_id, owner_id, ds_id, model)
                    if current_results is not None:
                        results = current_results
                        total_results += len(current_results)
                
                query_time = float(np.sum(wrapper.metrics['query_times'])) if wrapper.metrics['query_times'] else 0
                decrypt_time = float(np.sum(wrapper.metrics['decrypt_times'])) if wrapper.metrics['decrypt_times'] else 0
                
            elif model_type == 'decision_tree':
                # ?
                print(f"\nPreparing encrypted decision-tree query...")
                
                # 
                pk_h = wrapper.curator.system.he.public_key
                encrypt_model_start = time.perf_counter()
                
                # ?
                if hasattr(wrapper.curator.system, 'encrypt_decision_tree'):
                    print(f"   Encrypting decision-tree model...")
                    encrypted_model = wrapper.curator.system.encrypt_decision_tree(model, pk_h)
                else:
                    # 
                    print(f"   ...")
                    # ?
                    if hasattr(model, 'get_encryptable_params'):
                        params = model.get_encryptable_params()
                        # 
                        encrypted_internal = []
                        for node in params.get('internal_nodes', []):
                            encrypted_node = {
                                'node_id': node['node_id'],
                                'feature_idx': node['feature_idx'],
                                'threshold': wrapper.curator.system.he.encrypt([node['threshold']]),
                                'left': node['left'],
                                'right': node['right']
                            }
                            encrypted_internal.append(encrypted_node)
                        
                        # 
                        encrypted_leaves = []
                        for node in params.get('leaf_nodes', []):
                            encrypted_node = {
                                'node_id': node['node_id'],
                                'value': wrapper.curator.system.he.encrypt([node['value']])
                            }
                            encrypted_leaves.append(encrypted_node)
                        
                        encrypted_model = {
                            'type': 'decision_tree',
                            'internal_nodes': encrypted_internal,
                            'leaf_nodes': encrypted_leaves,
                            'root_id': params.get('root_id', 0),
                            'node_count': params.get('node_count', 0)
                        }
                        print(f"     Encrypted internal nodes: {len(encrypted_internal)}, leaves: {len(encrypted_leaves)}")
                    else:
                        # 
                        encrypted_model = {
                            'type': 'decision_tree',
                            'encrypted': True,
                            'nodes': model.get('nodes', []) if isinstance(model, dict) else []
                        }
                        print(f"     Using fallback encrypted decision-tree structure")
                wrapper.metrics['encrypt_times'].append(time.perf_counter() - encrypt_model_start)
                
                query_times = []
                decrypt_times = []
                results = None
                for repetition_idx in range(query_repetitions):
                    querier = wrapper.create_querier(active_querier_id)
                    check_start = time.perf_counter()
                    C_M = querier.check_access(C_m)
                    wrapper.metrics['check_times'].append(time.perf_counter() - check_start)
                    check_req_payload = {
                        'querier_id': active_querier_id,
                        'owner_id': owner_id,
                        'dataset_id': C_m.get('dataset_id') if isinstance(C_m, dict) else None,
                    }
                    check_req_size = wrapper._safe_obj_size(check_req_payload, fallback=32)
                    check_res_size = wrapper._safe_obj_size(C_M) if C_M is not None else 0
                    wrapper.metrics['communication_sizes'].append({
                        'type': 'check',
                        'size': check_req_size + check_res_size,
                        'request_size': check_req_size,
                        'response_size': check_res_size,
                        'records': 0
                    })

                    if C_M is None:
                        print(f"      Access denied for querier={active_querier_id}")
                        return None

                    C_M['encrypted_model'] = encrypted_model
                    C_M['model_type'] = 'decision_tree'
                    C_M['access_granted'] = True

                    req_payload = {
                        'querier_id': active_querier_id,
                        'owner_id': owner_id,
                        'dataset_id': ds_id,
                        'encrypted_model': C_M.get('encrypted_model'),
                        'model_type': C_M.get('model_type')
                    }
                    req_size = wrapper._safe_obj_size(req_payload)
                    wrapper.metrics['communication_sizes'].append({
                        'type': 'query',
                        'size': req_size,
                        'records': len(C_m.get('c6_i', [])) if isinstance(C_m, dict) else 0
                    })

                    if 'sk_h_u' not in C_M:
                        C_M['sk_h_u'] = b'demo_secret_key'

                    print(f"   Executing decision-tree query {repetition_idx + 1}/{query_repetitions}, querier={active_querier_id}")
                    start_query = time.perf_counter()
                    ER = wrapper.curator.system.query(C_M, C_m, sk_h_s)
                    single_query_time = time.perf_counter() - start_query
                    query_times.append(single_query_time)
                    print(f"   Query time: {single_query_time*1000:.2f} ms")

                    if ER is not None:
                        res_size = wrapper._safe_obj_size(ER)
                        wrapper.metrics['communication_sizes'].append({
                            'type': 'decrypt',
                            'size': res_size,
                            'records': len(C_m.get('c6_i', [])) if isinstance(C_m, dict) else 0
                        })

                        start_decrypt = time.perf_counter()
                        current_results = wrapper.curator.system.decrypt(C_M['sk_h_u'], ER)
                        single_decrypt_time = time.perf_counter() - start_decrypt
                        decrypt_times.append(single_decrypt_time)
                        if current_results is not None:
                            results = current_results
                            total_results += len(current_results)
                        print(f"   : {single_decrypt_time*1000:.2f} ms")
                        print(f"   : {len(current_results) if current_results else 0}")
                        if current_results:
                            print(f"   : {current_results[:3]}")
                    else:
                        print(f"      ")

                query_time = float(np.sum(query_times)) if query_times else 0
                decrypt_time = float(np.sum(decrypt_times)) if decrypt_times else 0
            
            else:
                print(f"      : {model_type}")
                return None
            
            # 
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
        """Execute all configured experiment runs and summarize the results."""
        print("\n" + "=" * 80)
        print("Starting DeCart experiments")
        print("=" * 80)
        print(f"\nConfiguration:")
        for key, value in asdict(self.config).items():
            print(f"   {key}: {value}")
        
        # ?
        for model_type in self.config.model_types:
            print(f"\n{'#' * 70}")
            print(f"Model type: {model_type}")
            print(f"{'#' * 70}")
            
            model_results = self.results['models'][model_type]
            
            for i in range(self.config.num_runs):
                run_result = self.run_single_experiment(i, model_type)
                
                if run_result:
                    # 
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
        
        # 
        self._compute_statistics()
        
        # 
        if self.config.save_results:
            self.save_results()
        
        return self.results
    
    def _compute_statistics(self):
        """Compute summary statistics from the collected run metrics."""
        summary = {}
        
        for model_type, model_data in self.results['models'].items():
            stats = {}

            # ?
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
            
            # 
            if model_data['encrypt_times']:
                times = model_data['encrypt_times']
                stats['avg_encrypt_time'] = float(np.mean(times))
                stats['std_encrypt_time'] = float(np.std(times))
                stats['min_encrypt_time'] = float(np.min(times))
                stats['max_encrypt_time'] = float(np.max(times))
            
            # 
            if model_data['query_times']:
                times = model_data['query_times']
                stats['avg_query_time'] = float(np.mean(times))
                stats['std_query_time'] = float(np.std(times))
                stats['min_query_time'] = float(np.min(times))
                stats['max_query_time'] = float(np.max(times))
            
            # 
            if model_data['decrypt_times']:
                times = model_data['decrypt_times']
                stats['avg_decrypt_time'] = float(np.mean(times))
                stats['std_decrypt_time'] = float(np.std(times))
                stats['min_decrypt_time'] = float(np.min(times))
                stats['max_decrypt_time'] = float(np.max(times))
            
            # 
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
        
        print("\n" + "=" * 80)
        print("DeCart Result Summary")
        print("=" * 80)
        
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
        json_filename = f"decart_multimodel_exp_{timestamp}.json"
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
    
    parser = argparse.ArgumentParser(description="Run the DeCart experiment runner.")
    parser.add_argument("--N", type=int, default=Config.MAX_USERS, help="Maximum number of users")
    parser.add_argument("--n", type=int, default=Config.BLOCK_SIZE, help="Block size")
    parser.add_argument("--num-records", type=int, default=Config.EXPERIMENT_NUM_RECORDS, help="Number of records")
    parser.add_argument("--record-dim", type=int, default=Config.EXPERIMENT_RECORD_DIM, help="Record dimension")
    parser.add_argument("--policy-size", type=int, default=Config.EXPERIMENT_POLICY_SIZE, help="Access policy size")
    parser.add_argument("--num-queriers", type=int, default=1, help="Number of queriers")
    parser.add_argument("--num-runs", type=int, default=Config.EXPERIMENT_NUM_RUNS, help="Number of repeated runs")
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=['dot', 'decision_tree', 'neural_network'],
        help="Model types to evaluate",
    )
    parser.add_argument("--results-dir", type=str, default="experiments/results/our_decart", help="Directory for JSON results")
    parser.add_argument("--no-save", action="store_true", help="Skip saving JSON results")

    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(f"DeCart Experiment (N={args.N}, n={args.n})")
    print("=" * 80)
    print("\nConfiguration")
    print(f"   N: {args.N}")
    print(f"   n: {args.n}")
    print(f"   num-records: {args.num_records}")
    print(f"   record-dim: {args.record_dim}")
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
        policy_size=args.policy_size,
        num_queriers=args.num_queriers,
        model_types=args.model_types,
        num_runs=args.num_runs,
        save_results=not args.no_save,
        results_dir=args.results_dir
    )

    runner = ExperimentRunner(config)
    runner.run()
    
    print("\n" + "=" * 80)
    print("DeCart Experiment Completed")
    print("=" * 80)
