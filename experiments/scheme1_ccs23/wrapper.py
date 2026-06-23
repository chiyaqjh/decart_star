# decart/experiments/scheme1_ccs23/wrapper.py
import sys
import os
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Reuse the core cryptographic module
from core.homomorphic import HomomorphicEncryption


class CCS23ExperimentWrapper:

    def __init__(self, N: int = 64, n: int = 16):
        
        self.N = N
        self.n = n
        
        # Initialize homomorphic encryption (for measurement only, not used in baseline logic)
        self.he = HomomorphicEncryption(poly_modulus_degree=32768)
        
        # Data storage (plaintext)
        self.datasets = {}  # owner_id -> {dataset_id -> {'data': data, 'policy': policy}}
        
        # User registry
        self.registered_users = set()
        
        # Experiment metrics
        self.metrics = {
            'setup_time': 0,
            'keygen_times': [],
                'register_times': [],
            'check_times': [],
            'encrypt_times': [],
            'query_times': [],
            'decrypt_times': [],
            'communication_sizes': [],
            'memory_sizes': []
        }

    def _safe_obj_size(self, obj: Any, fallback: int = 1024) -> int:
        seen_ids = set()

        def _measure(x: Any, depth: int = 0) -> int:
            if x is None:
                return 0

            obj_id = id(x)
            if obj_id in seen_ids:
                return 0

            if isinstance(x, (bytes, bytearray, memoryview)):
                return len(x)

            serializer = getattr(x, 'serialize', None)
            if callable(serializer):
                try:
                    ser = serializer()
                    if isinstance(ser, (bytes, bytearray, memoryview)):
                        return len(ser)
                except Exception:
                    pass

            try:
                return len(pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                pass

            if depth < 20 and isinstance(x, dict):
                seen_ids.add(obj_id)
                total = 0
                for k, v in x.items():
                    total += _measure(k, depth + 1)
                    total += _measure(v, depth + 1)
                seen_ids.discard(obj_id)
                return total

            if depth < 20 and isinstance(x, (list, tuple, set)):
                seen_ids.add(obj_id)
                total = sum(_measure(item, depth + 1) for item in x)
                seen_ids.discard(obj_id)
                return total

            try:
                return len(str(x).encode('utf-8'))
            except Exception:
                return fallback

        size = _measure(obj)
        return size if size > 0 else fallback

    def get_auxiliary_sizes(self) -> Dict[str, int]:
        return {
            'crs_size_bytes': 0,
            'pp_size_bytes': 0,
            'aux_size_bytes': 0,
            'total_auxiliary_size_bytes': 0
        }

    @staticmethod
    def _should_report_progress(current: int, total: int) -> bool:
        if total <= 10:
            return True
        interval = max(1, total // 5)
        return current == 1 or current == total or current % interval == 0

    @classmethod
    def _report_progress(cls, stage: str, current: int, total: int, progress_label: Optional[str] = None):
        if cls._should_report_progress(current, total):
            prefix = f"{progress_label} " if progress_label else ""
            print(f"      {prefix}{stage} progress: {current}/{total}")
    
    def setup(self) -> float:

        start = time.perf_counter()
        # CCS23 does not require complex cryptographic setup
        elapsed = time.perf_counter() - start
        self.metrics['setup_time'] = elapsed
        print(f"   CCS23 initialization completed: {elapsed:.4f}s")
        return elapsed
    
    def register_user(self, user_id: int) -> Tuple[int, Any]:

        self.metrics['keygen_times'].append(0.0)

        register_start = time.perf_counter()
        self.registered_users.add(user_id)

        register_elapsed = time.perf_counter() - register_start
        self.metrics['register_times'].append(register_elapsed)
        
        # CCS23 does not require real keys; return placeholders
        return user_id, f"dummy_pk_{user_id}"
    
    def encrypt_dataset(self, 
                       owner_id: int,
                       data: List[List[float]],
                       policy: List[int],
                       metadata: Optional[Dict] = None) -> Tuple[Dict, Any, str]:

        # Generate dataset ID
        import time
        timestamp = int(time.time() * 1000)
        dataset_id = f"ds_{owner_id}_{timestamp}"
        
        # Store plaintext data
        if owner_id not in self.datasets:
            self.datasets[owner_id] = {}
        
        self.datasets[owner_id][dataset_id] = {
            'data': data.copy(),
            'policy': policy.copy(),
            'metadata': metadata or {},
            'store_time': time.time()
        }
        
        # Plaintext baseline performs no encryption; fixed encryption time is 0
        elapsed = 0.0
        self.metrics['encrypt_times'].append(elapsed)
        
        # Measure data size (for communication-overhead comparison)
        import sys
        import pickle
        size = sys.getsizeof(pickle.dumps(data))
        
        self.metrics['communication_sizes'].append({
            'type': 'encrypt',
            'size': size,
            'records': len(data)
        })

        self._report_progress('Data storage', len(data), len(data))
        
        print(f"    CCS23 data storage: {elapsed*1000:.2f} ms (plaintext baseline), data size: {size/1024:.2f} KB")
        
        # Return metadata in a format-compatible structure
        C_m = {
            'type': 'ccs23_plain',
            'dataset_id': dataset_id,
            'owner_id': owner_id,
            'policy': policy,
            'num_records': len(data)
        }
        
        return C_m, None, dataset_id
    
    def store_dataset(self, owner_id: int, dataset_id: str, C_m: Dict, sk_h_s: Any):
        
        pass

    def prepare_query_model(self, querier_id: int, model: Any) -> Any:
        
        return model
    
    def execute_query(self,
                     querier_id: int,
                     owner_id: int,
                     dataset_id: str,
                     model: Any,
                     prepared_model: Any = None,
                     progress_label: Optional[str] = None) -> Optional[List[float]]:

        # Check whether dataset exists
        if owner_id not in self.datasets or dataset_id not in self.datasets[owner_id]:
            print(f"      Dataset does not exist")
            return None
        
        dataset_info = self.datasets[owner_id][dataset_id]
        data = dataset_info['data']
        policy = dataset_info['policy']

        check_start = time.perf_counter()
        _authorized = querier_id in policy
        self.metrics['check_times'].append(time.perf_counter() - check_start)

        # Query-request stage communication: unified accounting of packets sent to the server
        active_model = prepared_model if prepared_model is not None else model

        req_payload = {
            'querier_id': querier_id,
            'owner_id': owner_id,
            'dataset_id': dataset_id,
            'encrypted_model': active_model,
            'model_type': active_model.get('type', 'dot_product') if isinstance(active_model, dict) else 'dot_product'
        }
        req_size = self._safe_obj_size(req_payload)
        self.metrics['communication_sizes'].append({
            'type': 'query',
            'size': req_size,
            'records': len(data)
        })
        
        # CCS23 baseline does not enforce access control (direct access)
        
        # Execute plaintext query
        start_query = time.perf_counter()
        
        total_records = len(data)
        results = []
        if isinstance(active_model, list):
            # Dot-product model: y = model · x
            for index, record in enumerate(data, start=1):
                # Ensure dimensions match
                min_len = min(len(active_model), len(record))
                result = sum(active_model[i] * record[i] for i in range(min_len))
                results.append(result)
                self._report_progress('Query computation', index, total_records, progress_label)
        elif isinstance(active_model, dict) and active_model.get('type') == 'neural_network':
            for index, record in enumerate(data, start=1):
                values = [float(v) for v in record]
                layers = active_model.get('layers') or [
                    {
                        'weights': active_model.get('weights', []),
                        'bias': active_model.get('bias', []),
                        'weights_shape': (int(active_model.get('output_dim', 10) or 10), len(record)),
                        'activation': 'linear',
                    }
                ]

                for layer in layers:
                    weights = layer.get('weights', [])
                    bias = layer.get('bias', [])
                    weights_shape = tuple(layer.get('weights_shape', (int(active_model.get('output_dim', 10) or 10), len(values))))
                    output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else len(bias)
                    input_dim = int(weights_shape[1]) if len(weights_shape) > 1 else len(values)
                    record_vec = values[:input_dim]
                    if len(record_vec) < input_dim:
                        record_vec.extend([0.0] * (input_dim - len(record_vec)))

                    next_values = []
                    for i in range(output_dim):
                        start_idx = i * input_dim
                        row_weights = weights[start_idx:start_idx + input_dim] if len(weights) > start_idx else []
                        val = sum(float(w) * record_vec[j] for j, w in enumerate(row_weights) if j < input_dim)
                        if i < len(bias):
                            val += float(bias[i])
                        if layer.get('activation', 'linear') == 'square':
                            val = val * val
                        next_values.append(val)
                    values = next_values

                results.append(values[0] if values else 0.0)
                self._report_progress('Query computation', index, total_records, progress_label)
        elif isinstance(active_model, dict) and active_model.get('type') == 'decision_tree':
            # Simple decision tree using unified baseline logic
            # Rule: feature[0] <= 0.5 -> left leaf, otherwise right leaf
            nodes = active_model.get('nodes', [])
            node_map = {n.get('id'): n for n in nodes}
            root_id = active_model.get('root', 0)

            for index, record in enumerate(data, start=1):
                current_id = root_id
                depth = 0
                max_depth = 10

                while depth < max_depth and current_id in node_map:
                    node = node_map[current_id]

                    if 'value' in node:
                        results.append(float(node['value']))
                        break

                    feature_idx = int(node.get('feature', 0))
                    threshold = float(node.get('threshold', 0.0))
                    feature_val = float(record[feature_idx]) if feature_idx < len(record) else 0.0

                    if feature_val <= threshold:
                        current_id = node.get('left')
                    else:
                        current_id = node.get('right')
                    depth += 1
                else:
                    # Defensive fallback
                    results.append(0.0)
                self._report_progress('Query computation', index, total_records, progress_label)
        else:
            # Default handling
            for index, record in enumerate(data, start=1):
                results.append(float(np.random.randn()))
                self._report_progress('Query computation', index, total_records, progress_label)
        
        query_time = time.perf_counter() - start_query
        self.metrics['query_times'].append(query_time)

        # Response-packet stage communication: plaintext results returned
        res_size = self._safe_obj_size(results, fallback=max(1, len(results)) * 16)
        self.metrics['communication_sizes'].append({
            'type': 'decrypt',
            'size': res_size,
            'records': len(results)
        })
        
        # Decryption time is 0 in CCS23 baseline
        decrypt_time = 0
        self.metrics['decrypt_times'].append(decrypt_time)
        
        self._report_progress('Result return', len(results), len(results), progress_label)
        print(f"      Plaintext results returned: {len(results)}/{len(results)}")
        print(f"    CCS23 query time: {query_time*1000:.2f} ms, number of results: {len(results)}")
        
        return results
    
    def reset_metrics(self):

        self.metrics = {
            'setup_time': 0,
            'keygen_times': [],
            'register_times': [],
            'check_times': [],
            'encrypt_times': [],
            'query_times': [],
            'decrypt_times': [],
            'communication_sizes': [],
            'memory_sizes': []
        }
    
    def get_metrics(self) -> Dict:

        metrics = self.metrics.copy()
        
        # Compute averages
        if metrics['encrypt_times']:
            metrics['avg_encrypt_time'] = np.mean(metrics['encrypt_times'])
            metrics['std_encrypt_time'] = np.std(metrics['encrypt_times'])
        
        if metrics['query_times']:
            metrics['avg_query_time'] = np.mean(metrics['query_times'])
            metrics['std_query_time'] = np.std(metrics['query_times'])

        if metrics['check_times']:
            metrics['avg_check_time'] = np.mean(metrics['check_times'])
            metrics['std_check_time'] = np.std(metrics['check_times'])
        
        if metrics['decrypt_times']:
            metrics['avg_decrypt_time'] = np.mean(metrics['decrypt_times'])
            metrics['std_decrypt_time'] = np.std(metrics['decrypt_times'])
        
        # Communication size
        if metrics['communication_sizes']:
            sizes = [s['size'] for s in metrics['communication_sizes']]
            metrics['avg_communication_size'] = np.mean(sizes)
            metrics['total_communication'] = np.sum(sizes)
        
        return metrics