# decart/experiments/scheme2_server/wrapper.py
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


class ServerSchemeExperimentWrapper:

    def __init__(self, N: int = 64, n: int = 16):

        self.N = N
        self.n = n
        
        # Initialize homomorphic encryption (server owns keys)
        self.he = HomomorphicEncryption(poly_modulus_degree=32768)
        
        # Server key pair
        self.server_public_key = self.he.public_key
        self.server_secret_key = self.he.secret_key
        
        # Encrypted data storage
        self.encrypted_datasets = {}  # owner_id -> {dataset_id -> encrypted_data}
        
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
        # Server scheme only needs homomorphic-encryption initialization (already done in __init__)
        elapsed = time.perf_counter() - start
        self.metrics['setup_time'] = elapsed
        print(f"   Server scheme initialization completed: {elapsed:.4f}s")
        return elapsed
    
    def register_user(self, user_id: int) -> Tuple[int, Any]:

        self.metrics['keygen_times'].append(0.0)

        register_start = time.perf_counter()
        self.registered_users.add(user_id)

        register_elapsed = time.perf_counter() - register_start
        self.metrics['register_times'].append(register_elapsed)
        
        # Users have no keys; all encryption uses the server public key
        return user_id, None
    
    def encrypt_dataset(self, 
                       owner_id: int,
                       data: List[List[float]],
                       policy: List[int],
                       metadata: Optional[Dict] = None) -> Tuple[Dict, Any, str]:

        # Generate dataset ID
        import time
        timestamp = int(time.time() * 1000)
        dataset_id = f"ds_{owner_id}_{timestamp}"
        
        # Measure encryption time
        start = time.perf_counter()
        
        # Encrypt each data record
        encrypted_data = []
        total_records = len(data)
        for index, record in enumerate(data, start=1):
            encrypted_record = self.he.encrypt(record)
            encrypted_data.append(encrypted_record)
            self._report_progress('Dataset encryption', index, total_records)
        
        elapsed = time.perf_counter() - start
        self.metrics['encrypt_times'].append(elapsed)
        
        # Store encrypted data
        if owner_id not in self.encrypted_datasets:
            self.encrypted_datasets[owner_id] = {}
        
        self.encrypted_datasets[owner_id][dataset_id] = {
            'encrypted_data': encrypted_data,
            'policy': policy.copy(),
            'metadata': metadata or {},
            'store_time': time.time()
        }
        
        # Measure ciphertext size
        try:
            import pickle
            size = len(pickle.dumps(encrypted_data))
        except:
            size = len(data) * 1024 * 1024  # Estimated 1MB per record
        
        self.metrics['communication_sizes'].append({
            'type': 'encrypt',
            'size': size,
            'records': len(data)
        })
        
        print(f"     Server-scheme encryption: {elapsed*1000:.2f} ms, ciphertext size: {size/1024:.2f} KB")
        
        # Return metadata in a compatible format
        C_m = {
            'type': 'server_scheme',
            'dataset_id': dataset_id,
            'owner_id': owner_id,
            'policy': policy,
            'num_records': len(data)
        }
        
        return C_m, None, dataset_id
    
    def store_dataset(self, owner_id: int, dataset_id: str, C_m: Dict, sk_h_s: Any):

        pass
    
    def encrypt_model(self, model: Any) -> Any:

        if isinstance(model, list):
            # Dot-product model - encrypt list
            return self.he.encrypt(model)
        elif isinstance(model, dict) and model.get('type') == 'neural_network':
            encrypted_layers = []
            layers = model.get('layers') or [
                {
                    'weights': model.get('weights', []),
                    'bias': model.get('bias', []),
                    'weights_shape': (model.get('output_dim'), model.get('input_dim')),
                    'activation': 'linear',
                }
            ]

            total_units = 0
            for layer in layers:
                weights_shape = tuple(layer.get('weights_shape', (0, 0)))
                output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else 0
                total_units += output_dim
                if layer.get('bias', []):
                    total_units += 1

            processed_units = 0

            for layer in layers:
                weights = layer.get('weights', [])
                bias = layer.get('bias', [])
                weights_shape = tuple(layer.get('weights_shape', (0, 0)))
                output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else 0
                input_dim = int(weights_shape[1]) if len(weights_shape) > 1 else 0

                encrypted_weight_rows = []
                for row_idx in range(output_dim):
                    row_start = row_idx * input_dim
                    row_end = row_start + input_dim
                    weight_row = [float(value) for value in weights[row_start:row_end]]
                    if len(weight_row) < input_dim:
                        weight_row.extend([0.0] * (input_dim - len(weight_row)))
                    encrypted_weight_rows.append(self.he.encrypt(weight_row))
                    processed_units += 1
                    self._report_progress('Model encryption', processed_units, total_units)

                encrypted_bias_vector = None
                bias_values = [float(value) for value in bias]
                if bias_values:
                    encrypted_bias_vector = self.he.encrypt(bias_values)
                    processed_units += 1
                    self._report_progress('Model encryption', processed_units, total_units)

                encrypted_layers.append({
                    'layer_idx': layer.get('layer_idx', len(encrypted_layers)),
                    'layer_type': layer.get('layer_type', 'linear'),
                    'activation': layer.get('activation', 'linear'),
                    'weights_shape': layer.get('weights_shape'),
                    'bias_shape': tuple(layer.get('bias_shape', (len(bias_values),))),
                    'encrypted_weights': [],
                    'encrypted_bias': [],
                    'encrypted_weight_rows': encrypted_weight_rows,
                    'encrypted_bias_vector': encrypted_bias_vector,
                })

            return {
                'type': 'neural_network',
                'layers': encrypted_layers,
                'input_dim': model.get('input_dim'),
                'output_dim': model.get('output_dim')
            }
        else:
            # Default handling
            return self.he.encrypt([0.0])

    def prepare_query_model(self, querier_id: int, model: Any) -> Any:

        start_encrypt_model = time.perf_counter()
        print("   Preparing query model...")
        encrypted_model = self.encrypt_model(model)
        elapsed = time.perf_counter() - start_encrypt_model
        self.metrics['encrypt_times'].append(elapsed)
        print(f"   Query model preparation completed: {elapsed*1000:.2f} ms")
        return encrypted_model

    @staticmethod
    def _evaluate_neural_network_plain(model: Dict[str, Any], plain_record: List[float]) -> float:
        values = [float(v) for v in plain_record]
        layers = model.get('layers') or [
            {
                'weights': model.get('weights', []),
                'bias': model.get('bias', []),
                'weights_shape': (int(model.get('output_dim', 10) or 10), len(plain_record)),
                'activation': 'linear',
            }
        ]

        for layer in layers:
            weights = layer.get('weights', [])
            bias = layer.get('bias', [])
            weights_shape = tuple(layer.get('weights_shape', (int(model.get('output_dim', 10) or 10), len(values))))
            output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else len(bias)
            input_dim = int(weights_shape[1]) if len(weights_shape) > 1 else len(values)
            record_vec = values[:input_dim]
            if len(record_vec) < input_dim:
                record_vec.extend([0.0] * (input_dim - len(record_vec)))

            next_values = []
            for output_index in range(output_dim):
                start_idx = output_index * input_dim
                row_weights = weights[start_idx:start_idx + input_dim] if len(weights) > start_idx else []
                value = sum(float(weight) * record_vec[idx] for idx, weight in enumerate(row_weights) if idx < input_dim)
                if output_index < len(bias):
                    value += float(bias[output_index])
                if layer.get('activation', 'linear') == 'square':
                    value = value * value
                next_values.append(value)
            values = next_values
        return values[0] if values else 0.0
    
    def execute_query(self,
                     querier_id: int,
                     owner_id: int,
                     dataset_id: str,
                     model: Any,
                     prepared_model: Any = None,
                     progress_label: Optional[str] = None) -> Optional[List[float]]:

        # Check whether the dataset exists
        if owner_id not in self.encrypted_datasets or dataset_id not in self.encrypted_datasets[owner_id]:
            print(f"     Dataset does not exist")
            return None

        check_start = time.perf_counter()
        dataset_info = self.encrypted_datasets.get(owner_id, {}).get(dataset_id)
        self.metrics['check_times'].append(time.perf_counter() - check_start)
        self.metrics['communication_sizes'].append({
            'type': 'check',
            'size': 0,
            'records': 0,
        })

        if dataset_info is None:
            return None

        encrypted_data = dataset_info['encrypted_data']
        
        encrypted_model = prepared_model if prepared_model is not None else self.prepare_query_model(querier_id, model)

        # Query-request stage communication: unified accounting for packets sent to server
        req_payload = {
            'querier_id': querier_id,
            'owner_id': owner_id,
            'dataset_id': dataset_id,
            'encrypted_model': encrypted_model,
            'model_type': model.get('type', 'dot_product') if isinstance(model, dict) else 'dot_product'
        }
        req_size = self._safe_obj_size(req_payload)
        self.metrics['communication_sizes'].append({
            'type': 'query',
            'size': req_size,
            'records': len(encrypted_data)
        })
        
        # Execute homomorphic query
        start_query = time.perf_counter()
        
        total_records = len(encrypted_data)
        results = []
        if isinstance(model, list):
            # Dot-product model
            for index, enc_record in enumerate(encrypted_data, start=1):
                try:
                    # Homomorphic dot product
                    result = enc_record.dot(encrypted_model)
                    results.append(result)
                except:
                    results.append(self.he.encrypt([0.0]))
                self._report_progress('Query computation', index, total_records, progress_label)
        elif isinstance(model, dict) and model.get('type') == 'decision_tree':
            # Simple decision tree using unified baseline logic
            # Rule: feature[0] <= 0.5 -> left leaf, otherwise right leaf
            nodes = model.get('nodes', [])
            node_map = {n.get('id'): n for n in nodes}
            root_id = model.get('root', 0)

            for index, enc_record in enumerate(encrypted_data, start=1):
                try:
                    plain_record = self.he.decrypt(enc_record)
                    if not isinstance(plain_record, list):
                        plain_record = [float(plain_record)]

                    current_id = root_id
                    depth = 0
                    max_depth = 10
                    pred = 0.0

                    while depth < max_depth and current_id in node_map:
                        node = node_map[current_id]

                        if 'value' in node:
                            pred = float(node['value'])
                            break

                        feature_idx = int(node.get('feature', 0))
                        threshold = float(node.get('threshold', 0.0))
                        feature_val = float(plain_record[feature_idx]) if feature_idx < len(plain_record) else 0.0

                        if feature_val <= threshold:
                            current_id = node.get('left')
                        else:
                            current_id = node.get('right')
                        depth += 1

                    results.append(self.he.encrypt([pred]))
                except:
                    results.append(self.he.encrypt([0.0]))
                self._report_progress('Query computation', index, total_records, progress_label)
        elif isinstance(model, dict) and model.get('type') == 'neural_network':
            for index, enc_record in enumerate(encrypted_data, start=1):
                try:
                    plain_record = self.he.decrypt(enc_record)
                    if not isinstance(plain_record, list):
                        plain_record = [float(plain_record)]
                    pred = self._evaluate_neural_network_plain(model, plain_record)
                    results.append(self.he.encrypt([pred]))
                except:
                    results.append(self.he.encrypt([0.0]))
                self._report_progress('Query computation', index, total_records, progress_label)
        else:
            # Simplified fallback
            for index, enc_record in enumerate(encrypted_data, start=1):
                results.append(self.he.encrypt([0.0]))
                self._report_progress('Query computation', index, total_records, progress_label)
        
        query_time = time.perf_counter() - start_query
        self.metrics['query_times'].append(query_time)

        # Response-packet stage communication: encrypted result list returned by server
        res_size = self._safe_obj_size(results, fallback=max(1, len(results)) * 1024)
        self.metrics['communication_sizes'].append({
            'type': 'decrypt',
            'size': res_size,
            'records': len(results)
        })
        
        # Decrypt results (server holds private key)
        start_decrypt = time.perf_counter()
        decrypted_results = []
        total_results = len(results)
        for index, enc_result in enumerate(results, start=1):
            try:
                dec = self.he.decrypt(enc_result)
                if isinstance(dec, list):
                    decrypted_results.append(dec[0] if dec else 0.0)
                else:
                    decrypted_results.append(float(dec))
            except:
                decrypted_results.append(0.0)
            self._report_progress('Result decryption', index, total_results, progress_label)
        
        decrypt_time = time.perf_counter() - start_decrypt
        self.metrics['decrypt_times'].append(decrypt_time)
        
        print(f"      Server-scheme query: {query_time*1000:.2f} ms")
        if prepared_model is None:
            print(f"      Model encryption: {self.metrics['encrypt_times'][-1]*1000:.2f} ms")
        else:
            print(f"      Model encryption: 0.00 ms (reusing prepared model)")
        print(f"      Result decryption: {decrypt_time*1000:.2f} ms")
        
        return decrypted_results
    
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