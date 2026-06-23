# decart/experiments/our_decart/owner.py

import sys
import os
import time
import pickle
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from entities.data_owner import DataOwner as BaseDataOwner


class DataOwner(BaseDataOwner):
    
    def __init__(self, owner_id: int, key_curator, scheme: str = "decart"):
        super().__init__(owner_id, key_curator, scheme)
        
        # Experiment measurement metrics
        self.experiment_metrics = {
            'encrypt_times': [],
            'policy_sizes': [],
            'data_sizes': [],
            'model_encrypt_times': [],
            'communication_overhead': []
        }
    
    def encrypt_data_with_metrics(self,
                                 data_records: List[List[float]],
                                 access_policy: List[int],
                                 metadata: Optional[Dict] = None,
                                 store_original: bool = False) -> Tuple[Dict, Any, str, Dict]:
        # Record input sizes
        data_size = sum(len(record) for record in data_records) * 8  # float: 8 bytes
        policy_size = len(access_policy) * 4  # int: 4 bytes
        
        # Measure encryption time
        start = time.perf_counter()
        C_m, sk_h_s, ds_id = self.encrypt_data(
            data_records, access_policy, metadata, store_original=store_original
        )
        encrypt_time = time.perf_counter() - start
        
        # Measure ciphertext size
        import sys
        import pickle
        cipher_size = sys.getsizeof(pickle.dumps(C_m))
        
        # Record metrics
        metrics = {
            'encrypt_time': encrypt_time,
            'data_size': data_size,
            'policy_size': policy_size,
            'cipher_size': cipher_size,
            'num_records': len(data_records),
            'record_dim': len(data_records[0]) if data_records else 0,
            'policy_length': len(access_policy)
        }
        
        self.experiment_metrics['encrypt_times'].append(encrypt_time)
        self.experiment_metrics['policy_sizes'].append(policy_size)
        self.experiment_metrics['data_sizes'].append(data_size)
        self.experiment_metrics['communication_overhead'].append(cipher_size)
        
        print(f"\n [DataOwner {self.owner_id}] Encryption metrics:")
        print(f"   Encryption time: {encrypt_time*1000:.2f} ms")
        print(f"   Data size: {data_size/1024:.2f} KB")
        print(f"   Ciphertext size: {cipher_size/1024:.2f} KB")
        print(f"   Expansion ratio: {cipher_size/data_size:.2f}x")
        
        return C_m, sk_h_s, ds_id, metrics
    
    def encrypt_model_with_metrics(self,
                                  model_id: str,
                                  access_policy: List[int]) -> Tuple[Dict, str, Dict]:

        # Get model information
        model_info = self.model_metadata.get(model_id, {})
        architecture = model_info.get('architecture', {})
        
        # Calculate model size
        if 'weights' in architecture:
            model_size = len(architecture['weights']) * 8  # float: 8 bytes
        elif 'combined_weights' in architecture:
            model_size = len(architecture['combined_weights']) * 8
        else:
            model_size = 0
        
        # Measure encryption time
        start = time.perf_counter()
        encrypted_model, enc_id = self.encrypt_model(model_id, access_policy)
        encrypt_time = time.perf_counter() - start
        
        # Measure ciphertext size
        import sys
        import pickle
        cipher_size = sys.getsizeof(pickle.dumps(encrypted_model))
        
        metrics = {
            'encrypt_time': encrypt_time,
            'model_size': model_size,
            'cipher_size': cipher_size,
            'policy_length': len(access_policy),
            'model_type': model_info.get('model_type', 'unknown'),
            'encrypt_time_ms': encrypt_time * 1000
        }
        
        self.experiment_metrics['model_encrypt_times'].append(encrypt_time)
        
        print(f"\n [DataOwner {self.owner_id}] Model encryption metrics:")
        print(f"   Encryption time: {encrypt_time*1000:.2f} ms")
        print(f"   Model size: {model_size/1024:.2f} KB")
        print(f"   Ciphertext size: {cipher_size/1024:.2f} KB")
        print(f"   Expansion ratio: {cipher_size/model_size:.2f}x")
        
        return encrypted_model, enc_id, metrics
    
    def get_experiment_metrics(self) -> Dict:
        metrics = {}
        
        if self.experiment_metrics['encrypt_times']:
            times = self.experiment_metrics['encrypt_times']
            metrics['avg_encrypt_time'] = np.mean(times)
            metrics['std_encrypt_time'] = np.std(times)
            metrics['min_encrypt_time'] = np.min(times)
            metrics['max_encrypt_time'] = np.max(times)
        
        if self.experiment_metrics['model_encrypt_times']:
            times = self.experiment_metrics['model_encrypt_times']
            metrics['avg_model_encrypt_time'] = np.mean(times)
            metrics['std_model_encrypt_time'] = np.std(times)
        
        if self.experiment_metrics['communication_overhead']:
            sizes = self.experiment_metrics['communication_overhead']
            metrics['avg_cipher_size'] = np.mean(sizes)
            metrics['total_communication'] = sum(sizes)
        
        if self.experiment_metrics['data_sizes'] and self.experiment_metrics['communication_overhead']:
            ratios = [c/d for c, d in zip(self.experiment_metrics['communication_overhead'], 
                                          self.experiment_metrics['data_sizes'])]
            metrics['avg_expansion_ratio'] = np.mean(ratios)
        
        return metrics
    
    def reset_metrics(self):
        self.experiment_metrics = {
            'encrypt_times': [],
            'policy_sizes': [],
            'data_sizes': [],
            'model_encrypt_times': [],
            'communication_overhead': []
        }