# decart/experiments/our_decart/server.py

import sys
import os
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from entities.database_server import DatabaseServer as BaseDatabaseServer


class DatabaseServer(BaseDatabaseServer):
    
    def __init__(self, server_id: str, key_curator, scheme: str = "decart"):
        super().__init__(server_id, key_curator, scheme)
        
        # Experiment measurement metrics
        self.experiment_metrics = {
            'query_times': [],
            'result_sizes': [],
            'model_types': [],
            'dataset_access_times': []
        }
    
    def execute_query_with_metrics(self,
                                querier_id: int,
                                owner_id: int,
                                dataset_id: str,
                                C_M: Dict) -> Tuple[Optional[Dict], Dict]:
        # Record start time
        start = time.perf_counter()
        
        # Execute query
        ER = self.execute_query(querier_id, owner_id, dataset_id, C_M)
        
        # Calculate query time
        query_time = time.perf_counter() - start
        
        metrics = {
            'query_time': query_time,
            'query_time_ms': query_time * 1000,
            'success': ER is not None
        }
        
        if ER is not None:
            # Measure result size using an estimation fallback
            try:
                import pickle
                result_size = len(pickle.dumps(ER))
            except (TypeError, pickle.PicklingError):
                # Estimate size
                if 'encrypted_results' in ER:
                    result_size = len(ER['encrypted_results']) * 1024 * 1024  # 1MB per result
                else:
                    result_size = 1024 * 1024  # default 1MB
            
            metrics['result_size'] = result_size
            metrics['num_results'] = ER.get('num_results', 0)
            
            self.experiment_metrics['result_sizes'].append(result_size)
            
            # Record model type
            model_type = C_M.get('model_type', 'unknown')
            self.experiment_metrics['model_types'].append(model_type)
            
            print(f"\n [DatabaseServer] Query metrics:")
            print(f"   Query time: {query_time*1000:.2f} ms")
            print(f"   Result size: {result_size/1024:.2f} KB")
            print(f"   Number of results: {metrics['num_results']}")
        
        self.experiment_metrics['query_times'].append(query_time)
        
        return ER, metrics
    
    def get_experiment_metrics(self) -> Dict:
        metrics = {}
        
        if self.experiment_metrics['query_times']:
            times = self.experiment_metrics['query_times']
            metrics['avg_query_time'] = np.mean(times)
            metrics['std_query_time'] = np.std(times)
            metrics['min_query_time'] = np.min(times)
            metrics['max_query_time'] = np.max(times)
        
        if self.experiment_metrics['result_sizes']:
            sizes = self.experiment_metrics['result_sizes']
            metrics['avg_result_size'] = np.mean(sizes)
            metrics['total_results_size'] = sum(sizes)
        
        # Aggregate by model type
        if self.experiment_metrics['model_types']:
            from collections import Counter
            model_counts = Counter(self.experiment_metrics['model_types'])
            metrics['model_type_counts'] = dict(model_counts)
        
        return metrics
    
    def reset_metrics(self):
        self.experiment_metrics = {
            'query_times': [],
            'result_sizes': [],
            'model_types': [],
            'dataset_access_times': []
        }