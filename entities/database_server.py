# decart/entities/database_server.py

import sys
import os
import copy
import time
import importlib
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.homomorphic import HomomorphicEncryption
from entities.key_curator import KeyCurator
from entities.data_owner import DataOwner
from entities.data_querier import DataQuerier
from schemes.decart_star import DeCartStarParams


class DatabaseServer:
    """
    Database Server
    
    1. Store encrypted datasets from data owners
    2. Receive encrypted AI models from data queriers
    3. Execute encrypted AI queries - Query(C_M, C_m) → ER
    4. Return encrypted query results
    5. Support dataset updates after revocation
    6. Reject queries from revoked users
    7. Track queries by model type
    """
    
    def __init__(self,
                 server_id: str = "ds1",
                 key_curator: Optional[KeyCurator] = None,
                 scheme: str = "decart_star"):
        
        self.server_id = server_id
        self.key_curator = key_curator
        self.scheme = scheme.lower()
        
        # Storage structures
        self.datasets = {}           # owner_id -> {dataset_id -> {'C_m': , 'sk_h_s': , 'metadata': , 'valid': bool}}
        self.query_logs = []         # Query logs
        self.access_logs = []        # Access logs
        self.revoked_queries_blocked = 0  # Count of revoked-user queries blocked
        
        # ===== Model type statistics =====
        self.model_type_stats = defaultdict(int)  # Query count by model type
        self.model_query_times = defaultdict(list)  # Query time by model type
        
        # Dataset version tracking
        self.dataset_versions = {}   # owner_id -> {dataset_id -> version}
        
        # Performance statistics
        self.stats = {
            'total_datasets': 0,
            'total_queries': 0,
            'total_computation_time': 0,
            'blocked_revoked_queries': 0,
            'dataset_updates': 0,
            'start_time': time.time(),
            # Additional statistics
            'dot_queries': 0,
            'decision_tree_queries': 0,
            'neural_network_queries': 0,
            'unknown_model_queries': 0
        }
        
        print(f"\n  Database Server entity initialized")
        print(f"   Server ID: {server_id}")
        print(f"   Scheme: {self.scheme_name() if key_curator else scheme}")

    
    def scheme_name(self) -> str:
        if self.key_curator:
            return self.key_curator.scheme_name
        return "DeCart*" if self.scheme == "decart_star" else "DeCart"
    
    # Revocation checks
    
    def _check_querier_revoked(self, querier_id: int) -> bool:
        if self.key_curator and self.key_curator.is_revoked(querier_id):
            self.revoked_queries_blocked += 1
            self.stats['blocked_revoked_queries'] += 1
            print(f"    Querier {querier_id} has been revoked; query denied")
            return True
        return False
    
    def _check_dataset_valid(self, owner_id: int, dataset_id: str) -> bool:

        if owner_id not in self.datasets:
            return False
        
        if dataset_id not in self.datasets[owner_id]:
            return False
        
        dataset_info = self.datasets[owner_id][dataset_id]
        return dataset_info.get('valid', True)
    
    #  Data storage interface
    
    def store_dataset(self,
                     owner_id: int,
                     dataset_id: str,
                     C_m: Dict,
                     sk_h_s: Any,
                     metadata: Optional[Dict] = None) -> bool:

        print(f"\n[Database Server {self.server_id}] Storing dataset")
        print(f"   Owner: {owner_id}")
        print(f"   Dataset: {dataset_id}")
        print(f"   Record count: {len(C_m.get('c6_i', []))}")
        
        # Initialize the owner's storage space
        if owner_id not in self.datasets:
            self.datasets[owner_id] = {}
            print(f"   Created storage space for owner {owner_id}")
        
        # Check whether it already exists
        if dataset_id in self.datasets[owner_id]:
            print(f"     Dataset already exists; overwriting")
        
        # Get the current version number
        if owner_id not in self.dataset_versions:
            self.dataset_versions[owner_id] = {}
        version = self.dataset_versions[owner_id].get(dataset_id, 0) + 1
        
        # Store the dataset
        self.datasets[owner_id][dataset_id] = {
            'C_m': C_m,
            'sk_h_s': sk_h_s,
            'metadata': metadata or {},
            'store_time': time.time(),
            'owner_id': owner_id,
            'dataset_id': dataset_id,
            'record_count': len(C_m.get('c6_i', [])),
            'access_count': 0,
            'valid': True,
            'version': version,
            'original_policy': C_m.get('P', []).copy()
        }
        
        self.dataset_versions[owner_id][dataset_id] = version
        self.stats['total_datasets'] += 1
        
        print(f"     Dataset stored successfully")
        print(f"      Version: {version}")
        print(f"      Owner {owner_id} datasets: {list(self.datasets[owner_id].keys())}")
        print(f"      Total datasets: {self.stats['total_datasets']}")
        
        return True
    
    def update_dataset(self,
                      owner_id: int,
                      dataset_id: str,
                      C_m_new: Dict,
                      sk_h_s_new: Optional[Any] = None,
                      metadata: Optional[Dict] = None) -> bool:
        
        print(f"\n[Database Server {self.server_id}] Updating dataset")
        print(f"   Owner: {owner_id}")
        print(f"   Dataset: {dataset_id}")
        
        if owner_id not in self.datasets:
            print(f"     Owner {owner_id} does not exist")
            return False
        
        if dataset_id not in self.datasets[owner_id]:
            print(f"     Dataset {dataset_id} does not exist")
            return False
        
        # Get the original dataset information
        old_info = self.datasets[owner_id][dataset_id]
        old_policy = old_info['C_m'].get('P', [])
        new_policy = C_m_new.get('P', [])
        
        print(f"   Original policy: {old_policy}")
        print(f"   New policy: {new_policy}")
        
        # Get the new version number
        version = self.dataset_versions[owner_id].get(dataset_id, 0) + 1
        
        # Update the dataset
        self.datasets[owner_id][dataset_id] = {
            'C_m': C_m_new,
            'sk_h_s': sk_h_s_new if sk_h_s_new is not None else old_info['sk_h_s'],
            'metadata': {**old_info['metadata'], **(metadata or {})},
            'store_time': time.time(),
            'owner_id': owner_id,
            'dataset_id': dataset_id,
            'record_count': len(C_m_new.get('c6_i', [])),
            'access_count': old_info['access_count'],
            'valid': True,
            'version': version,
            'previous_version': old_info.get('version', 0),
            'original_policy': old_policy,
            'updated_after_revoke': True,
            'removed_users': [u for u in old_policy if u not in new_policy]
        }
        
        self.dataset_versions[owner_id][dataset_id] = version
        self.stats['dataset_updates'] += 1
        
        print(f"     Dataset updated successfully")
        print(f"      New version: {version}")
        print(f"      Removed users: {[u for u in old_policy if u not in new_policy]}")
        
        return True
    
    def mark_dataset_invalid(self, owner_id: int, dataset_id: str, reason: str = "") -> bool:

        if owner_id not in self.datasets:
            return False
        
        if dataset_id not in self.datasets[owner_id]:
            return False
        
        self.datasets[owner_id][dataset_id]['valid'] = False
        self.datasets[owner_id][dataset_id]['invalid_reason'] = reason
        self.datasets[owner_id][dataset_id]['invalid_time'] = time.time()
        
        print(f"\n[Database Server] Dataset {dataset_id} marked as invalid")
        print(f"   Reason: {reason}")
        
        return True
    
    def batch_store_datasets(self,
                           datasets: List[Tuple[int, str, Dict, Any, Optional[Dict]]]) -> int:

        success_count = 0
        for owner_id, dataset_id, C_m, sk_h_s, metadata in datasets:
            if self.store_dataset(owner_id, dataset_id, C_m, sk_h_s, metadata):
                success_count += 1
        
        print(f"\n[Database Server] Batch storage completed: {success_count}/{len(datasets)}")
        return success_count
    
    def get_dataset(self, owner_id: int, dataset_id: str) -> Tuple[Optional[Dict], Optional[Any]]:

        if owner_id not in self.datasets:
            print(f"     Owner {owner_id} has not stored any dataset")
            return None, None
        
        if dataset_id not in self.datasets[owner_id]:
            print(f"     Dataset {dataset_id} does not exist")
            print(f"   Available datasets: {list(self.datasets[owner_id].keys())}")
            return None, None
        
        dataset = self.datasets[owner_id][dataset_id]
        
        # Check whether the dataset is valid
        if not dataset.get('valid', True):
            print(f"    Dataset {dataset_id} is invalid")
            print(f"      Reason: {dataset.get('invalid_reason', 'unknown')}")
            return None, None
        
        dataset['access_count'] += 1
        
        return dataset['C_m'], dataset['sk_h_s']
    
    def list_datasets(self, owner_id: Optional[int] = None, include_invalid: bool = False) -> List[Dict]:

        result = []
        
        if owner_id is not None:
            # List datasets for a specific owner
            if owner_id in self.datasets:
                for ds_id, info in self.datasets[owner_id].items():
                    if not include_invalid and not info.get('valid', True):
                        continue
                    result.append({
                        'owner_id': owner_id,
                        'dataset_id': ds_id,
                        'record_count': info['record_count'],
                        'store_time': info['store_time'],
                        'access_count': info['access_count'],
                        'metadata': info['metadata'],
                        'valid': info.get('valid', True),
                        'version': info.get('version', 0),
                        'policy': info['C_m'].get('P', [])
                    })
        else:
            # List all datasets
            for oid in self.datasets:
                for ds_id, info in self.datasets[oid].items():
                    if not include_invalid and not info.get('valid', True):
                        continue
                    result.append({
                        'owner_id': oid,
                        'dataset_id': ds_id,
                        'record_count': info['record_count'],
                        'store_time': info['store_time'],
                        'access_count': info['access_count'],
                        'metadata': info['metadata'],
                        'valid': info.get('valid', True),
                        'version': info.get('version', 0),
                        'policy': info['C_m'].get('P', [])
                    })
        
        return sorted(result, key=lambda x: x['store_time'], reverse=True)
    
    def delete_dataset(self, owner_id: int, dataset_id: str) -> bool:

        if owner_id not in self.datasets:
            print(f"     Owner {owner_id} does not exist")
            return False
        
        if dataset_id not in self.datasets[owner_id]:
            print(f"     Dataset {dataset_id} does not exist")
            return False
        
        del self.datasets[owner_id][dataset_id]
        if not self.datasets[owner_id]:
            del self.datasets[owner_id]
        
        self.stats['total_datasets'] -= 1
        print(f"[Database Server] Dataset {dataset_id} deleted")
        print(f"   Remaining datasets: {self.stats['total_datasets']}")
        
        return True
    
    #  Query
    
    def execute_query(self,
                     querier_id: int,
                     owner_id: int,
                     dataset_id: str,
                     C_M: Dict) -> Optional[Dict]:

        print(f"\n[Database Server {self.server_id}] Executing encrypted query")
        print(f"   Querier: {querier_id}")
        print(f"   Owner: {owner_id}")
        print(f"   Dataset: {dataset_id}")
        
        # 1. Check whether the querier has been revoked
        if self._check_querier_revoked(querier_id):
            print(f"     Querier has been revoked; execution denied")
            
            self.query_logs.append({
                'timestamp': time.time(),
                'querier_id': querier_id,
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'status': 'blocked_revoked',
                'error': 'Querier revoked'
            })
            
            return None
        
        # 2. Fetch the dataset
        C_m, sk_h_s = self.get_dataset(owner_id, dataset_id)
        if C_m is None:
            print(f"     Dataset does not exist or is invalid")
            return None
        
        # 3. Verify access permissions (guaranteed by the system Check algorithm)
        if not C_M.get('access_granted', False):
            print(f"     Unauthorized access")
            return None
        
        # 4. Determine the model type
        encrypted_model = C_M.get('encrypted_model', {})
        if isinstance(encrypted_model, dict):
            model_type = encrypted_model.get('type', 'unknown')
        else:
            model_type = 'dot_product'
        
        print(f"   Model type: {model_type}")
        
        # 5. Execute the encrypted query
        try:
            start_time = time.time()
            
            # Call the Query algorithm of the selected scheme
            ER = self.key_curator.system.query(C_M, C_m, sk_h_s)
            
            query_time = time.time() - start_time
            
            # 6. Update statistics
            self.model_type_stats[model_type] += 1
            self.model_query_times[model_type].append(query_time)
            
            if model_type == 'decision_tree':
                self.stats['decision_tree_queries'] += 1
            elif model_type == 'neural_network':
                self.stats['neural_network_queries'] += 1
            elif model_type == 'dot_product':
                self.stats['dot_queries'] += 1
            else:
                self.stats['unknown_model_queries'] += 1
            
            # 7. Record logs
            self.query_logs.append({
                'timestamp': time.time(),
                'querier_id': querier_id,
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'query_time': query_time,
                'result_count': len(ER.get('encrypted_results', [])),
                'model_type': model_type,
                'status': 'success'
            })
            
            self.stats['total_queries'] += 1
            self.stats['total_computation_time'] += query_time
            
            print(f"     Query executed successfully")
            print(f"      Execution time: {query_time*1000:.2f} ms")
            print(f"      Result count: {len(ER.get('encrypted_results', []))}")
            print(f"      Model type: {model_type}")
            
            return ER
            
        except Exception as e:
            print(f"     Query execution failed: {e}")
            
            self.query_logs.append({
                'timestamp': time.time(),
                'querier_id': querier_id,
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'error': str(e),
                'model_type': model_type,
                'status': 'failed'
            })
            
            return None
    
    #  Batch queries
    
    def batch_execute_queries(self,
                            queries: List[Tuple[int, int, str, Dict]]) -> List[Optional[Dict]]:

        print(f"\n[Database Server] Executing {len(queries)} batch queries")
        
        results = []
        blocked_count = 0
        
        for querier_id, owner_id, dataset_id, C_M in queries:
            # Check each querier
            if self._check_querier_revoked(querier_id):
                results.append(None)
                blocked_count += 1
                continue
                
            ER = self.execute_query(querier_id, owner_id, dataset_id, C_M)
            results.append(ER)
        
        success_count = sum(1 for r in results if r is not None)
        print(f"     Batch query completed: {success_count}/{len(queries)} succeeded")
        print(f"      {blocked_count} revoked users were blocked")
        
        return results
    
    #  Model type statistics
    
    def get_model_type_stats(self) -> Dict:

        avg_times = {}
        for model_type, times in self.model_query_times.items():
            if times:
                avg_times[model_type] = sum(times) / len(times)
            else:
                avg_times[model_type] = 0
        
        return {
            'dot_product': self.stats['dot_queries'],
            'decision_tree': self.stats['decision_tree_queries'],
            'neural_network': self.stats['neural_network_queries'],
            'unknown': self.stats['unknown_model_queries'],
            'total': self.stats['total_queries'],
            'avg_times': avg_times,
            'raw_stats': dict(self.model_type_stats)
        }
    
    def print_model_stats(self):

        print("\n" + "="*60)
        print(" Model query statistics")
        print("="*60)
        
        stats = self.get_model_type_stats()
        
        print(f"   Dot-product models: {stats['dot_product']}")
        print(f"   Decision tree models: {stats['decision_tree']}")
        print(f"   Neural network models: {stats['neural_network']}")
        print(f"   Unknown types: {stats['unknown']}")
        print(f"   Total: {stats['total']}")
        
        if stats['avg_times']:
            print(f"\n   Average query time:")
            for model_type, avg_time in stats['avg_times'].items():
                if avg_time > 0:
                    print(f"     {model_type}: {avg_time*1000:.2f} ms")
    
    #  Query logs and statistics
    
    def get_query_logs(self, limit: int = 100, include_blocked: bool = True) -> List[Dict]:

        logs = self.query_logs
        if not include_blocked:
            logs = [log for log in logs if log.get('status') != 'blocked_revoked']
        
        return sorted(
            logs[-limit:],
            key=lambda x: x['timestamp'],
            reverse=True
        )
    
    def get_access_logs(self, limit: int = 100) -> List[Dict]:

        logs = []
        for owner_id in self.datasets:
            for ds_id, info in self.datasets[owner_id].items():
                if info.get('valid', True):
                    logs.append({
                        'timestamp': info['store_time'],
                        'owner_id': owner_id,
                        'dataset_id': ds_id,
                        'action': 'store',
                        'record_count': info['record_count'],
                        'version': info.get('version', 0)
                    })
        
        logs.extend(self.query_logs)
        
        return sorted(logs[-limit:], key=lambda x: x['timestamp'], reverse=True)
    
    def get_server_stats(self) -> Dict:

        total_records = 0
        valid_datasets = 0
        invalid_datasets = 0
        
        for owner_id in self.datasets:
            for ds_id, info in self.datasets[owner_id].items():
                total_records += info['record_count']
                if info.get('valid', True):
                    valid_datasets += 1
                else:
                    invalid_datasets += 1
        
        # Get model statistics
        model_stats = self.get_model_type_stats()
        
        return {
            'server_id': self.server_id,
            'scheme': self.scheme_name(),
            'total_datasets': self.stats['total_datasets'],
            'valid_datasets': valid_datasets,
            'invalid_datasets': invalid_datasets,
            'total_queries': self.stats['total_queries'],
            'blocked_revoked_queries': self.stats['blocked_revoked_queries'],
            'dataset_updates': self.stats['dataset_updates'],
            'total_records': total_records,
            'unique_owners': len(self.datasets),
            'avg_query_time': (self.stats['total_computation_time'] / self.stats['total_queries']) 
                             if self.stats['total_queries'] > 0 else 0,
            'model_stats': model_stats,
            'uptime': time.time() - self.stats['start_time']
        }
    
    def clear_logs(self):

        self.query_logs = []
        self.access_logs = []
        self.revoked_queries_blocked = 0
        self.model_type_stats.clear()
        self.model_query_times.clear()
        print(f"[Database Server] Logs cleared")
    
    def clear_all_data(self):
        self.datasets = {}
        self.query_logs = []
        self.access_logs = []
        self.dataset_versions = {}
        self.model_type_stats.clear()
        self.model_query_times.clear()
        self.stats['total_datasets'] = 0
        self.stats['total_queries'] = 0
        self.stats['total_computation_time'] = 0
        self.stats['blocked_revoked_queries'] = 0
        self.stats['dataset_updates'] = 0
        self.stats['dot_queries'] = 0
        self.stats['decision_tree_queries'] = 0
        self.stats['neural_network_queries'] = 0
        self.stats['unknown_model_queries'] = 0
        self.revoked_queries_blocked = 0
        print(f"[Database Server] All data cleared")


#  Test code


def test_database_server_model_stats():
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.data_querier import DataQuerier
    from schemes.decart_star import DeCartStarParams
    import glob
    
    # 1. Initialize system
    print("\n1. Initialize system...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. Create users
    print("\n2. Create users...")
    owner_id = 5
    querier_id = 6
    
    sk_o, pk_o, pap_o = curator.generate_user_key(owner_id)
    curator.register(owner_id, pk_o, pap_o)
    
    sk_q, pk_q, pap_q = curator.generate_user_key(querier_id)
    curator.register(querier_id, pk_q, pap_q)
    
    # 3. Establish trust relationships
    print("\n3. Establish trust relationships...")
    curator.add_trust(querier_id, owner_id)
    
    # 4. Data owner encrypts data
    print("\n4. Data owner encrypts data...")
    owner = DataOwner(owner_id=owner_id, key_curator=curator, scheme="decart_star")
    data = [np.random.randn(5).tolist() for _ in range(3)]
    policy = [owner_id, querier_id]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy)
    
    # 5. Create the database server
    print("\n5. Create the database server...")
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(owner_id, ds_id, C_m, sk_h_s)
    
    # 6. Create the querier
    print("\n6. Create the querier...")
    querier = DataQuerier(querier_id=querier_id, key_curator=curator, scheme="decart_star")
    
    # 7. Execute queries of different types
    print("\n7. Execute queries of different types...")
    
    # Dot-product query
    print(f"\n   [Dot-product query]")
    C_M_base = querier.check_access(C_m)
    model = [0.1, 0.2, 0.3, 0.4, 0.5]
    C_M = querier.encrypt_ai_model(model, C_M_base)
    db_server.execute_query(querier_id, owner_id, ds_id, C_M)
    
    # Neural network query (simulated)
    print(f"\n   [Neural network query]")
    C_M_base = querier.check_access(C_m)
    C_M_base['encrypted_model'] = {'type': 'neural_network', 'layers': []}
    C_M_base['access_granted'] = True
    db_server.execute_query(querier_id, owner_id, ds_id, C_M_base)
    
    # Decision tree query (simulated)
    print(f"\n   [Decision tree query]")
    C_M_base = querier.check_access(C_m)
    C_M_base['encrypted_model'] = {'type': 'decision_tree', 'nodes': []}
    C_M_base['access_granted'] = True
    db_server.execute_query(querier_id, owner_id, ds_id, C_M_base)
    
    # 8. View statistics
    print("\n8. View model statistics...")
    db_server.print_model_stats()
    
    # Fix: use the correct method name get_model_type_stats()
    stats = db_server.get_model_type_stats()
    print(f"\n   Statistics: {stats}")
    
    assert stats['dot_product'] == 1, f"Dot-product queries should be 1, got {stats['dot_product']}"
    assert stats['neural_network'] == 1, f"Neural network queries should be 1, got {stats['neural_network']}"
    assert stats['decision_tree'] == 1, f"Decision tree queries should be 1, got {stats['decision_tree']}"
    
    print(f"\n  Database Server model statistics test passed")
    
    return db_server

def test_database_server_revoke_handling():
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.data_querier import DataQuerier
    from schemes.decart_star import DeCartStarParams

    # 1. Initialize system
    print("\n1. Initialize system...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()

    # 2. Create users
    print("\n2. Create users...")
    users = [5, 6, 7, 8]
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)

    # 3. Establish trust relationships
    print("\n3. Establish trust relationships...")
    curator.add_trust(6, 5)
    curator.add_trust(7, 5)
    curator.add_trust(8, 5)

    # 4. Create data owner
    print("\n4. Create data owner...")
    owner = DataOwner(owner_id=5, key_curator=curator, scheme="decart_star")

    # 5. Encrypt dataset
    print("\n5. Encrypt dataset...")
    data = [[1.0, 2.0, 3.0, 4.0, 5.0]] * 3
    policy = [5, 6, 7, 8]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy, store_original=True)

    # 6. Create the database server
    print("\n6. Create the database server...")
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(5, ds_id, C_m, sk_h_s, {'name': 'test'})

    # 7. Create queriers
    print("\n7. Create queriers...")
    querier6 = DataQuerier(querier_id=6, key_curator=curator, scheme="decart_star")
    querier7 = DataQuerier(querier_id=7, key_curator=curator, scheme="decart_star")

    # 8. Normal query
    print("\n8. Normal user executes query...")
    C_M6 = querier6.check_access(C_m)
    model6 = querier6.create_ai_model(dimension=5)
    C_M6 = querier6.encrypt_ai_model(model6, C_M6)
    result6 = db_server.execute_query(6, 5, ds_id, C_M6)
    assert result6 is not None, "The normal user's query should succeed"
    print(f"     User 6 query succeeded")

    # 9. Revoke user 6
    print("\n9. Revoke user 6...")
    curator.revoke_user(6)

    # 10. Try letting revoked user 6 query (expected to fail)
    print("\n10. Revoked user tries to query (expected failure)...")
    result6_after = db_server.execute_query(6, 5, ds_id, C_M6)
    assert result6_after is None, "A revoked user's query should fail"
    print(f"     Revoked user query denied")

    # 11. Check statistics
    print("\n11. Check statistics...")
    stats = db_server.get_server_stats()
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Blocked revoked queries: {stats['blocked_revoked_queries']}")
    assert stats['blocked_revoked_queries'] >= 1, "There should be blocked revoked queries"

    print(f"\n  Database Server revocation handling test passed")

    return db_server


def test_database_server_revoke_handling():
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.data_querier import DataQuerier
    from schemes.decart_star import DeCartStarParams

    # 1. Initialize system
    print("\n1. Initialize system...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()

    # 2. Create users
    print("\n2. Create users...")
    users = [5, 6, 7, 8]
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)

    # 3. Establish trust relationships
    print("\n3. Establish trust relationships...")
    curator.add_trust(6, 5)
    curator.add_trust(7, 5)
    curator.add_trust(8, 5)

    # 4. Create data owner
    print("\n4. Create data owner...")
    owner = DataOwner(owner_id=5, key_curator=curator, scheme="decart_star")

    # 5. Encrypt dataset
    print("\n5. Encrypt dataset...")
    data = [[1.0, 2.0, 3.0, 4.0, 5.0]] * 3
    policy = [5, 6, 7, 8]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy, store_original=True)

    # 6. Create the database server
    print("\n6. Create the database server...")
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(5, ds_id, C_m, sk_h_s, {'name': 'test'})

    # 7. Create queriers
    print("\n7. Create queriers...")
    querier6 = DataQuerier(querier_id=6, key_curator=curator, scheme="decart_star")
    querier7 = DataQuerier(querier_id=7, key_curator=curator, scheme="decart_star")

    # 8. Normal query
    print("\n8. Normal user executes query...")
    C_M6 = querier6.check_access(C_m)
    model6 = querier6.create_ai_model(dimension=5)
    C_M6 = querier6.encrypt_ai_model(model6, C_M6)
    result6 = db_server.execute_query(6, 5, ds_id, C_M6)
    assert result6 is not None, "The normal user's query should succeed"
    print(f"     User 6 query succeeded")

    # 9. Revoke user 6
    print("\n9. Revoke user 6...")
    curator.revoke_user(6)

    # 10. Try letting revoked user 6 query (expected to fail)
    print("\n10. Revoked user tries to query (expected failure)...")
    result6_after = db_server.execute_query(6, 5, ds_id, C_M6)
    assert result6_after is None, "A revoked user's query should fail"
    print(f"     Revoked user query denied")

    # 11. Check statistics
    print("\n11. Check statistics...")
    stats = db_server.get_server_stats()
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Blocked revoked queries: {stats['blocked_revoked_queries']}")
    assert stats['blocked_revoked_queries'] >= 1, "There should be blocked revoked queries"

    print(f"\n  Database Server revocation handling test passed")

    return db_server


def test_database_server_batch_revoke():
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.data_querier import DataQuerier
    from schemes.decart_star import DeCartStarParams

    # 1. Initialize system
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()

    # 2. Create users
    users = [5, 6, 7, 8]
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)

    # 3. Establish trust relationships
    for uid in [6, 7, 8]:
        curator.add_trust(uid, 5)

    # 4. Data owner encrypts data
    owner = DataOwner(owner_id=5, key_curator=curator, scheme="decart_star")
    data = [[1.0, 2.0, 3.0]]
    policy = [5, 6, 7, 8]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy)

    # 5. Database server
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(5, ds_id, C_m, sk_h_s)

    # 6. Create C_M for all queriers
    queries = []
    for uid in [6, 7, 8]:
        querier = DataQuerier(querier_id=uid, key_curator=curator, scheme="decart_star")
        C_M = querier.check_access(C_m)
        model = querier.create_ai_model(dimension=3)
        C_M = querier.encrypt_ai_model(model, C_M)
        queries.append((uid, 5, ds_id, C_M))

    # 7. Revoke user 7
    print("\n7. Revoke user 7...")
    curator.revoke_user(7)

    # 8. Execute batch queries
    print("\n8. Execute batch queries...")
    results = db_server.batch_execute_queries(queries)

    # 9. Verify results
    print("\n9. Verify results...")
    success_count = sum(1 for r in results if r is not None)
    print(f"   Successful queries: {success_count}")
    print(f"   Total queries: {len(queries)}")
    assert success_count == 2, "There should be 2 successful queries (users 6 and 8)"
    assert results[1] is None, "User 7's query should be denied"

    print(f"\n  Batch revocation handling test passed")


if __name__ == "__main__":

    # Run tests
    test_database_server_revoke_handling()
    test_database_server_batch_revoke()
    test_database_server_model_stats()

