# decart/entities/data_querier.py

import sys
import os
import copy
import time
import pickle
import glob
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.homomorphic import HomomorphicEncryption
from entities.key_curator import KeyCurator
from config import Config


class DataQuerier:
    """
    Data Querier

    1. Submit AI queries to the database server
    2. Use its own key to verify access permissions
    3. Encrypt AI models and send them to the database server
    4. Decrypt and obtain query results
    5. Revocation checks - revoked users cannot query
    6. Support loading pretrained models for queries
    
    """
    
    def __init__(self,
                 querier_id: int,
                 key_curator: KeyCurator,
                 scheme: str = "decart_star"):
        self.querier_id = querier_id
        self.key_curator = key_curator
        self.scheme = scheme.lower()
        
        # Verify that the scheme matches the key curator's scheme
        if self.scheme not in key_curator.scheme.lower():
            print(f"     Warning: DataQuerier uses {self.scheme}, "
                  f"KeyCurator uses {key_curator.scheme_name}")
        
        # Get system parameters
        self.crs = key_curator.crs
        self.pp = key_curator.pp
        self.aux = key_curator.aux
        
        if self.crs is None or self.pp is None:
            raise ValueError("Key Curator has not executed setup() yet")
        
        # Initialize homomorphic encryption (real TenSEAL CKKS)
        self.he = HomomorphicEncryption(poly_modulus_degree=Config.POLY_MODULUS_DEGREE)
        
        # User keys (retrieved from KeyCurator)
        self._load_user_keys()
        
        # Query history
        self.query_history = []
        
        # Loaded models
        self.loaded_models = {}          # model_id -> model_info
        self.encrypted_models = {}       # enc_model_id -> encrypted_model
        
        # State cache
        self._cached_aux = None
        self._cached_aux_time = 0
        
        # Revocation notification callbacks
        self._revoke_handlers = []
        
        print(f"\n  Data Querier entity initialized")
        print(f"   Querier ID: {querier_id}")
        print(f"   Scheme: {key_curator.scheme_name}")
        print(f"   Block: {self.block}")
        print(f"   u_id': {self.u_id_prime}")
        print(f"   aux length: {len(self.key_curator.get_user_aux(querier_id))}")
        print(f"   Supports revocation checks")
        print(f"   Supports pretrained model queries")
    
    def _load_user_keys(self):
        # Verify whether the user is registered
        if self.querier_id not in self.key_curator.registered_users:
            raise ValueError(f"User {self.querier_id} is not registered with Key Curator")
        
        # Check whether the user has been revoked
        if self.key_curator.is_revoked(self.querier_id):
            raise ValueError(f"User {self.querier_id} has been revoked and cannot be initialized as a querier")
        
        # Get user information
        self.block = self.key_curator.user_blocks.get(self.querier_id)
        self.u_id_prime = self.key_curator.user_id_prime.get(self.querier_id)
        self.pk_id = self.key_curator.user_public_keys.get(self.querier_id)
        self.pap_id = self.key_curator.user_pap.get(self.querier_id)
        
        # Get the private key (from system.user_secrets)
        self.sk_id = self.key_curator.system.user_secrets[self.querier_id]['sk_id']
    
    def check_revoked(self) -> bool:

        is_revoked = self.key_curator.is_revoked(self.querier_id)
        if is_revoked:
            print(f"\n[Data Querier {self.querier_id}] Warning: you have been revoked by the system")
            print(f"   Query operations are unavailable")
        return is_revoked
    
    def _check_before_operation(self, operation: str) -> bool:

        if self.check_revoked():
            print(f"     {operation} failed: user has been revoked")
            
            self.query_history.append({
                'timestamp': time.time(),
                'operation': operation,
                'status': 'failed_revoked',
                'error': 'User revoked'
            })
            
            return False
        return True
    
    # ========== New: load pretrained models ==========
    
    def load_pretrained_model(self, model_path: str, model_type: str = "cnn") -> str:
        print(f"\n[Data Querier {self.querier_id}] Loading pretrained model")
        print(f"   File: {model_path}")
        print(f"   Type: {model_type}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        # Load the configuration file
        with open(model_path, 'rb') as f:
            config = pickle.load(f)
        
        model_name = config.get('model_name', 'unknown')
        test_accuracy = config.get('test_accuracy', 0.0)
        architecture = config.get('architecture', {})
        
        print(f"   Model name: {model_name}")
        print(f"   Test accuracy: {test_accuracy:.4f}")
        
        # Generate the model ID
        timestamp = int(time.time() * 1000)
        model_id = f"querier_model_{self.querier_id}_{model_type}_{timestamp}"
        
        # Store model information
        self.loaded_models[model_id] = {
            'model_path': model_path,
            'model_type': model_type,
            'model_name': model_name,
            'test_accuracy': test_accuracy,
            'architecture': architecture,
            'load_time': time.time()
        }
        
        print(f"     Model loaded successfully: {model_id}")
        
        return model_id
    
    def load_all_models_from_dir(self, models_dir: str) -> Dict[str, str]:

        print(f"\n[Data Querier {self.querier_id}] Loading all models from directory")
        print(f"   Directory: {models_dir}")
        
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Directory does not exist: {models_dir}")
        
        import glob
        model_files = glob.glob(os.path.join(models_dir, "*.pkl"))
        
        if not model_files:
            print(f"   Warning: no model files found in directory")
            return {}
        
        result = {}
        for model_path in model_files:
            filename = os.path.basename(model_path)
            
            # Determine the model type from the file name
            if 'cnn_flattened' in filename:
                model_type = 'cnn'
            elif 'cnn_test' in filename:
                model_type = 'cnn_test'
            elif 'mlp' in filename:
                model_type = 'mlp'
            elif 'svm' in filename:
                model_type = 'svm'
            else:
                model_type = 'unknown'
            
            try:
                model_id = self.load_pretrained_model(model_path, model_type)
                result[model_type] = model_id
                print(f"     Loaded {model_type}: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"     Failed to load {filename}: {e}")
        
        return result
    
    # ========== Paper algorithm: Check (delegated to the selected scheme) ==========
    
    def check_access(self, C_m: Dict) -> Optional[Dict]:
        # Check revocation status before the operation
        if not self._check_before_operation("access check"):
            return None
        
        print(f"\n[Data Querier {self.querier_id}] Checking access permissions")
        print(f"   Scheme: {self.key_curator.scheme_name}")
        print(f"   Owner: {C_m.get('owner_id', 'unknown')}")
        
        # Call the Check algorithm of the selected scheme
        C_M = self.key_curator.system.check(
            self.querier_id,
            self.sk_id,
            C_m
        )
        
        if C_M:
            print(f"     Access authorized successfully")
            self.query_history.append({
                'timestamp': time.time(),
                'owner_id': C_m.get('owner_id'),
                'status': 'authorized'
            })
        else:
            print(f"     Access denied")
            self.query_history.append({
                'timestamp': time.time(),
                'owner_id': C_m.get('owner_id'),
                'status': 'denied'
            })
        
        return C_M
    
    # ========== AI model management ==========
    
    def create_ai_model(self, 
                       model_type: str = "linear",
                       dimension: int = 5,
                       weights: Optional[List[float]] = None) -> List[float]:

        if self.check_revoked():
            print(f"    Warning: user has been revoked, but model creation is still allowed (for testing only)")
        
        if weights is not None:
            model = weights
        else:
            # Generate random model weights
            np.random.seed(int(time.time()) % 1000)
            if model_type == "linear":
                model = np.random.randn(dimension).tolist()
            elif model_type == "cnn":
                # Simplified CNN model
                model = np.random.randn(dimension * 2).tolist()
            else:
                model = np.random.randn(dimension).tolist()
        
        print(f"\n[Data Querier {self.querier_id}] Creating AI model")
        print(f"   Type: {model_type}")
        print(f"   Dimension: {len(model)}")
        print(f"   Weight sample: {model[:3]}")
        
        return model


    def encrypt_ai_model(self, model: Any, C_M: Dict) -> Dict:
        # Check revocation status before the operation
        if not self._check_before_operation("model encryption"):
            raise ValueError(f"User {self.querier_id} has been revoked and cannot encrypt models")
        
        print(f"\n[Data Querier {self.querier_id}] Encrypting AI model")
        
        if not C_M.get('access_granted', False):
            raise ValueError("No access permission, cannot encrypt model")

        # AI models should be encrypted with the homomorphic context restored by the system to avoid mismatches with the HE instance used during querying.
        system_he = self.key_curator.system.he
        
        # Handle based on model type
        if isinstance(model, list):
            # Dot-product model - encrypt the list directly
            print(f"   Encrypting dot-product model (list)")
            encrypted_model = system_he.encrypt(model)
            C_M['encrypted_model'] = encrypted_model
            C_M['model_dim'] = len(model)
            C_M['model_type'] = 'dot_product'
            
        elif isinstance(model, dict):
            # Dictionary-based models (neural networks, decision trees, etc.)
            model_type = model.get('type', 'unknown')
            print(f"   Encrypting {model_type} model (dict)")
            
            if model_type == 'neural_network':
                # Neural network model: supports both the legacy single-layer format and the new layers format.
                raw_layers = model.get('layers') or [{
                    'layer_idx': 0,
                    'layer_type': 'linear',
                    'activation': model.get('activation', 'linear'),
                    'weights': model.get('weights', []),
                    'bias': model.get('bias', []),
                    'weights_shape': (
                        int(model.get('output_dim', 10) or 10),
                        int(model.get('input_dim', 784) or 784),
                    ),
                    'bias_shape': (int(model.get('output_dim', 10) or 10),),
                }]

                encrypted_layers = []
                for layer in raw_layers:
                    weights = layer.get('weights', [])
                    bias = layer.get('bias', [])
                    weights_shape = tuple(layer.get('weights_shape', (0, 0)))
                    output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else int(layer.get('output_dim', 0) or 0)
                    input_dim = int(weights_shape[1]) if len(weights_shape) > 1 else int(layer.get('input_dim', 0) or 0)

                    encrypted_weight_rows = []
                    for row_idx in range(output_dim):
                        row_start = row_idx * input_dim
                        row_end = row_start + input_dim
                        weight_row = [float(value) for value in weights[row_start:row_end]]
                        if len(weight_row) < input_dim:
                            weight_row.extend([0.0] * (input_dim - len(weight_row)))
                        try:
                            encrypted_row = system_he.encrypt(weight_row)
                            encrypted_weight_rows.append(encrypted_row)
                        except Exception as e:
                            print(f"     Failed to encrypt weights for layer {layer.get('layer_idx', 0)} row {row_idx}: {e}")
                            encrypted_weight_rows.append(None)

                    encrypted_bias_vector = None
                    bias_values = [float(value) for value in bias]
                    if bias_values:
                        try:
                            encrypted_bias_vector = system_he.encrypt(bias_values)
                        except Exception as e:
                            print(f"     Failed to encrypt bias vector for layer {layer.get('layer_idx', 0)}: {e}")

                    encrypted_bias = []
                    if encrypted_bias_vector is None:
                        for b in bias_values:
                            try:
                                encrypted_b = system_he.encrypt([float(b)])
                                encrypted_bias.append(encrypted_b)
                            except Exception as e:
                                print(f"     Failed to encrypt bias for layer {layer.get('layer_idx', 0)}: {e}")
                                encrypted_bias.append(None)

                    encrypted_layers.append({
                        'layer_idx': layer.get('layer_idx', len(encrypted_layers)),
                        'layer_type': layer.get('layer_type', 'linear'),
                        'activation': layer.get('activation', 'linear'),
                        'weights_shape': (output_dim, input_dim),
                        'bias_shape': tuple(layer.get('bias_shape', (output_dim,))),
                        'encrypted_weights': [],
                        'encrypted_bias': encrypted_bias,
                        'encrypted_weight_rows': encrypted_weight_rows,
                        'encrypted_bias_vector': encrypted_bias_vector
                    })

                encrypted_model = {
                    'type': 'neural_network',
                    'layer_count': len(encrypted_layers),
                    'layers': encrypted_layers
                }
                C_M['encrypted_model'] = encrypted_model
                C_M['model_type'] = 'neural_network'
                
            elif model_type == 'decision_tree':
                # Decision tree model - requires system-level encryption
                print(f"   Decision tree models require system-level encryption")
                # Not handled here; call the system method from the upper layer
                C_M['encrypted_model'] = model
                C_M['model_type'] = 'decision_tree'
                
            else:
                # Unknown dict type, try storing directly
                print(f"   Unknown dict type: {model_type}")
                C_M['encrypted_model'] = model
                C_M['model_type'] = model_type
                
        elif hasattr(model, 'get_encryptable_params'):
            # Objects that expose encryption methods (e.g. DecisionTreeHE)
            print(f"   Encrypting model object")
            # Handled by the system method at a higher layer
            C_M['encrypted_model'] = model
            C_M['model_type'] = 'object'
            
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
        
        C_M['encrypt_time'] = time.time()
        
        print(f"     AI model encrypted successfully")
        print(f"      Model type: {C_M.get('model_type', 'unknown')}")
        
        return C_M
    
    # ========== Query using pretrained models ==========
    
    def prepare_encrypted_model(self, 
                               model_id: str, 
                               C_m: Dict) -> Optional[Dict]:

        print(f"\n[Data Querier {self.querier_id}] Preparing encrypted model")
        
        if model_id not in self.loaded_models:
            print(f"    Model does not exist: {model_id}")
            return None
        
        # Check access permissions
        C_M = self.check_access(C_m)
        if C_M is None:
            print(f"     No permission to access this dataset")
            return None
        
        model_info = self.loaded_models[model_id]
        architecture = model_info.get('architecture', {})
        
        # Build the encrypted model according to the model type
        if model_info['model_type'] in ['cnn', 'cnn_test', 'mlp', 'svm']:
            raw_layers = architecture.get('layers')
            if raw_layers:
                print(f"   Model architecture: multi-layer network with {len(raw_layers)} layers")
            else:
                weights = architecture.get('weights', [])
                bias = architecture.get('bias', [])
                if 'combined_weights' in architecture:
                    weights = architecture['combined_weights']
                    bias = architecture['combined_bias']
                input_dim = architecture.get('input_dim', 784)
                output_dim = architecture.get('output_dim', 10)
                print(f"   Model architecture: {input_dim} -> {output_dim}")
                print(f"   Weight count: {len(weights)}")
                print(f"   Bias count: {len(bias)}")
                raw_layers = [{
                    'layer_idx': 0,
                    'layer_type': 'linear',
                    'activation': architecture.get('activation', 'linear'),
                    'weights': weights,
                    'bias': bias,
                    'weights_shape': tuple(architecture.get('weights_shape', (output_dim, input_dim))),
                    'bias_shape': tuple(architecture.get('bias_shape', (output_dim,))),
                }]

            system_he = self.key_curator.system.he
            encrypted_layers = []
            for layer in raw_layers:
                weights = layer.get('weights', [])
                bias = layer.get('bias', [])
                weights_shape = tuple(layer.get('weights_shape', (0, 0)))
                output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else int(layer.get('output_dim', 0) or 0)
                input_dim = int(weights_shape[1]) if len(weights_shape) > 1 else int(layer.get('input_dim', 0) or 0)

                encrypted_weight_rows = []
                for row_idx in range(output_dim):
                    row_start = row_idx * input_dim
                    row_end = row_start + input_dim
                    weight_row = [float(value) for value in weights[row_start:row_end]]
                    if len(weight_row) < input_dim:
                        weight_row.extend([0.0] * (input_dim - len(weight_row)))
                    try:
                        encrypted_row = system_he.encrypt(weight_row)
                        encrypted_weight_rows.append(encrypted_row)
                    except Exception as e:
                        print(f"     Failed to encrypt weights for layer {layer.get('layer_idx', 0)} row {row_idx}: {e}")
                        encrypted_weight_rows.append(None)

                encrypted_bias_vector = None
                bias_values = [float(value) for value in bias]
                if bias_values:
                    try:
                        encrypted_bias_vector = system_he.encrypt(bias_values)
                    except Exception as e:
                        print(f"     Failed to encrypt bias vector for layer {layer.get('layer_idx', 0)}: {e}")

                encrypted_bias = []
                if encrypted_bias_vector is None:
                    for b in bias_values:
                        try:
                            encrypted_b = system_he.encrypt([float(b)])
                            encrypted_bias.append(encrypted_b)
                        except Exception as e:
                            print(f"     Failed to encrypt bias for layer {layer.get('layer_idx', 0)}: {e}")
                            encrypted_bias.append(None)

                encrypted_layers.append({
                    'layer_idx': layer.get('layer_idx', len(encrypted_layers)),
                    'layer_type': layer.get('layer_type', 'linear'),
                    'activation': layer.get('activation', 'linear'),
                    'weights_shape': (output_dim, input_dim),
                    'bias_shape': tuple(layer.get('bias_shape', (output_dim,))),
                    'encrypted_weights': [],
                    'encrypted_bias': encrypted_bias,
                    'encrypted_weight_rows': encrypted_weight_rows,
                    'encrypted_bias_vector': encrypted_bias_vector
                })

            encrypted_model = {
                'type': 'neural_network',
                'layer_count': len(encrypted_layers),
                'layers': encrypted_layers
            }
            
            C_M['encrypted_model'] = encrypted_model
            C_M['model_type'] = model_info['model_type']
            C_M['model_name'] = model_info['model_name']
            
            # Store the encrypted model
            enc_id = f"enc_{model_id}_{int(time.time())}"
            self.encrypted_models[enc_id] = encrypted_model
            
            print(f"     Encrypted model prepared")
            print(f"      Encrypted layer count: {len(encrypted_layers)}")
            print(f"      Valid encrypted weight rows: {sum(len([w for w in layer.get('encrypted_weight_rows', []) if w]) for layer in encrypted_layers)}")
            print(f"      Valid encrypted biases: {sum(len([b for b in layer.get('encrypted_bias', []) if b]) for layer in encrypted_layers)}")
            
            return C_M
        else:
            print(f"     Unsupported model type: {model_info['model_type']}")
            return None
    
    def query_with_model(self,
                        database_server,
                        owner_id: int,
                        dataset_id: str,
                        model_id: str) -> Optional[List[float]]:
        print(f"\n" + "="*60)
        print(f"[Data Querier {self.querier_id}] Querying with pretrained model")
        print(f"="*60)
        print(f"   Owner: {owner_id}")
        print(f"   Dataset: {dataset_id}")
        print(f"   Model ID: {model_id}")
        
        # 1. Check revocation status
        if self.check_revoked():
            print(f"     Query failed: user has been revoked")
            return None
        
        try:
            # 2. Fetch the encrypted dataset
            print(f"\n1. Fetch the encrypted dataset...")
            C_m, sk_h_s = database_server.get_dataset(owner_id, dataset_id)
            if C_m is None:
                print(f"     Dataset does not exist")
                return None
            
            # 3. Prepare the encrypted model
            print(f"\n2. Prepare the encrypted model...")
            C_M = self.prepare_encrypted_model(model_id, C_m)
            if C_M is None:
                print(f"     Model preparation failed")
                return None
            
            # 4. Execute the encrypted query
            print(f"\n3. Execute the encrypted query...")
            ER = self.key_curator.system.query(C_M, C_m, sk_h_s)
            
            # 5. Decrypt the results
            print(f"\n4. Decrypt the query results...")
            results = self.key_curator.system.decrypt(C_M['sk_h_u'], ER)
            
            print(f"\n  Query completed!")
            print(f"   Result count: {len(results)}")
            print(f"   Result sample: {results[:5]}")
            
            # Record query history
            self.query_history.append({
                'timestamp': time.time(),
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'model_id': model_id,
                'results_count': len(results),
                'success': True
            })
            
            return results
            
        except Exception as e:
            print(f"\n  Query failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.query_history.append({
                'timestamp': time.time(),
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'model_id': model_id,
                'error': str(e),
                'success': False
            })
            
            return None
    
    
    def query(self, 
             database_server,
             owner_id: int,
             dataset_id: str,
             model: Optional[List[float]] = None,
             model_type: str = "linear") -> Optional[List[float]]:

        print(f"\n" + "="*60)
        print(f"[Data Querier {self.querier_id}] Performing full query")
        print(f"="*60)
        print(f"   Owner: {owner_id}")
        print(f"   Dataset: {dataset_id}")
        
        # 1. Check revocation status
        if self.check_revoked():
            print(f"     Query failed: user has been revoked")
            return None
        
        try:
            # 2. Fetch the encrypted dataset
            print(f"\n1. Fetch the encrypted dataset...")
            C_m, sk_h_s = database_server.get_dataset(owner_id, dataset_id)
            if C_m is None:
                print(f"     Dataset does not exist")
                return None
            
            # 3. Check access permissions
            print(f"\n2. Check access permissions...")
            C_M = self.check_access(C_m)
            if C_M is None:
                print(f"     No permission to access this dataset")
                return None
            
            # 4. Create and encrypt the AI model
            print(f"\n3. Prepare the AI model...")
            if model is None:
                # Automatically create a model based on the dataset dimension
                if C_m.get('c6_i') and len(C_m['c6_i']) > 0:
                    try:
                        sample_data = self.he.decrypt(C_m['c6_i'][0])
                        dim = len(sample_data)
                    except:
                        dim = 5
                else:
                    dim = 5
                
                model = self.create_ai_model(model_type, dim)
            
            C_M = self.encrypt_ai_model(model, C_M)
            
            # 5. Execute the encrypted query
            print(f"\n4. Execute the encrypted query...")
            ER = self.key_curator.system.query(C_M, C_m, sk_h_s)
            
            # 6. Decrypt the results
            print(f"\n5. Decrypt the query results...")
            results = self.key_curator.system.decrypt(C_M['sk_h_u'], ER)
            
            print(f"\n  Query completed!")
            print(f"   Result count: {len(results)}")
            print(f"   Result sample: {results[:5]}")
            
            # Record query history
            self.query_history.append({
                'timestamp': time.time(),
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'model_type': model_type,
                'results_count': len(results),
                'success': True
            })
            
            return results
            
        except Exception as e:
            print(f"\n  Query failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.query_history.append({
                'timestamp': time.time(),
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'error': str(e),
                'success': False
            })
            
            return None
    
    # ========== Batch queries ==========
    
    def batch_query(self,
                   database_server,
                   queries: List[Tuple[int, str]],
                   model_type: str = "linear") -> List[Optional[List[float]]]:
        if not self._check_before_operation("batch query"):
            return [None] * len(queries)
        
        print(f"\n[Data Querier {self.querier_id}] Running {len(queries)} batch queries")
        
        results = []
        for owner_id, dataset_id in queries:
            result = self.query(
                database_server,
                owner_id,
                dataset_id,
                model=None,
                model_type=model_type
            )
            results.append(result)
        
        success_count = sum(1 for r in results if r is not None)
        print(f"\n  Batch query completed: {success_count}/{len(queries)} succeeded")
        
        return results
    
    # ========== Revocation notification handling ==========
    
    def on_user_revoked(self):
        print(f"\n[Data Querier {self.querier_id}] ⚠️ Received revocation notice")
        print(f"   You have been revoked by the system and can no longer perform queries")
        
        for handler in self._revoke_handlers:
            try:
                handler(self.querier_id)
            except Exception as e:
                print(f"    Handler execution failed: {e}")
    
    def register_revoke_handler(self, handler_func):
        self._revoke_handlers.append(handler_func)
        print(f"   [Notification] Revocation handler registered")
    
    # ========== Query history ==========
    
    def get_query_history(self, limit: int = 10) -> List[Dict]:
        return sorted(
            self.query_history[-limit:],
            key=lambda x: x['timestamp'],
            reverse=True
        )
    
    def clear_history(self):
        self.query_history = []
        print(f"[Data Querier {self.querier_id}] Query history cleared")
    
    # ========== Utility methods ==========
    
    def verify_aux_status(self) -> Dict:
        aux_list = self.key_curator.get_user_aux(self.querier_id)
        
        info = {
            'querier_id': self.querier_id,
            'block': self.block,
            'u_id_prime': self.u_id_prime,
            'aux_length': len(aux_list),
            'can_query_others': len(aux_list) > 0,
            'trusted_by': list(self.key_curator.get_trusted_by(self.querier_id)),
            'registered': self.querier_id in self.key_curator.registered_users,
            'revoked': self.key_curator.is_revoked(self.querier_id)
        }
        
        print(f"\n[Data Querier {self.querier_id}] aux status:")
        print(f"   aux length: {info['aux_length']}")
        print(f"   Can query others: {info['can_query_others']}")
        print(f"   Trusted by: {info['trusted_by']}")
        print(f"   Revoked: {info['revoked']}")
        
        return info
    
    def get_querier_info(self) -> Dict:
        return {
            'querier_id': self.querier_id,
            'scheme': self.key_curator.scheme_name,
            'block': self.block,
            'u_id_prime': self.u_id_prime,
            'registered': self.querier_id in self.key_curator.registered_users,
            'revoked': self.key_curator.is_revoked(self.querier_id),
            'aux_length': len(self.key_curator.get_user_aux(self.querier_id)),
            'query_count': len(self.query_history),
            'loaded_models': len(self.loaded_models),
            'public_key': str(self.pk_id)[:50] + '...' if self.pk_id else None
        }


# ========== Existing tests retained ==========

def test_data_querier_revoked():
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.database_server import DatabaseServer
    from schemes.decart_star import DeCartStarParams
    
    # 1. Initialize system
    print("\n1. Initialize system...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. Create users
    print("\n2. Create users...")
    users = [5, 6, 7]
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)
    
    # 3. Establish trust relationships
    print("\n3. Establish trust relationships...")
    curator.add_trust(6, 5)  # 6 trusts 5
    curator.add_trust(7, 5)  # 7 trusts 5
    
    # 4. Data owner encrypts data
    print("\n4. Data owner encrypts data...")
    owner = DataOwner(owner_id=5, key_curator=curator, scheme="decart_star")
    data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    policy = [5, 6, 7]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy, {'name': 'test'})
    
    # 5. Create the database server
    print("\n5. Create the database server...")
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(5, ds_id, C_m, sk_h_s)
    
    # 6. Create querier 6 (normal user)
    print("\n6. Create querier 6 (normal user)...")
    querier6 = DataQuerier(querier_id=6, key_curator=curator, scheme="decart_star")
    info6 = querier6.get_querier_info()
    print(f"   User 6 revoked status: {info6['revoked']}")
    assert not info6['revoked'], "User 6 should not be revoked initially"
    
    # 7. Revoke user 6
    print("\n7. Revoke user 6...")
    curator.revoke_user(6)
    
    # 8. Verify user 6 is revoked
    print("\n8. Verify user 6 status...")
    info6_after = querier6.get_querier_info()
    print(f"   User 6 revoked status: {info6_after['revoked']}")
    assert info6_after['revoked'], "User 6 should be marked as revoked"
    
    # 9. Try creating a new querier 6 (expected to fail)
    print("\n9. Try creating a new querier 6 (expected failure)...")
    try:
        querier6_new = DataQuerier(querier_id=6, key_curator=curator, scheme="decart_star")
        print(f"     Should have failed but succeeded")
        assert False, "Initializing a revoked user should fail"
    except ValueError as e:
        print(f"     Correctly rejected: {e}")
    
    # 10. Try letting revoked user 6 execute a query (expected to fail)
    print("\n10. Try letting revoked user 6 execute a query (expected failure)...")
    result = querier6.query(db_server, 5, ds_id)
    print(f"   Query result: {result is None}")
    assert result is None, "A revoked user's query should return None"
    
    print(f"\n  Data Querier revocation test passed")


def test_data_querier_normal():
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.database_server import DatabaseServer
    from schemes.decart_star import DeCartStarParams
    
    # 1. Initialize system
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. Create users
    users = [5, 6, 7]
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)
    
    # 3. Establish trust relationships
    curator.add_trust(6, 5)
    
    # 4. Data owner encrypts data
    owner = DataOwner(owner_id=5, key_curator=curator, scheme="decart_star")
    data = [[1.0, 2.0, 3.0]]
    policy = [5, 6]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy)
    
    # 5. Database server
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(5, ds_id, C_m, sk_h_s)
    
    # 6. Create querier 6
    querier = DataQuerier(querier_id=6, key_curator=curator, scheme="decart_star")
    
    # 7. Execute query
    print("\n7. Normal user executes query...")
    result = querier.query(db_server, 5, ds_id)
    
    print(f"   Query result: {result is not None}")
    assert result is not None, "The normal user's query should succeed"
    
    print(f"\n  Normal query test passed")


def test_self_query_after_revoke():
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.database_server import DatabaseServer
    from schemes.decart_star import DeCartStarParams
    
    # 1. Initialize system
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. Create user
    uid = 5
    sk, pk, pap = curator.generate_user_key(uid)
    curator.register(uid, pk, pap)
    
    # 3. Create the owner (who is also the querier)
    owner = DataOwner(owner_id=uid, key_curator=curator, scheme="decart_star")
    data = [[1.0, 2.0, 3.0]]
    policy = [uid]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy)
    
    # 4. Database server
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(uid, ds_id, C_m, sk_h_s)
    
    # 5. Create the querier (same user)
    querier = DataQuerier(querier_id=uid, key_curator=curator, scheme="decart_star")
    
    # 6. Revoke the user
    print("\n6. Revoke user 5...")
    curator.revoke_user(uid)
    
    # 7. Try self-query after revocation (expected to fail)
    print("\n7. Try self-query after revocation (expected failure)...")
    result = querier.query(db_server, uid, ds_id)
    print(f"   Query result: {result is None}")
    assert result is None, "Self-query after revocation should fail"
    
    print(f"\n  Self-query revocation test passed")


# ========== New: pretrained model tests ==========

def test_pretrained_models():
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.database_server import DatabaseServer
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
    
    # Create MNIST-format test data (784 dimensions)
    data = [np.random.randn(784).tolist() for _ in range(3)]
    policy = [owner_id, querier_id]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy, {'name': 'mnist_test'})
    
    # 5. Create the database server
    print("\n5. Create the database server...")
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(owner_id, ds_id, C_m, sk_h_s)
    
    # 6. Create the querier
    print("\n6. Create the querier...")
    querier = DataQuerier(querier_id=querier_id, key_curator=curator, scheme="decart_star")
    
    # 7. Set the model directory
    models_dir = r"E:\decart\experiments\models\trained"
    print(f"\n7. Model directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"     Model directory does not exist; skipping test")
        return
    
    # 8. Load all models
    print(f"\n8. Load all models...")
    model_map = querier.load_all_models_from_dir(models_dir)
    
    if not model_map:
        print(f"     No models were loaded")
        return
    
    print(f"\n   Loaded models: {list(model_map.keys())}")
    
    # 9. Query with each model type
    print("\n" + "="*60)
    print("9. Test each model query")
    print("="*60)
    
    results = {}
    
    for model_type, model_id in model_map.items():
        print(f"\n{'-'*50}")
        print(f"Testing {model_type} model query")
        print(f"{'-'*50}")
        
        try:
            query_result = querier.query_with_model(
                db_server,
                owner_id,
                ds_id,
                model_id
            )
            
            if query_result is not None:
                results[model_type] = True
                print(f"     {model_type} query succeeded")
                print(f"      Result: {query_result[:3]}")
            else:
                results[model_type] = False
                print(f"     {model_type} query failed")
                
        except Exception as e:
            results[model_type] = False
            print(f"     {model_type} query exception: {e}")
    
    # 10. Summarize results
    
    all_passed = True
    for model_type, passed in results.items():
        status = "  Passed" if passed else "  Failed"
        print(f"   {status} - {model_type}")
        all_passed = all_passed and passed
    
    if all_passed:
        print(f"\n  All pretrained model query tests passed!")
    else:
        print(f"\n  Pretrained model query tests failed")


if __name__ == "__main__":
    
    # Run the original tests
    test_data_querier_normal()
    test_data_querier_revoked()
    test_self_query_after_revoke()
    
    # Run the new pretrained model tests
    test_pretrained_models()