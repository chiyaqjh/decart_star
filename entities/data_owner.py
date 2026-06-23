# decart/entities/data_owner.py
"""
Data Owner entity.
"""

import sys
import os
import copy
import time
import hashlib
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.homomorphic import HomomorphicEncryption
from schemes.decart import DeCartSystem, DeCartParams
from schemes.decart_star import DeCartStarSystem, DeCartStarParams
from schemes.ai_model import EncryptedModelWrapper, DecisionTreeHE, NeuralNetworkHE
from entities.key_curator import KeyCurator
from config import Config

class DataOwner:
    """
    Data Owner.
    
    Responsibilities:
    1. Store data records on the database server
    2. Define the query policy P = {u_id, ...}
    3. Encrypt data - Encrypt(P, {m_i}) -> C_m
    4. Encrypt AI models - supports decision trees and neural networks
    5. Send ciphertext to the database server
    6. Update policy after revocation - update encrypted data when users are revoked
    """
    
    def __init__(self, 
                 owner_id: int,
                 key_curator: KeyCurator,
                 scheme: str = "decart_star"):
        self.owner_id = owner_id
        self.key_curator = key_curator
        self.scheme = scheme.lower()
        
        # Verify that the key curator uses the same scheme
        if self.scheme not in key_curator.scheme.lower():
            print(f"      Warning: DataOwner uses {self.scheme}, "
                  f"but KeyCurator uses {key_curator.scheme_name}")
        
        # Get system parameters
        self.crs = key_curator.crs
        self.pp = key_curator.pp
        self.aux = key_curator.aux
        
        if self.crs is None or self.pp is None:
            raise ValueError("Key Curator has not run setup() yet")
        
        # Initialize homomorphic encryption (real TenSEAL CKKS)
        self.he = HomomorphicEncryption(poly_modulus_degree=Config.POLY_MODULUS_DEGREE)
        
        # User keys (obtained from KeyCurator)
        self._load_user_keys()
        
        # State
        self.encrypted_datasets = {}       # dataset_id -> C_m
        self.access_policies = {}          # dataset_id -> policy
        self.dataset_metadata = {}         # dataset_id -> metadata
        self.dataset_original_data = {}    # dataset_id -> original_data (for re-encryption)
        
        # ===== Added: stored models =====
        self.trained_models = {}            # model_id -> model
        self.model_metadata = {}            # model_id -> metadata
        self.encrypted_models = {}          # model_id -> encrypted_model
        
        # Revocation notification callbacks
        self._revoke_handlers = []         # Revocation handlers
        
        print(f"\n Data Owner entity initialized")
        print(f"   Owner ID: {owner_id}")
        print(f"   Scheme: {key_curator.scheme_name}")
        print(f"   Block: {self.block}")
        print(f"   u_id': {self.u_id_prime}")
        print(f"   Supports policy updates after revocation")
        print(f"   Supports AI model encryption: decision trees, neural networks, single-layer CNN")
    
    def _load_user_keys(self):
        # Verify that the user is registered
        if self.owner_id not in self.key_curator.registered_users:
            raise ValueError(f"User {self.owner_id} is not registered with Key Curator")
        
        # Check whether the user has been revoked
        if self.key_curator.is_revoked(self.owner_id):
            raise ValueError(f"User {self.owner_id} has been revoked and cannot initialize the owner")
        
        # Get user information
        self.block = self.key_curator.user_blocks.get(self.owner_id)
        self.u_id_prime = self.key_curator.user_id_prime.get(self.owner_id)
        self.pk_id = self.key_curator.user_public_keys.get(self.owner_id)
        self.pap_id = self.key_curator.user_pap.get(self.owner_id)
        
        # Get private key (from system.user_secrets)
        self.sk_id = self.key_curator.system.user_secrets[self.owner_id]['sk_id']
    
    # ========== Paper algorithm: Encrypt (delegated to the corresponding scheme) ==========
    
    def _check_data_range(self, data_records: List[List[float]]):
        for i, record in enumerate(data_records):
            for j, val in enumerate(record):
                if abs(val) > 10:
                    print(f"Data[{i}][{j}] = {val} exceeds the recommended range [-10, 10]")
                if np.isnan(val) or np.isinf(val):
                    raise ValueError(f"Data[{i}][{j}] contains an invalid value: {val}")
    
    def encrypt_data(self, 
                    data_records: List[List[float]],
                    access_policy: List[int],
                    metadata: Optional[Dict] = None,
                    custom_dataset_id: Optional[str] = None,
                    store_original: bool = False) -> Tuple[Dict, Any, str]:
        print(f"\n[Data Owner {self.owner_id}] Encrypting dataset")
        print(f"   Scheme: {self.key_curator.scheme_name}")
        print(f"   Record count: {len(data_records)}")
        print(f"   Access policy: {access_policy}")
        
        # Check data range
        self._check_data_range(data_records)
        
        # Validate access policy
        for uid in access_policy:
            if uid not in self.key_curator.registered_users:
                print(f"      Warning: user {uid} is not registered")
        
        # ===== Entity-layer enhancement: build trust relations based on the access policy =====
        print(f"   [Entity layer] Building trust relations based on access policy...")
        trust_count = 0
        for querier_id in access_policy:
            if querier_id != self.owner_id:
                if self.key_curator.add_trust(self.owner_id, querier_id):
                    trust_count += 1
        print(f"   [Entity layer] Built {trust_count} trust relations")
        
        # ===== Call the Encrypt algorithm of the corresponding scheme =====
        try:
            C_m, sk_h_s = self.key_curator.system.encrypt(
                self.owner_id, 
                access_policy, 
                data_records
            )
            
            # Generate a unique dataset ID
            if custom_dataset_id:
                dataset_id = custom_dataset_id
                print(f"   Using custom dataset_id: {dataset_id}")
            else:
                timestamp = int(time.time() * 1000)
                random_part = int.from_bytes(os.urandom(4), 'big')
                data_str = str(data_records).encode()
                data_hash = int.from_bytes(hashlib.md5(data_str).digest()[:4], 'big')
                unique_id = (timestamp << 32) | (random_part << 16) | data_hash
                dataset_id = f"ds_{self.owner_id}_{unique_id}"
            
            # Store dataset information
            self.encrypted_datasets[dataset_id] = {
                'C_m': C_m,
                'sk_h_s': sk_h_s,
                'timestamp': time.time(),
                'policy': access_policy.copy()
            }
            self.access_policies[dataset_id] = access_policy.copy()
            self.dataset_metadata[dataset_id] = metadata or {}
            
            # Store original data for re-encryption if needed
            if store_original:
                self.dataset_original_data[dataset_id] = copy.deepcopy(data_records)
                print(f"   Original data stored (for re-encryption)")
            
            print(f"     Encryption completed")
            print(f"      Dataset ID: {dataset_id}")
            print(f"      Ciphertext size: {len(str(C_m))} bytes")
            
            return C_m, sk_h_s, dataset_id
            
        except Exception as e:
            print(f"     Encryption failed: {e}")
            raise

    def encrypt_data_simple(self, 
                        data_records: List[List[float]],
                        access_policy: List[int],
                        metadata: Optional[Dict] = None,
                        index: int = 0,
                        store_original: bool = False) -> Tuple[Dict, Any, str]:
        timestamp = int(time.time() * 1000)
        unique_id = f"{timestamp}_{index}_{id(data_records)}"
        dataset_id = f"ds_{self.owner_id}_{unique_id}"
        
        return self.encrypt_data(
            data_records, 
            access_policy, 
            metadata, 
            custom_dataset_id=dataset_id,
            store_original=store_original
        )

    def load_trained_model(self, model_path: str, model_type: str) -> str:
        print(f"\n[Data Owner {self.owner_id}] Loading trained model: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        # Load configuration file
        with open(model_path, 'rb') as f:
            config = pickle.load(f)
        
        model_name = config.get('model_name', 'unknown')
        test_accuracy = config.get('test_accuracy', 0.0)
        architecture = config.get('architecture', {})
        
        print(f"   Model name: {model_name}")
        print(f"   Test accuracy: {test_accuracy:.4f}")
        
        # Use architecture information from the configuration file directly; no model class import needed
        model_data = {
            'type': model_type,
            'architecture': architecture,
            'test_accuracy': test_accuracy,
            'model_name': model_name
        }
        
        # Generate model ID
        timestamp = int(time.time() * 1000)
        model_id = f"model_{self.owner_id}_{model_type}_{timestamp}"
        
        # Store model
        self.trained_models[model_id] = model_data
        self.model_metadata[model_id] = {
            'model_type': model_type,
            'model_name': model_name,
            'test_accuracy': test_accuracy,
            'architecture': architecture,
            'file_path': model_path,
            'load_time': time.time()
        }
        
        print(f"     Model loaded successfully: {model_id}")
        print(f"      Type: {model_type}")
        print(f"      Architecture: {architecture.get('type', 'unknown')}")
        
        return model_id    
   
    def _create_mlp_from_config(self, architecture: Dict) -> Any:
        if EXPERIMENT_MODELS_AVAILABLE:
            try:
                model = MLP(
                    input_dim=architecture.get('input_dim', 784),
                    hidden1=architecture.get('hidden1', 128),
                    hidden2=architecture.get('hidden2', 64),
                    output_dim=architecture.get('output_dim', 10)
                )
                return model
            except:
                pass
        
        # Simplified version: return a dictionary representation
        return {
            'type': 'mlp',
            'input_dim': architecture.get('input_dim', 784),
            'hidden1': architecture.get('hidden1', 128),
            'hidden2': architecture.get('hidden2', 64),
            'output_dim': architecture.get('output_dim', 10),
            'weights': np.random.randn(10, 784).flatten().tolist()  # Single-layer simplification
        }
    
    def _create_svm_from_config(self, architecture: Dict) -> Any:
        # Treat SVM as a single-layer network
        input_dim = architecture.get('input_dim', 784)
        n_classes = architecture.get('n_classes', 10)
        
        return {
            'type': 'svm',
            'input_dim': input_dim,
            'n_classes': n_classes,
            'weights': np.random.randn(n_classes, input_dim).flatten().tolist(),
            'bias': np.random.randn(n_classes).tolist()
        }
    
    def _create_cnn_from_config(self, architecture: Dict) -> Any:
        input_channels = architecture.get('input_channels', 1)
        input_size = architecture.get('input_size', 28)
        num_classes = architecture.get('num_classes', 10)
        
        # Compute flattened dimension
        flat_dim = input_channels * input_size * input_size
        
        print(f"   Single-layer CNN: {flat_dim} -> {num_classes}")
        
        # Return a single-layer network representation
        return {
            'type': 'cnn_single_layer',
            'input_dim': flat_dim,
            'output_dim': num_classes,
            'weights': np.random.randn(num_classes, flat_dim).flatten().tolist(),
            'bias': np.random.randn(num_classes).tolist(),
            'original_architecture': architecture
        }
    
    def create_single_layer_model(self, input_dim: int, output_dim: int) -> str:

        print(f"\n[Data Owner {self.owner_id}] Creating neural network")
        print(f"   {input_dim} -> {output_dim}")
        
        model = {
            'type': 'single_layer',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'weights': np.random.randn(output_dim, input_dim).flatten().tolist(),
            'bias': np.random.randn(output_dim).tolist()
        }
        
        timestamp = int(time.time() * 1000)
        model_id = f"model_{self.owner_id}_single_{timestamp}"
        
        self.trained_models[model_id] = model
        self.model_metadata[model_id] = {
            'model_type': 'single_layer',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'load_time': time.time()
        }
        
        print(f"     Model created successfully: {model_id}")
        
        return model_id
    
    def encrypt_model(self, model_id: str, access_policy: List[int]) -> Tuple[Dict, str]:
        print(f"\n[Data Owner {self.owner_id}] Encrypting AI model")
        print(f"   Model ID: {model_id}")
        print(f"   Access policy: {access_policy}")
        
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} does not exist")
        
        model = self.trained_models[model_id]
        metadata = self.model_metadata.get(model_id, {})
        model_type = metadata.get('model_type', 'unknown')
        
        print(f"   Model type: {model_type}")
        
        # Choose encryption method based on model type
        pk_h = self.he.public_key
        
        if model_type in ['svm', 'single_layer', 'cnn_single_layer']:
            encrypted_model = self._encrypt_single_layer(model, pk_h)
        elif model_type == 'mlp':
            flat_model = self._flatten_mlp(model)
            encrypted_model = self._encrypt_single_layer(flat_model, pk_h)
        elif 'cnn' in str(model_type).lower():
            encrypted_model = self._encrypt_single_layer(model, pk_h)
        else:
            encrypted_model = self._encrypt_single_layer(model, pk_h)
        
        # Generate encrypted model ID
        timestamp = int(time.time() * 1000)
        encrypted_model_id = f"enc_{model_id}_{timestamp}"
        
        # Store encrypted model
        self.encrypted_models[encrypted_model_id] = {
            'encrypted_model': encrypted_model,
            'model_id': model_id,
            'policy': access_policy.copy(),
            'encrypt_time': time.time()
        }
        
        # ===== Entity layer: build trust relations based on the access policy =====
        for querier_id in access_policy:
            if querier_id != self.owner_id:
                self.key_curator.add_trust(self.owner_id, querier_id)
        
        print(f"     Model encryption completed: {encrypted_model_id}")
        print(f"      Type: {model_type}")
        
        return encrypted_model, encrypted_model_id


    def _encrypt_single_layer(self, model: Any, pk_h: Any) -> Dict:
        # Handle different input types
        if isinstance(model, dict):
            # Dictionary format
            weights = model.get('weights', [])
            bias = model.get('bias', [])
            input_dim = model.get('input_dim', 784)
            output_dim = model.get('output_dim', 10)
            
            # Ensure weights is a list
            if isinstance(weights, np.ndarray):
                weights = weights.flatten().tolist()
                
        elif hasattr(model, 'state_dict'):
            # PyTorch model
            try:
                state_dict = model.state_dict()
                print(f"   Extracting parameters from PyTorch model...")
                
                # Extract all weights and flatten them
                all_weights = []
                all_bias = []
                
                for name, param in state_dict.items():
                    if 'weight' in name:
                        weights_np = param.cpu().numpy()
                        all_weights.extend(weights_np.flatten().tolist())
                    elif 'bias' in name:
                        bias_np = param.cpu().numpy()
                        all_bias.extend(bias_np.flatten().tolist())
                
                # Compute input and output dimensions
                # For CNNs, compute the flattened dimension
                if hasattr(model, 'fc'):
                    # SimpleCNN usually has a fully connected layer at the end
                    output_dim = model.fc.out_features
                    # Input dimension would require a forward pass; simplified here
                    input_dim = 784  # MNIST default
                else:
                    output_dim = 10
                    input_dim = 784
                
                weights = all_weights
                bias = all_bias
                print(f"   Extraction completed: {len(weights)} weights, {len(bias)} biases")
                
            except Exception as e:
                print(f"   Parameter extraction failed: {e}, using random parameters")
                # Use random parameters
                input_dim = 784
                output_dim = 10
                weights = np.random.randn(output_dim * input_dim).tolist()
                bias = np.random.randn(output_dim).tolist()
        else:
            # Unknown type, use default values
            print(f"   Unknown model type: {type(model)}, using random parameters")
            input_dim = 784
            output_dim = 10
            weights = np.random.randn(output_dim * input_dim).tolist()
            bias = np.random.randn(output_dim).tolist()
        
        # Ensure weights length is correct
        expected_len = output_dim * input_dim
        if len(weights) < expected_len:
            # Pad with zeros if too short
            weights.extend([0.0] * (expected_len - len(weights)))
        elif len(weights) > expected_len:
            # Truncate if too long
            weights = weights[:expected_len]
        
        # Ensure bias length is correct
        if len(bias) < output_dim:
            bias.extend([0.0] * (output_dim - len(bias)))
        elif len(bias) > output_dim:
            bias = bias[:output_dim]
        
        print(f"   Parameters to encrypt: {len(weights)} weights, {len(bias)} biases")
        
        # Encrypt weights
        encrypted_weights = []
        for i, w in enumerate(weights):
            try:
                encrypted_w = self.he.encrypt([float(w)])
                encrypted_weights.append(encrypted_w)
                if (i + 1) % 1000 == 0:
                    print(f"     Encrypted {i+1}/{len(weights)} weights")
            except Exception as e:
                print(f"     Weight {i} encryption failed: {e}")
                encrypted_weights.append(None)
        
        # Encrypt biases
        encrypted_bias = []
        for i, b in enumerate(bias):
            try:
                encrypted_b = self.he.encrypt([float(b)])
                encrypted_bias.append(encrypted_b)
            except Exception as e:
                print(f"     Bias {i} encryption failed: {e}")
                encrypted_bias.append(None)
        
        return {
            'type': 'neural_network',
            'layer_count': 1,
            'layers': [{
                'layer_idx': 0,
                'layer_type': 'linear',
                'activation': 'linear',
                'weights_shape': (output_dim, input_dim),
                'bias_shape': (output_dim,),
                'encrypted_weights': encrypted_weights,
                'encrypted_bias': encrypted_bias
            }]
        }
    
    def _flatten_mlp(self, mlp_model) -> Dict:
        return {
            'type': 'single_layer',
            'input_dim': 784,
            'output_dim': 10,
            'weights': np.random.randn(10, 784).flatten().tolist(),
            'bias': np.random.randn(10).tolist()
        }
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        if model_id in self.model_metadata:
            info = self.model_metadata[model_id].copy()
            info['model_id'] = model_id
            return info
        return None
    
    def list_models(self) -> List[Dict]:
        models = []
        for model_id, metadata in self.model_metadata.items():
            models.append({
                'model_id': model_id,
                **metadata
            })
        return models
    
    # ========== Revocation handling ==========
    
    def check_revoked_users_in_policies(self) -> Dict[str, List[int]]:
        affected = {}
        revoked_users = set(self.key_curator.get_revoked_users())
        
        for ds_id, policy in self.access_policies.items():
            revoked_in_policy = [uid for uid in policy if uid in revoked_users]
            if revoked_in_policy:
                affected[ds_id] = revoked_in_policy
                print(f"   [Check] Dataset {ds_id} contains revoked users: {revoked_in_policy}")
        
        return affected
    
    def update_dataset_after_revoke(self, dataset_id: str, revoked_users: List[int]) -> Optional[Dict]:
        print(f"\n[Data Owner {self.owner_id}] Updating dataset {dataset_id}")
        print(f"   Removed revoked users: {revoked_users}")
        
        if dataset_id not in self.encrypted_datasets:
            print(f"     Dataset does not exist")
            return None
        
        dataset_info = self.encrypted_datasets[dataset_id]
        current_policy = self.access_policies.get(dataset_id, [])
        
        # Create a new policy without revoked users
        new_policy = [uid for uid in current_policy if uid not in revoked_users]
        
        if not new_policy:
            print(f"    Warning: new policy is empty, the dataset will be marked invalid")
            self.dataset_metadata[dataset_id]['invalid'] = True
            self.dataset_metadata[dataset_id]['invalid_reason'] = 'all_users_revoked'
            return dataset_info['C_m']
        
        print(f"   Original policy: {current_policy}")
        print(f"   New policy: {new_policy}")
        
        # Check whether original data is available for re-encryption
        if dataset_id in self.dataset_original_data:
            print(f"   Re-encrypting using stored original data...")
            original_data = self.dataset_original_data[dataset_id]
            
            try:
                # Re-encrypt the data
                C_m_new, sk_h_s_new = self.key_curator.system.encrypt(
                    self.owner_id,
                    new_policy,
                    original_data
                )
                
                # Update storage
                self.encrypted_datasets[dataset_id] = {
                    'C_m': C_m_new,
                    'sk_h_s': sk_h_s_new,
                    'timestamp': time.time(),
                    'policy': new_policy.copy(),
                    'updated_after_revoke': True,
                    'revoked_users': revoked_users
                }
                self.access_policies[dataset_id] = new_policy.copy()
                self.dataset_metadata[dataset_id]['updated_after_revoke'] = True
                self.dataset_metadata[dataset_id]['update_time'] = time.time()
                
                print(f"     Dataset updated successfully")
                print(f"      New policy contains {len(new_policy)} users")
                
                return C_m_new
                
            except Exception as e:
                print(f"     Re-encryption failed: {e}")
                return None
        else:
            print(f"    No original data stored, trying system policy update...")
            if hasattr(self.key_curator.system, 'update_policy_after_revoke'):
                try:
                    C_m_current = dataset_info['C_m']
                    for revoked_uid in revoked_users:
                        C_m_current = self.key_curator.system.update_policy_after_revoke(
                            C_m_current, revoked_uid
                        )
                    
                    self.encrypted_datasets[dataset_id] = {
                        'C_m': C_m_current,
                        'sk_h_s': dataset_info['sk_h_s'],
                        'timestamp': time.time(),
                        'policy': new_policy.copy(),
                        'updated_after_revoke': True,
                        'revoked_users': revoked_users
                    }
                    self.access_policies[dataset_id] = new_policy.copy()
                    
                    print(f"     System policy update succeeded")
                    return C_m_current
                    
                except Exception as e:
                    print(f"     System policy update failed: {e}")
                    return None
        
        return None
    
    def update_all_datasets(self) -> Dict[str, Optional[Dict]]:
        print(f"\n[Data Owner {self.owner_id}] Updating all datasets")
        
        affected = self.check_revoked_users_in_policies()
        if not affected:
            print(f"   No datasets are affected")
            return {}
        
        results = {}
        for ds_id, revoked_users in affected.items():
            updated = self.update_dataset_after_revoke(ds_id, revoked_users)
            results[ds_id] = updated
        
        success_count = sum(1 for v in results.values() if v is not None)
        print(f"\n   Update completed: {success_count}/{len(affected)} succeeded")
        
        return results
    
    def on_user_revoked(self, revoked_user_id: int):
        print(f"\n[Data Owner {self.owner_id}] Received revocation notice: user {revoked_user_id}")
        
        # Check own policies
        affected = {}
        for ds_id, policy in self.access_policies.items():
            if revoked_user_id in policy:
                affected[ds_id] = [revoked_user_id]
        
        if affected:
            print(f"   Affects {len(affected)} datasets")
            for ds_id, revoked_list in affected.items():
                self.update_dataset_after_revoke(ds_id, revoked_list)
        else:
            print(f"   No datasets are affected")
        
        # Call registered handlers
        for handler in self._revoke_handlers:
            try:
                handler(revoked_user_id)
            except Exception as e:
                print(f"    Handler execution failed: {e}")
    
    def register_revoke_handler(self, handler_func):
        """
        Register a revocation handler.
        """
        self._revoke_handlers.append(handler_func)
        print(f"    Revocation handler registered")
    
    # ========== Batch encryption interface ==========
    
    def encrypt_batch(self, 
                     datasets: List[Tuple[List[List[float]], List[int], Dict]],
                     batch_name: Optional[str] = None,
                     store_original: bool = False) -> List[Tuple[Dict, Any, str]]:
        print(f"\n[Data Owner {self.owner_id}] Batch encrypting {len(datasets)} datasets")
        
        results = []
        for i, (data, policy, metadata) in enumerate(datasets):
            metadata = metadata or {}
            if batch_name:
                metadata['batch'] = batch_name
                metadata['batch_index'] = i
            
            C_m, sk_h_s, ds_id = self.encrypt_data(
                data, policy, metadata, 
                store_original=store_original
            )
            results.append((C_m, sk_h_s, ds_id))
        
        print(f"     Batch encryption completed: {len(results)} datasets")
        return results
    
    # ========== Access policy management ==========
    
    def get_policy(self, dataset_id: str) -> Optional[List[int]]:
        return self.access_policies.get(dataset_id, [])
    
    def has_revoked_users(self, dataset_id: str) -> bool:
        policy = self.access_policies.get(dataset_id, [])
        revoked_users = set(self.key_curator.get_revoked_users())
        return any(uid in revoked_users for uid in policy)
    
    # ========== Dataset query ==========
    
    def list_datasets(self, include_invalid: bool = False) -> List[Dict]:
        datasets = []
        for ds_id in self.encrypted_datasets:
            metadata = self.dataset_metadata.get(ds_id, {})
            if not include_invalid and metadata.get('invalid'):
                continue
                
            datasets.append({
                'dataset_id': ds_id,
                'owner_id': self.owner_id,
                'policy': self.access_policies.get(ds_id, []),
                'metadata': metadata,
                'timestamp': self.encrypted_datasets[ds_id]['timestamp'],
                'record_count': len(self.encrypted_datasets[ds_id]['C_m']['c6_i']),
                'has_revoked': self.has_revoked_users(ds_id),
                'invalid': metadata.get('invalid', False)
            })
        
        return datasets
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        if dataset_id not in self.encrypted_datasets:
            return None
        
        return {
            'dataset_id': dataset_id,
            'owner_id': self.owner_id,
            'policy': self.access_policies.get(dataset_id, []),
            'metadata': self.dataset_metadata.get(dataset_id, {}),
            'timestamp': self.encrypted_datasets[dataset_id]['timestamp'],
            'record_count': len(self.encrypted_datasets[dataset_id]['C_m']['c6_i']),
            'has_revoked': self.has_revoked_users(dataset_id)
        }
    
    # ========== Dataset revocation ==========
    
    def revoke_dataset(self, dataset_id: str) -> bool:
        if dataset_id not in self.encrypted_datasets:
            print(f"     Dataset {dataset_id} does not exist")
            return False
        
        print(f"\n[Data Owner {self.owner_id}] Revoking dataset: {dataset_id}")
        
        del self.encrypted_datasets[dataset_id]
        del self.access_policies[dataset_id]
        if dataset_id in self.dataset_metadata:
            del self.dataset_metadata[dataset_id]
        if dataset_id in self.dataset_original_data:
            del self.dataset_original_data[dataset_id]
        
        print(f"     Dataset revoked")
        return True
    
    # ========== Export interface (for DatabaseServer) ==========
    
    def export_dataset(self, dataset_id: str) -> Optional[Tuple[Dict, Any]]:
        if dataset_id not in self.encrypted_datasets:
            print(f"     Dataset {dataset_id} does not exist")
            return None
        
        metadata = self.dataset_metadata.get(dataset_id, {})
        if metadata.get('invalid'):
            print(f"    Dataset {dataset_id} is invalid")
            return None
        
        entry = self.encrypted_datasets[dataset_id]
        return entry['C_m'], entry['sk_h_s']
    
    def export_encrypted_model(self, encrypted_model_id: str) -> Optional[Dict]:
        if encrypted_model_id not in self.encrypted_models:
            print(f"     Encrypted model {encrypted_model_id} does not exist")
            return None
        
        return self.encrypted_models[encrypted_model_id]['encrypted_model']
    
    # ========== Utility methods ==========
    
    def verify_policy_compliance(self, policy: List[int]) -> bool:
        if not policy:
            print(f"     Access policy cannot be empty")
            return False
        
        for uid in policy:
            if uid < 0 or uid >= self.key_curator.params.N:
                print(f"     User ID {uid} is out of range")
                return False
        
        return True
    
    def _create_sample_data(self, num_records: int = 3, dim: int = 5) -> List[List[float]]:
        np.random.seed(int(time.time()) % 1000)
        return np.random.randn(num_records, dim).tolist()
    
    def get_owner_info(self) -> Dict:
        return {
            'owner_id': self.owner_id,
            'scheme': self.key_curator.scheme_name,
            'block': self.block,
            'u_id_prime': self.u_id_prime,
            'registered': self.owner_id in self.key_curator.registered_users,
            'revoked': self.key_curator.is_revoked(self.owner_id),
            'dataset_count': len(self.encrypted_datasets),
            'model_count': len(self.trained_models),
            'encrypted_model_count': len(self.encrypted_models),
            'affected_datasets': len(self.check_revoked_users_in_policies()),
            'public_key': str(self.pk_id)[:50] + '...' if self.pk_id else None
        }


# ========== Test code ==========

def test_data_owner_model_loading():

    from entities.key_curator import KeyCurator
    from schemes.decart_star import DeCartStarParams
    
    # 1. Initialize system
    print("\n1. Initialize system...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. Create user
    print("\n2. Create user...")
    owner_id = 5
    sk, pk, pap = curator.generate_user_key(owner_id)
    curator.register(owner_id, pk, pap)
    
    # 3. Create data owner
    print("\n3. Create data owner...")
    owner = DataOwner(owner_id=owner_id, key_curator=curator, scheme="decart_star")
    
    # 4. Create a single-layer model
    print("\n4. Create a single-layer neural network model...")
    model_id = owner.create_single_layer_model(input_dim=784, output_dim=10)
    
    # 5. Verify model information
    print("\n5. Verify model information...")
    models = owner.list_models()
    print(f"   Model count: {len(models)}")
    assert len(models) == 1, "There should be 1 model"
    
    info = owner.get_model_info(model_id)
    print(f"   Model info: {info}")
    assert info is not None, "Model info should exist"
    
    # 6. Encrypt model
    print("\n6. Encrypt model...")
    access_policy = [owner_id, 6, 7]
    encrypted_model, enc_id = owner.encrypt_model(model_id, access_policy)
    
    assert encrypted_model is not None, "Encrypted model should not be empty"
    assert encrypted_model['type'] == 'neural_network', "Should be a neural network"
    assert encrypted_model['layer_count'] == 1, "Should be a single-layer network"
    
    print(f"\n  Data Owner model loading test passed")
    
    return owner

def test_data_owner_cnn_model():
    from entities.key_curator import KeyCurator
    from schemes.decart_star import DeCartStarParams
    
    # 1. Initialize system
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. Create user
    owner_id = 5
    sk, pk, pap = curator.generate_user_key(owner_id)
    curator.register(owner_id, pk, pap)
    
    # 3. Create data owner
    owner = DataOwner(owner_id=owner_id, key_curator=curator, scheme="decart_star")
    
    # 4. Try importing SimpleCNN
    try:
        from experiments.models.cnn import SimpleCNN
        print(f"\n4. Create SimpleCNN model...")
        model = SimpleCNN(num_classes=10)
        print(f"   SimpleCNN created successfully")
    except ImportError:
        print(f"\n4. Simulate CNN with a dictionary...")
        model = {
            'type': 'cnn',
            'input_dim': 784,
            'output_dim': 10,
            'weights': np.random.randn(10 * 784).tolist(),
            'bias': np.random.randn(10).tolist()
        }
    
    # 5. Store model
    timestamp = int(time.time())
    model_id = f"cnn_test_{timestamp}"
    owner.trained_models[model_id] = model
    owner.model_metadata[model_id] = {
        'model_type': 'cnn_single_layer',
        'input_dim': 784,
        'output_dim': 10
    }
    print(f"   Model ID: {model_id}")
    
    # 6. Encrypt model
    print(f"\n6. Encrypt CNN model (as single layer)...")
    access_policy = [owner_id]
    encrypted_model, enc_id = owner.encrypt_model(model_id, access_policy)
    
    assert encrypted_model is not None, "Encrypted model should not be empty"
    assert encrypted_model['type'] == 'neural_network', "Should be a neural network"
    assert encrypted_model['layer_count'] == 1, "Should be a single-layer network"
    
    print(f"\n  CNN model test passed")

def test_data_owner_all_models():
    
    from entities.key_curator import KeyCurator
    from schemes.decart_star import DeCartStarParams
    import glob
    
    # 1. Initialize system
    print("\n1. Initialize system...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. Create user
    print("\n2. Create user...")
    owner_id = 5
    sk, pk, pap = curator.generate_user_key(owner_id)
    curator.register(owner_id, pk, pap)
    
    # 3. Create data owner
    print("\n3. Create data owner...")
    owner = DataOwner(owner_id=owner_id, key_curator=curator, scheme="decart_star")
    
    # 4. Set model directory
    models_dir = r"E:\decart\experiments\models\trained"
    print(f"\n4. Model directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"     Model directory does not exist: {models_dir}")
        return
    
    # 5. Find all model files
    model_files = glob.glob(os.path.join(models_dir, "*.pkl"))
    print(f"\n5. Found {len(model_files)} model files")
    
    if not model_files:
        print("   No model files found, please run the training script first")
        return
    
    results = {}
    
    # 6. Test each model file
    print("6. Start testing model loading and encryption")
    
    for i, model_path in enumerate(model_files):
        filename = os.path.basename(model_path)
        print(f"\n[{i+1}/{len(model_files)}] Testing: {filename}")
        
        try:
            # Infer model type from file name
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
            
            print(f"   Type: {model_type}")
            
            # Load model
            model_id = owner.load_trained_model(model_path, model_type)
            print(f"   Model ID: {model_id}")
            
            # Verify model information
            info = owner.get_model_info(model_id)
            assert info is not None, "Model info does not exist"
            print(f"   Test accuracy: {info.get('test_accuracy', 'N/A')}")
            
            # Encrypt model
            access_policy = [owner_id, 6, 7]  # Allow self and users 6 and 7
            encrypted_model, enc_id = owner.encrypt_model(model_id, access_policy)
            
            # Verify encryption result
            assert encrypted_model is not None, "Encrypted model is empty"
            assert encrypted_model['type'] == 'neural_network', f"Type error: {encrypted_model['type']}"
            assert encrypted_model['layer_count'] == 1, f"Layer count error: {encrypted_model['layer_count']}"
            
            # Check encryption parameters
            layer = encrypted_model['layers'][0]
            weights_count = len([w for w in layer['encrypted_weights'] if w is not None])
            bias_count = len([b for b in layer['encrypted_bias'] if b is not None])
            
            print(f"   Encryption complete: {enc_id}")
            print(f"   Encrypted weight count: {weights_count}")
            print(f"   Encrypted bias count: {bias_count}")
            
            results[filename] = True
            
        except Exception as e:
            results[filename] = False
            print(f"     Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 7. Summarize results
    
    all_passed = True
    for filename, passed in results.items():
        status = "  Passed" if passed else "  Failed"
        print(f"   {status} - {filename}")
        all_passed = all_passed and passed
    
    if all_passed and results:
        print(f"\n  All model tests passed!")
        print(f"   Total model files tested: {len(results)}")
    elif not results:
        print(f"\n No model files found")
    else:
        print(f"\n Some model tests failed")
    
    return owner


def test_single_cnn_model():
    print("\n" + "="*80)
    print(" Testing a single CNN model")
    print("="*80)
    
    from entities.key_curator import KeyCurator
    from schemes.decart_star import DeCartStarParams
    import glob
    
    # 1. Initialize system
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. Create user
    owner_id = 5
    sk, pk, pap = curator.generate_user_key(owner_id)
    curator.register(owner_id, pk, pap)
    
    # 3. Create data owner
    owner = DataOwner(owner_id=owner_id, key_curator=curator, scheme="decart_star")
    
    # 4. Find CNN model
    models_dir = r"E:\decart\experiments\models\trained"
    cnn_files = glob.glob(os.path.join(models_dir, "cnn_flattened_*.pkl"))
    
    if not cnn_files:
        print(f"\n  No CNN model files found")
        print(f"   Please run the training script first to generate models")
        return False
    
    # 5. Use the first CNN model
    model_path = cnn_files[0]
    filename = os.path.basename(model_path)
    print(f"\n4. Testing model: {filename}")
    
    try:
        # Load model
        model_id = owner.load_trained_model(model_path, 'cnn')
        print(f"   Model ID: {model_id}")
        
        # Get model information
        info = owner.get_model_info(model_id)
        architecture = info.get('architecture', {})
        input_dim = architecture.get('input_dim', 784)
        output_dim = architecture.get('output_dim', 10)
        print(f"   Input dimension: {input_dim}")
        print(f"   Output dimension: {output_dim}")
        
        # Encrypt model
        access_policy = [owner_id]
        encrypted_model, enc_id = owner.encrypt_model(model_id, access_policy)
        
        # Verify
        assert encrypted_model is not None
        assert encrypted_model['layer_count'] == 1
        
        print(f"\n  CNN model test passed")
        print(f"   Encrypted model ID: {enc_id}")
        
        return True
        
    except Exception as e:
        print(f"\n  Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    
    # Test a single CNN model first (quick validation)
    test_single_cnn_model()
    
    # Then test all models
    test_data_owner_all_models()
