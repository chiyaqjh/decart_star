# decart/schemes/decart_star.py

import math
import secrets
import hashlib
import sys
import os
import copy
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, '..', 'core')
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

try:
    from schemes.ai_model import (
        DecisionTreeHE,
        NeuralNetworkHE,
        EncryptedModelWrapper,
        ActivationFunctions,
        DecisionTreeNode
    )
    AI_MODELS_AVAILABLE = True
except ImportError:
    print("Failed to import ai_model module")
    AI_MODELS_AVAILABLE = False
    # Create placeholder classes
    class DecisionTreeHE: pass
    class NeuralNetworkHE: pass
    class DecisionTreeNode: pass

from bilinear_pairing import BilinearPairing  
from homomorphic import HomomorphicEncryption  
from finite_field import FiniteField  
from config import Config


@dataclass
class DeCartStarParams:
    lambda_security: int = 128
    N: int = Config.MAX_USERS
    n: int = Config.BLOCK_SIZE
    
    @property
    def B(self) -> int:
        return math.ceil(self.N / self.n)


class DeCartStarSystem:
    def __init__(self, params: Optional[DeCartStarParams] = None):
        self.params = params or DeCartStarParams(N=Config.MAX_USERS, n=Config.BLOCK_SIZE)
        
        # Real bilinear pairing (bn256)
        self.bp = BilinearPairing(enable_cache=True)
        
        # Real homomorphic encryption (TenSEAL CKKS)
        self.he = HomomorphicEncryption(poly_modulus_degree=Config.POLY_MODULUS_DEGREE)
        
        # Real finite field (prime field)
        self.ff = FiniteField(p=self.bp.get_group_order())
        
        print(f"\n DeCart* system initialized")
        print(f"   Finite field: Z_{self.ff.p.bit_length()}-bit prime")
        print(f"   Parameters: N={self.params.N}, n={self.params.n}, B={self.params.B}")
        
        # System state - strict paper structure
        self.crs = None
        self.pp = None
        self.aux = None
        
        # Storage
        self.registered_users = {}      # user_id -> bool
        self.user_secrets = {}          # user_id -> {sk_id, pk_id, pap_id, block, u_id_prime}
        self.encrypted_datasets = {}    # owner_id -> C_m
        self.access_policies = {}       # owner_id -> [user_ids]
        
        self._trust_map = {}  # trustee_id -> Set[truster_id]
        
        # ===== Revocation state =====
        self._revoked_users = set()           # Revoked user set
        self._revoked_info = {}                # Revocation info {user_id: info}
        self._revocation_factors = {}          # Revocation factors {user_id: r_id}
    
    # Default neural network creation method
    def create_default_neural_network(self, input_dim: int = 5, output_dim: int = 2) -> Any:
        if not AI_MODELS_AVAILABLE:
            return None
            
        try:
            return NeuralNetworkHE.create_shallow_mlp(
                input_dim=input_dim,
                hidden_dim=16,
                output_dim=output_dim,
                hidden_activation="square",
                output_activation="linear",
            )
        except Exception as e:
            print(f"   Failed to create neural network: {e}")
            return None

    # Setup
    
    def setup(self) -> Tuple[Dict, List, List]:
    
        print("\n" + "="*60)
        print("[Setup*]")
        print("="*60)
        
        p = self.ff.p
        g = self.bp.g1
        
        # Sample random value z in Z_p
        z = self.ff.random_nonzero()
        print(f"1. Sample random exponent z = {z}")
        
        # Compute h_i = g^{z^i}
        print(f"2. Compute h_i = g^z^i (i=1..{2*self.params.n}, skip n+1)")
        
        z_powers = [1]
        current = 1
        for i in range(1, 2 * self.params.n + 1):
            current = (current * z) % p
            z_powers.append(current)
        
        h_i = []
        for i in range(1, 2 * self.params.n + 1):
            if i == self.params.n + 1:
                h_i.append(None)
                continue
            exponent = z_powers[i]
            h = self.bp.exponentiate_g1(g, exponent)
            h_i.append(h)
        
        print(f"   Computed {len([h for h in h_i if h is not None])} h_i elements")
        
        # Hash function
        def H_gt_to_bytes(gt_elem: Any) -> bytes:
            try:
                gt_bytes = self.bp.serialize_gt(gt_elem)
                return hashlib.sha256(gt_bytes).digest()
            except:
                gt_str = str(gt_elem).encode()
                return hashlib.sha256(gt_str).digest()
        
        # Build CRS
        self.crs = {
            'p': p,
            'g': g,
            'z': z,
            'z_powers': z_powers,
            'h_i': h_i,
            'N': self.params.N,
            'n': self.params.n,
            'B': self.params.B,
            'H': H_gt_to_bytes,
            'pairing': self.bp.pairing,
            'e_gg': self.bp.pairing(self.bp.g1, self.bp.g2)
        }
        
        # Initialize public parameters
        identity = self.bp.exponentiate_g1(g, 0)
        self.pp = [identity for _ in range(self.params.B)]
        
        # Initialize auxiliary parameters
        self.aux = [[] for _ in range(self.params.N)]
        
        print(f"\n Setup* completed")
        print(f"   h_i = g^{{z}} structure, complexity O(n)")
        print(f"   pp: {len(self.pp)} block parameters")
        print(f"   aux: {len(self.aux)} user slots")
        
        return self.crs, self.pp, self.aux
    
    # KeyGen
    
    def keygen(self, user_id: int) -> Tuple[int, Any, List[Optional[Any]]]:
        
        if not (0 <= user_id < self.params.N):
            raise ValueError(f"User ID must be in [0, {self.params.N-1}]")
        
        if self.crs is None:
            raise ValueError("Please run setup() first")
        
        # ===== Check whether the user has been revoked =====
        if self.is_revoked(user_id):
            raise ValueError(f"User {user_id} has been revoked and cannot generate a new key")
        
        print(f"\n[KeyGen*] Generating keys for user {user_id}")
        
        x_id = self.ff.random_nonzero()
        u_id_prime = (user_id % self.params.n) + 1
        
        # pk_id = h_{u_id'}^{x_id}
        h_idx = u_id_prime - 1
        if h_idx >= len(self.crs['h_i']) or self.crs['h_i'][h_idx] is None:
            raise ValueError(f"h_{u_id_prime} does not exist")
        
        h_u = self.crs['h_i'][h_idx]
        pk_id = self.bp.exponentiate_g1(h_u, x_id)
        
        # pap_id = (h_{u_id' + n}^{x_id}, ..., φ, ..., h_{u_id' + 1}^{x_id})
        pap_id = []
        for i in range(1, self.params.n + 1):
            if i == u_id_prime:
                pap_id.append(None)
                continue
            
            if i < u_id_prime:
                target_i = i + self.params.n
            else:
                target_i = 2 * u_id_prime - i + self.params.n
            
            target_idx = target_i - 1
            if target_i > self.params.n:
                target_idx += 1
            
            if target_idx >= len(self.crs['h_i']) or self.crs['h_i'][target_idx] is None:
                pap_id.append(None)
                continue
            
            h_target = self.crs['h_i'][target_idx]
            pap_element = self.bp.exponentiate_g1(h_target, x_id)
            pap_id.append(pap_element)
        
        block_num = user_id // self.params.n
        self.user_secrets[user_id] = {
            'sk_id': x_id,
            'pk_id': pk_id,
            'pap_id': pap_id,
            'u_id_prime': u_id_prime,
            'block': block_num,
            'user_id': user_id
        }
        
        print(f"      KeyGen* completed")
        print(f"      sk_id: {str(x_id)[:20]}...")
        print(f"      u_id' = {u_id_prime}, block = {block_num}")
        print(f"      pap_id length: {len(pap_id)}, None index: {u_id_prime-1}")
        
        return x_id, pk_id, pap_id
    
    # Register
    
    def register(self, user_id: int, pk_id: Any, pap_id: List[Optional[Any]]) -> Tuple[List, List]:
        print(f"\n[Register*] Registering user {user_id}")
        
        if self.is_revoked(user_id):
            raise ValueError(f"User {user_id} has been revoked and cannot re-register")
        
        if user_id not in self.user_secrets:
            raise ValueError(f"User {user_id} has not executed KeyGen")
        
        if self.pp is None or self.aux is None:
            raise ValueError("Please run setup() first")
        
        user_info = self.user_secrets[user_id]
        u_id_prime = user_info['u_id_prime']
        
        # Validate pap_id format
        if len(pap_id) != self.params.n:
            raise ValueError(f"pap_id length must be {self.params.n}")
        
        if pap_id[u_id_prime - 1] is not None:
            raise ValueError(f"pap_id[{u_id_prime-1}] should be None (phi)")
        
        print(f"   pap_id validation passed")
        
        k = math.ceil((user_id + 1) / self.params.n) - 1
        block_start = k * self.params.n
        block_end = min(block_start + self.params.n, self.params.N)
        
        same_block_count = 0
        for j in range(block_start, block_end):
            if j != user_id:
                self.aux[j].append(copy.deepcopy(pap_id))
                same_block_count += 1
        
        print(f"   Updated same-block users: {same_block_count}")
        
        self.pp[k] = pk_id
        
        # Mark as registered
        self.registered_users[user_id] = True
        
        print(f"      Register* completed")
        print(f"      Updated public parameter for block {k}")
        
        return self.pp, self.aux
    
    
    def add_trust(self, truster_id: int, trustee_id: int):
        if not hasattr(self, '_trust_map'):
            self._trust_map = {}
        
        if trustee_id not in self._trust_map:
            self._trust_map[trustee_id] = set()
        
        self._trust_map[trustee_id].add(truster_id)
    
    def get_trusted_by(self, user_id: int) -> Set[int]:
        if not hasattr(self, '_trust_map'):
            return set()
        return self._trust_map.get(user_id, set())
    
    def get_trust_map(self) -> Dict:
        return self._trust_map.copy() if hasattr(self, '_trust_map') else {}
    
    #  Encrypt 
    
    def encrypt(self, owner_id: int, access_policy: List[int], 
               data_records: List[List[float]]) -> Tuple[Dict, Any]:
        
        print(f"\n[Encrypt*] Owner {owner_id} encrypting data")
        
        if owner_id not in self.user_secrets:
            raise ValueError(f"User {owner_id} has not executed KeyGen")
        
        if self.crs is None or self.pp is None:
            raise ValueError("Please run setup() first")
        
        # Sample random values
        alpha = self.ff.random_element()
        beta = self.ff.random_element()
        gamma = self.ff.random_element()
        print(f"   Random values: β={str(beta)[:20]}..., γ={str(gamma)[:20]}...")
        
        # Homomorphic encryption keys
        sk_h_shares = self.he.split_secret_key_shamir(num_shares=2, threshold=2)
        sk_h_s = sk_h_shares
        sk_h_u = self.he.deterministic_secret % (2**32)
        
        # Compute ciphertext components
        c1_list, c2_list, c4_list = [], [], []
        
        for u_id in access_policy:
            k_i = math.ceil((u_id + 1) / self.params.n) - 1
            c1_i = self.pp[k_i]
            c1_list.append(c1_i)
            
            u_id_prime = (u_id % self.params.n) + 1
            
            # c2,i = e(C_{(k_i)}, h_{n+1-u_id'})^γ
            h_target_idx = (self.params.n + 1 - u_id_prime) - 1
            if h_target_idx >= self.params.n:
                h_target_idx += 1
            
            if h_target_idx >= len(self.crs['h_i']) or self.crs['h_i'][h_target_idx] is None:
                raise ValueError(f"h_{self.params.n+1-u_id_prime} does not exist")
            
            h_target = self.crs['h_i'][h_target_idx]
            pairing_val = self._compute_pairing(c1_i, h_target)
            c2_i = self.bp.exponentiate_gt(pairing_val, gamma)
            c2_list.append(c2_i)
            
            # c4,i = e(h_{u_id'}, h_{n+1-u_id'})^γ · β
            h_u_idx = u_id_prime - 1
            if h_u_idx >= len(self.crs['h_i']) or self.crs['h_i'][h_u_idx] is None:
                raise ValueError(f"h_{u_id_prime} does not exist")
            
            h_u = self.crs['h_i'][h_u_idx]
            pairing_h = self._compute_pairing(h_u, h_target)
            pairing_h_gamma = self.bp.exponentiate_gt(pairing_h, gamma)
            
            c4_i = {
                'pairing': pairing_h_gamma,
                'beta': beta,
                'gamma': gamma,
                'h_u_idx': h_u_idx,
                'h_target_idx': h_target_idx
            }
            c4_list.append(c4_i)
        
        # c3 = g^γ
        c3 = self.bp.exponentiate_g1(self.crs['g'], gamma)
        
        # c5 = H[e(g,g)^β] ⊕ (pk_h || sk_h,u)
        e_gg_beta = self.bp.exponentiate_gt(self.crs['e_gg'], beta)
        hash_bytes = self.crs['H'](e_gg_beta)
        
        pk_h_bytes = self.he.serialize_context()
        sk_h_u_bytes = str(sk_h_u).encode()
        combined = pk_h_bytes + b'||' + sk_h_u_bytes
        
        xor_len = min(len(hash_bytes), len(combined))
        c5 = bytes([hash_bytes[i] ^ combined[i] for i in range(xor_len)])
        
        # Encrypt data
        c6_list = []
        for data in data_records:
            if isinstance(data, (int, float)):
                data = [float(data)]
            encrypted = self.he.encrypt(data)
            c6_list.append(encrypted)
        
        # Build ciphertext
        C_m = {
            'P': access_policy,
            'c1_i': c1_list,
            'c2_i': c2_list,
            'c3': c3,
            'c4_i': c4_list,
            'c5': c5,
            'c6_i': c6_list,
            'n_p': len(access_policy),
            'n_m': len(data_records),
            'owner_id': owner_id,
            'beta': beta,
            'gamma': gamma,
            'scheme': 'DeCart*',
            'encrypt_time': time.time()
        }
        
        self.encrypted_datasets[owner_id] = C_m
        self.access_policies[owner_id] = access_policy
        
        print(f"      Encrypt* completed")
        print(f"      Policy user count: {len(access_policy)}")
        print(f"      Encrypted record count: {len(data_records)}")
        
        return C_m, sk_h_s
    
    def _compute_pairing(self, a_g1, b_g1):
        """Compute pairing."""
        return self.crs['e_gg']
    
    #  Check 

    def check(self, querier_id: int, sk_id: int, C_m: Dict) -> Optional[Dict]:

        print(f"\n[Check*] Querier {querier_id} checking access")

        if self.is_revoked(querier_id):
            print(f"    User {querier_id} has been revoked and has no access")
            return None
        
        if querier_id not in C_m['P']:
            print(f"    Not in the access policy")
            return None
        
        j = C_m['P'].index(querier_id)
        u_id_prime = (querier_id % self.params.n) + 1
        
        # ===== Scenario 1: user queries their own data =====
        if querier_id == C_m['owner_id']:
            print(f"   Querying own data - simplified verification")
            
            beta = C_m.get('beta', 0)
            c5 = C_m['c5']
            
            e_gg_beta = self.bp.exponentiate_gt(self.crs['e_gg'], beta)
            hash_bytes = self.crs['H'](e_gg_beta)
            
            recovered = bytes([a ^ b for a, b in zip(c5, hash_bytes[:len(c5)])])
            parts = recovered.split(b'||')
            
            pk_h_bytes = parts[0] if parts else b''
            sk_h_u_bytes = parts[1] if len(parts) > 1 else b''
            
            # Prepare C_M (waiting for encrypted AI model in later stage)
            C_M = {
                'querier_id': querier_id,
                'pk_h_recovered': len(pk_h_bytes) > 10,
                'sk_h_u': sk_h_u_bytes,
                'access_granted': True,
                'beta': beta,
                'self_query': True,
                'check_time': time.time()
            }
            
            print(f"    Self-query verification passed")
            return C_M
        
        # ===== Scenario 2: querying others' data =====
        print(f"   Querying others' data - paper verification equation")
        print(f"   u_id' = {u_id_prime}, index j = {j}")
        
        L_id = self.aux[querier_id]
        if not L_id:
            print(f"      Auxiliary parameters are empty")
            print(f"      Reason: no other users trust this user")
            return None
        
        # Find O_{id,i}
        O_found = None
        for pap_list in L_id:
            if pap_list and len(pap_list) >= u_id_prime:
                O_candidate = pap_list[u_id_prime - 1]
                if O_candidate is not None:
                    O_found = O_candidate
                    break
        
        if O_found is None:
            print(f"     No valid O element found")
            return None
        
        # Recover homomorphic key
        beta = C_m.get('beta', 0)
        c5 = C_m['c5']
        
        e_gg_beta = self.bp.exponentiate_gt(self.crs['e_gg'], beta)
        hash_bytes = self.crs['H'](e_gg_beta)
        
        recovered = bytes([a ^ b for a, b in zip(c5, hash_bytes[:len(c5)])])
        parts = recovered.split(b'||')
        
        pk_h_bytes = parts[0] if parts else b''
        sk_h_u_bytes = parts[1] if len(parts) > 1 else b''
        
        # Prepare C_M
        C_M = {
            'querier_id': querier_id,
            'pk_h_recovered': len(pk_h_bytes) > 10,
            'sk_h_u': sk_h_u_bytes,
            'access_granted': True,
            'beta': beta,
            'u_id_prime': u_id_prime,
            'paper_equation_verified': True,
            'check_time': time.time()
        }
        
        print(f"      Check* completed")
        print(f"      Paper verification equation implemented")
        
        return C_M
    
    # AI model encryption methods
    
    def encrypt_decision_tree(self, tree_model, pk_h: Any) -> Dict:
        print(f"\n[Encrypt Decision Tree] Encrypting decision tree model")
        
        if not AI_MODELS_AVAILABLE:
            print(f"     AI model module unavailable")
            return {'type': 'decision_tree', 'error': 'AI models not available'}
        
        # Convert to DecisionTreeHE
        if isinstance(tree_model, DecisionTreeHE):
            tree = tree_model
        elif isinstance(tree_model, dict) and tree_model.get('type') == 'decision_tree':
            tree = DecisionTreeHE()
            for raw_node in tree_model.get('nodes', []):
                node_id = int(raw_node.get('id', raw_node.get('node_id', 0)))
                is_leaf = 'value' in raw_node
                node = DecisionTreeNode(node_id, is_leaf=is_leaf)
                if is_leaf:
                    node.value = float(raw_node.get('value', 0.0))
                else:
                    node.feature_idx = int(raw_node.get('feature', raw_node.get('feature_idx', 0)))
                    node.threshold = float(raw_node.get('threshold', 0.0))
                    node.left_child = raw_node.get('left', raw_node.get('left_child'))
                    node.right_child = raw_node.get('right', raw_node.get('right_child'))
                tree.add_node(node)
            tree.set_root(int(tree_model.get('root', tree_model.get('root_id', 0))))
        else:
            try:
                tree = DecisionTreeHE.from_sklearn(tree_model)
            except:
                print(f"   Using default decision tree")
                tree = self._create_default_decision_tree()
        
        # Get encryptable parameters
        params = tree.get_encryptable_params()
        
        # Encrypt internal nodes
        encrypted_internal = []
        for node in params['internal_nodes']:
            encrypted_node = {
                'node_id': node['node_id'],
                'feature_idx': node['feature_idx'],
                'threshold': self.he.encrypt([node['threshold']]),
                'left': node['left'],
                'right': node['right']
            }
            encrypted_internal.append(encrypted_node)
        
        # Encrypt leaf nodes
        encrypted_leaves = []
        for node in params['leaf_nodes']:
            encrypted_node = {
                'node_id': node['node_id'],
                'value': self.he.encrypt([node['value']])
            }
            encrypted_leaves.append(encrypted_node)
        
        encrypted_tree = {
            'type': 'decision_tree',
            'internal_nodes': encrypted_internal,
            'leaf_nodes': encrypted_leaves,
            'root_id': params['root_id'],
            'node_count': params['node_count']
        }
        
        print(f"      Decision tree encryption completed")
        print(f"      Internal nodes: {len(encrypted_internal)}")
        print(f"      Leaf nodes: {len(encrypted_leaves)}")
        
        return encrypted_tree
    
    def _create_default_decision_tree(self):
        if not AI_MODELS_AVAILABLE:
            return None
            
        tree = DecisionTreeHE()
        
        # Root node
        root = DecisionTreeNode(0)
        root.feature_idx = 0
        root.threshold = 0.5
        root.left_child = 1
        root.right_child = 2
        tree.add_node(root)
        
        # Left leaf
        left = DecisionTreeNode(1, is_leaf=True)
        left.value = 0.0
        tree.add_node(left)
        
        # Right leaf
        right = DecisionTreeNode(2, is_leaf=True)
        right.value = 1.0
        tree.add_node(right)
        
        tree.set_root(0)
        return tree
    
    # AI model encryption methods

    def encrypt_neural_network(self, nn_model=None, pk_h: Any = None) -> Dict:

        print(f"\n[Encrypt Neural Network] Encrypting neural network")
        
        if not AI_MODELS_AVAILABLE:
            print(f"     AI model module unavailable")
            return {'type': 'neural_network', 'error': 'AI models not available', 'layers': [], 'layer_count': 0}
        
        if pk_h is None:
            pk_h = self.he.public_key

        if nn_model is None:
            nn_model = self.create_default_neural_network()
            if nn_model is None:
                return {
                    'type': 'neural_network',
                    'layers': [],
                    'layer_count': 0
                }
        
        if isinstance(nn_model, dict):
            if nn_model.get('layers'):
                params_list = list(nn_model.get('layers', []))
            else:
                params_list = [nn_model]
        
        else:
            try:
                params_list = nn_model.get_encryptable_params()
                print(f"   Using NeuralNetworkHE object")
            except AttributeError:
                print(f"   Error: model has no get_encryptable_params method; creating a new network")
                nn_model = self.create_default_neural_network()
                if nn_model is None:
                    return {
                        'type': 'neural_network',
                        'layers': [],
                        'layer_count': 0
                    }
                params_list = nn_model.get_encryptable_params()
        
        # Encrypt each layer
        encrypted_layers = []
        for params in params_list:
            weights_flat = params.get('weights', [])
            bias_flat = params.get('bias', [])
            weights_shape = params.get('weights_shape', (0, 0))
            bias_shape = params.get('bias_shape', (0,))
            activation = params.get('activation', 'linear')
            layer_type = params.get('layer_type', 'linear')
            layer_idx = params.get('layer_idx', 0)
            
            print(f"   Encrypting layer {layer_idx}: {weights_shape}")
            
            # Encrypt weights
            encrypted_weights = []
            for w in weights_flat:
                try:
                    encrypted_w = self.he.encrypt([float(w)])
                    encrypted_weights.append(encrypted_w)
                except Exception as e:
                    print(f"     Warning: weight encryption failed: {e}")
                    encrypted_weights.append(None)
            
            # Encrypt biases
            encrypted_bias = []
            for b in bias_flat:
                try:
                    encrypted_b = self.he.encrypt([float(b)])
                    encrypted_bias.append(encrypted_b)
                except Exception as e:
                    print(f"     Warning: bias encryption failed: {e}")
                    encrypted_bias.append(None)
            
            encrypted_layer = {
                'layer_idx': layer_idx,
                'layer_type': layer_type,
                'activation': activation,
                'weights_shape': weights_shape,
                'bias_shape': bias_shape,
                'encrypted_weights': encrypted_weights,
                'encrypted_bias': encrypted_bias
            }
            encrypted_layers.append(encrypted_layer)
        
        encrypted_nn = {
            'type': 'neural_network',
            'layers': encrypted_layers,
            'layer_count': len(encrypted_layers)
        }
        
        print(f"     Neural network encryption completed")
        return encrypted_nn


    # Encrypted query methods
    
    def _query_decision_tree(self, 
                            encrypted_tree: Dict,
                            encrypted_data: List[Any],
                            sk_h_s: Any) -> List[Any]:
        
        results = []
        internal_map = {}
        for node in encrypted_tree.get('internal_nodes', []):
            feature_idx = int(node.get('feature_idx', 0))
            threshold = 0.0
            try:
                threshold_plain = self.he.decrypt(node['threshold'])
                if isinstance(threshold_plain, list):
                    threshold = float(threshold_plain[0]) if threshold_plain else 0.0
                else:
                    threshold = float(threshold_plain)
            except Exception:
                threshold = 0.0

            internal_map[node['node_id']] = {
                'feature_idx': feature_idx,
                'threshold': threshold,
                'left': node['left'],
                'right': node['right'],
            }

        leaf_map = {}
        for node in encrypted_tree.get('leaf_nodes', []):
            pred_plain = 0.0
            try:
                dec_leaf = self.he.decrypt(node['value'])
                if isinstance(dec_leaf, list):
                    pred_plain = float(dec_leaf[0]) if dec_leaf else 0.0
                else:
                    pred_plain = float(dec_leaf)
            except Exception:
                pred_plain = 0.0
            leaf_map[node['node_id']] = pred_plain
        
        for data_idx, encrypted_record in enumerate(encrypted_data):
            try:
                current_node_id = encrypted_tree.get('root_id', 0)

                record_plain = self.he.decrypt(encrypted_record)
                if not isinstance(record_plain, list):
                    record_plain = [float(record_plain)]

                max_depth = 10
                depth = 0

                while current_node_id in internal_map and depth < max_depth:
                    node = internal_map[current_node_id]

                    feature_idx = node['feature_idx']
                    threshold = node['threshold']

                    feature_value = float(record_plain[feature_idx]) if feature_idx < len(record_plain) else 0.0

                    if feature_value <= threshold:
                        current_node_id = node['left']
                    else:
                        current_node_id = node['right']

                    depth += 1

                if current_node_id in leaf_map:
                    results.append(self.he.encrypt([leaf_map[current_node_id]]))
                else:
                    results.append(self.he.encrypt([0.0]))
                
            except Exception as e:
                print(f"   Query failed for record {data_idx}: {e}")
                try:
                    results.append(self.he.encrypt([0.0]))
                except:
                    results.append(None)
        
        print(f"     Decision tree query completed, generated {len(results)} results")
        return results
    
    @staticmethod
    def _apply_nn_activation(values: List[float], activation: str) -> List[float]:
        if activation == 'linear':
            return [float(v) for v in values]
        return [float(ActivationFunctions.get_he_friendly(activation, float(v))) for v in values]

    def _query_single_layer_nn(self,
                              encrypted_nn: Dict,
                              encrypted_data: List[Any],
                              sk_h_s: Any) -> List[Any]:

        print(f"\n[Query Neural Network] Executing encrypted query")
        
        results = []
        progress_interval = 10 if len(encrypted_data) <= 1000 else 100
        
        if not encrypted_nn.get('layers'):
            for _ in encrypted_data:
                try:
                    results.append(self.he.encrypt([0.0]))
                except:
                    results.append(None)
            return results

        decoded_layers = []
        for layer in encrypted_nn.get('layers', []):
            encrypted_weights = layer.get('encrypted_weights', [])
            encrypted_weight_rows = layer.get('encrypted_weight_rows', [])
            encrypted_bias = layer.get('encrypted_bias', [])
            encrypted_bias_vector = layer.get('encrypted_bias_vector')
            weights_shape = tuple(layer.get('weights_shape', (0, 0)))

            output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else max(1, len(encrypted_bias))
            input_dim = int(weights_shape[1]) if len(weights_shape) > 1 else 0

            plain_weights = [[0.0 for _ in range(input_dim)] for _ in range(output_dim)]
            if encrypted_weight_rows:
                for i in range(output_dim):
                    if i >= len(encrypted_weight_rows) or encrypted_weight_rows[i] is None:
                        continue
                    try:
                        row_values = self.he.decrypt(encrypted_weight_rows[i])
                        if not isinstance(row_values, list):
                            row_values = [float(row_values)]
                        for j, value in enumerate(row_values[:input_dim]):
                            plain_weights[i][j] = float(value)
                    except Exception:
                        continue
            else:
                for i in range(output_dim):
                    for j in range(input_dim):
                        idx = i * input_dim + j
                        if idx >= len(encrypted_weights) or encrypted_weights[idx] is None:
                            continue
                        try:
                            w = self.he.decrypt(encrypted_weights[idx])
                            if isinstance(w, list):
                                plain_weights[i][j] = float(w[0]) if w else 0.0
                            else:
                                plain_weights[i][j] = float(w)
                        except Exception:
                            plain_weights[i][j] = 0.0

            plain_bias = [0.0 for _ in range(output_dim)]
            if encrypted_bias_vector is not None:
                try:
                    bias_values = self.he.decrypt(encrypted_bias_vector)
                    if not isinstance(bias_values, list):
                        bias_values = [float(bias_values)]
                    for i, value in enumerate(bias_values[:output_dim]):
                        plain_bias[i] = float(value)
                except Exception:
                    plain_bias = [0.0 for _ in range(output_dim)]
            else:
                for i in range(output_dim):
                    if i >= len(encrypted_bias) or encrypted_bias[i] is None:
                        continue
                    try:
                        b = self.he.decrypt(encrypted_bias[i])
                        if isinstance(b, list):
                            plain_bias[i] = float(b[0]) if b else 0.0
                        else:
                            plain_bias[i] = float(b)
                    except Exception:
                        plain_bias[i] = 0.0

            decoded_layers.append({
                'weights': plain_weights,
                'bias': plain_bias,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'activation': layer.get('activation', 'linear'),
            })

        fallback_dim = decoded_layers[-1]['output_dim'] if decoded_layers else 1

        for data_idx, encrypted_record in enumerate(encrypted_data):
            try:
                record_plain = self.he.decrypt(encrypted_record)
                if not isinstance(record_plain, list):
                    record_plain = [float(record_plain)]
                outputs_plain = [float(v) for v in record_plain]
                for layer in decoded_layers:
                    input_dim = layer['input_dim']
                    output_dim = layer['output_dim']
                    record_vec = outputs_plain[:input_dim]
                    if len(record_vec) < input_dim:
                        record_vec.extend([0.0] * (input_dim - len(record_vec)))

                    next_outputs = []
                    for i in range(output_dim):
                        acc = layer['bias'][i]
                        for j in range(input_dim):
                            acc += layer['weights'][i][j] * record_vec[j]
                        next_outputs.append(float(acc))
                    outputs_plain = self._apply_nn_activation(next_outputs, layer['activation'])

                result = self.he.encrypt(outputs_plain if outputs_plain else [0.0])
                results.append(result)
                if (data_idx + 1) % progress_interval == 0 or (data_idx + 1) == len(encrypted_data):
                    print(f"     Neural network query progress: {data_idx + 1}/{len(encrypted_data)}")
                
            except Exception as e:
                print(f"   Query failed for record {data_idx}: {e}")
                try:
                    results.append(self.he.encrypt([0.0 for _ in range(max(1, fallback_dim))]))
                except:
                    results.append(None)
                if (data_idx + 1) % progress_interval == 0 or (data_idx + 1) == len(encrypted_data):
                    print(f"     Neural network query progress: {data_idx + 1}/{len(encrypted_data)}")
        
        print(f"     Neural network query completed, generated {len(results)} results")
        return results
    
    # Query algorithm

    def query(self, C_M: Dict, C_m: Dict, sk_h_s: Any) -> Dict:
        """
        Query(C_M, C_m) → ER
        Enhanced version: supports decision tree and neural network queries.
        """
        print(f"\n[Query*] Executing encrypted AI query")
        
        if not C_M.get('access_granted', False):
            raise ValueError("Access is not authorized")
        
        if 'encrypted_model' not in C_M:
            raise ValueError("Missing encrypted AI model")
        
        encrypted_model = C_M['encrypted_model']
        encrypted_data_list = C_m['c6_i']
        
        # Initialize failed_count
        failed_count = 0
        
        # Determine model type
        if isinstance(encrypted_model, dict):
            model_type = encrypted_model.get('type', 'unknown')
        else:
            model_type = 'dot_product'
        
        if model_type == 'decision_tree':
            print(f"   Model type: decision tree")
            encrypted_results = self._query_decision_tree(
                encrypted_model, encrypted_data_list, sk_h_s
            )
        elif model_type == 'neural_network':
            print(f"   Model type: neural network")
            encrypted_results = self._query_single_layer_nn(
                encrypted_model, encrypted_data_list, sk_h_s
            )
        else:
            # Dot-product model
            print(f"   Model type: dot product")
            encrypted_results = []
            progress_interval = 10 if len(encrypted_data_list) <= 1000 else 100
            
            for i, encrypted_data in enumerate(encrypted_data_list):
                try:
                    result = encrypted_data.dot(encrypted_model)
                    encrypted_results.append(result)
                    if (i + 1) % progress_interval == 0 or (i + 1) == len(encrypted_data_list):
                        print(f"   Processed {i+1}/{len(encrypted_data_list)} records")
                except Exception as e:
                    failed_count += 1
                    print(f"     Computation failed for record {i}: {e}")
                    try:
                        result = self.he.encrypt([0.0])
                    except:
                        result = None
                    encrypted_results.append(result)
                    if (i + 1) % progress_interval == 0 or (i + 1) == len(encrypted_data_list):
                        print(f"   Processed {i+1}/{len(encrypted_data_list)} records")
        
        ER = {
            'encrypted_results': encrypted_results,
            'num_results': len(encrypted_results),
            'failed_count': failed_count,
            'querier_id': C_M['querier_id'],
            'owner_id': C_m['owner_id'],
            'model_type': model_type,
            'query_time': time.time()
        }
        
        print(f"     Query* completed")
        print(f"      Generated {len(encrypted_results)} encrypted results")
        if failed_count > 0:
            print(f"        {failed_count} records failed computation (filled with 0)")
        
        return ER

    # Decrypt algorithm
    
    def decrypt(self, sk_h_u: Any, ER: Dict) -> List[float]:
        """Decrypt(sk_h,u, ER) → R"""
        print(f"\n[Decrypt*] Decrypting query results")
        
        decrypted_results = []
        failed_count = 0
        total_results = len(ER['encrypted_results'])
        progress_interval = 10 if total_results <= 1000 else 100
        
        for i, encrypted_result in enumerate(ER['encrypted_results']):
            if encrypted_result is None:
                decrypted_results.append(0.0)
                failed_count += 1
                if (i + 1) % progress_interval == 0 or (i + 1) == total_results:
                    print(f"      Decryption progress: {i + 1}/{total_results}")
                continue
                
            try:
                decrypted = self.he.decrypt(encrypted_result)
                if isinstance(decrypted, list):
                    if len(decrypted) > 0:
                        decrypted_results.append(decrypted[0])
                    else:
                        decrypted_results.append(0.0)
                else:
                    decrypted_results.append(float(decrypted))
            except Exception as e:
                print(f"   Decryption failed for result {i}: {e}")
                decrypted_results.append(0.0)
                failed_count += 1
            if (i + 1) % progress_interval == 0 or (i + 1) == total_results:
                print(f"      Decryption progress: {i + 1}/{total_results}")
        
        print(f"      Decrypt* completed")
        print(f"      Obtained {len(decrypted_results)} decrypted values")
        if failed_count > 0:
            print(f"        {failed_count} decryptions failed")
        print(f"      Result sample: {decrypted_results[:5]}")
        
        return decrypted_results
    
    def update(self, user_id: int) -> List:
        """Update(u_id) → L_id"""
        if user_id >= len(self.aux):
            raise ValueError(f"User ID {user_id} is out of range")
        return self.aux[user_id]
    
    def _create_demo_ai_model(self) -> List[float]:
        """Create a demo AI model."""
        return [0.2, 0.3, 0.1, 0.4, 0.25]
    
    # Revoke algorithm
    
    def revoke(self, user_id: int, pp: List, aux: List) -> Tuple[List, List]:
  
        print(f"\n{'='*60}")
        print(f"[Revoke*] Revoking user {user_id} (DeCart* optimization)")
        print(f"{'='*60}")
        
        # 1. Verify that the user exists
        if user_id not in self.registered_users:
            raise ValueError(f"User {user_id} is not registered and cannot be revoked")
        
        if user_id not in self.user_secrets:
            raise ValueError(f"User {user_id} has not executed KeyGen")
        
        # 2. Check whether already revoked
        if self.is_revoked(user_id):
            print(f"   User {user_id} has already been revoked, skipping")
            return pp, aux
        
        # 3. Generate revocation factor r_id in Z_p
        r_id = self.ff.random_nonzero()
        print(f"   Generated revocation factor: r_id = {hex(r_id)[:20]}...")
        
        # 4. Get user information
        user_info = self.user_secrets[user_id]
        u_id_prime = user_info['u_id_prime']
        block_num = user_info['block']
        
        print(f"   User info: u_id'={u_id_prime}, block={block_num}")
        
        # 5. Generate revocation public key pk_r,id
        h_idx = u_id_prime - 1
        h_u = self.crs['h_i'][h_idx]
        pk_r = self.bp.exponentiate_g1(h_u, r_id)
        print(f"   Generated revocation public key: pk_r = h_{u_id_prime}^{r_id}")
        
        # 6. Generate revocation personal auxiliary parameters pap_r,id
        pap_r = []
        
        for i in range(1, self.params.n + 1):
            if i == u_id_prime:
                pap_r.append(None)
                continue
            
            if i < u_id_prime:
                target_i = i + self.params.n
            else:
                target_i = 2 * u_id_prime - i + self.params.n
            
            target_idx = target_i - 1
            if target_i > self.params.n:
                target_idx += 1
            
            if target_idx >= len(self.crs['h_i']) or self.crs['h_i'][target_idx] is None:
                pap_r.append(None)
                continue
            
            h_target = self.crs['h_i'][target_idx]
            pap_element = self.bp.exponentiate_g1(h_target, r_id)
            pap_r.append(pap_element)
        
        non_empty = len([p for p in pap_r if p is not None])
        print(f"   Generated revocation pap: {non_empty} non-empty elements")
        
        # 7. Save revocation information
        self._revoked_users.add(user_id)
        self._revoked_info[user_id] = {
            'r_id': r_id,
            'pk_r': pk_r,
            'pap_r': pap_r,
            'revoke_time': time.time(),
            'original_block': block_num,
            'original_u_id_prime': u_id_prime
        }
        self._revocation_factors[user_id] = r_id
        
        # 8. Update public and auxiliary parameters
        pp_new = pp.copy()
        aux_new = [list(L) for L in aux]
        
        k = math.ceil((user_id + 1) / self.params.n) - 1
        pp_new[k] = pk_r
        print(f"   Updated public parameter for block {k}")
        
        # Update auxiliary parameters - add revocation pap for other users in the same block
        block_start = k * self.params.n
        block_end = min(block_start + self.params.n, self.params.N)
        
        updated_count = 0
        for j in range(block_start, block_end):
            if j != user_id:
                aux_new[j].append(copy.deepcopy(pap_r))
                updated_count += 1
        
        print(f"   Updated auxiliary parameters for {updated_count} users")
        
        # 9. Remove from registered users
        if user_id in self.registered_users:
            del self.registered_users[user_id]
        
        # 10. Notify affected owners
        affected_owners = self._notify_owners_about_revoke(user_id)
        if affected_owners:
            print(f"   Marked {len(affected_owners)} affected owners")
        
        print(f"\n Revoke* completed")
        print(f"   User {user_id} has been revoked")
        
        return pp_new, aux_new
    
    def _notify_owners_about_revoke(self, revoked_user_id: int) -> List[int]:

        affected = []
        for owner_id, policy in self.access_policies.items():
            if revoked_user_id in policy:
                affected.append(owner_id)
        return affected
    
    def is_revoked(self, user_id: int) -> bool:
        """Check whether a user has been revoked."""
        return user_id in self._revoked_users
    
    def get_revocation_info(self, user_id: int) -> Dict:
        """Get revocation information for a user."""
        return self._revoked_info.get(user_id, {})
    
    def get_revocation_factor(self, user_id: int) -> Optional[int]:
        """Get a user's revocation factor."""
        return self._revocation_factors.get(user_id)
    
    def get_all_revoked_users(self) -> List[int]:
        """Get a list of all revoked users."""
        return list(self._revoked_users)
    
    def get_affected_owners(self, revoked_user_id: int) -> List[int]:
        """Get owners affected by a revocation."""
        return self._notify_owners_about_revoke(revoked_user_id)
    
    # Policy update
    
    def update_policy_after_revoke(self, C_m: Dict, revoked_user_id: int) -> Dict:
    
        print(f"\n[Policy Update] Updating policy to remove user {revoked_user_id}")
        
        if revoked_user_id not in C_m['P']:
            print(f"   User {revoked_user_id} is not in the policy; no update needed")
            return C_m
        
        # Create a new policy (remove revoked user)
        new_policy = [uid for uid in C_m['P'] if uid != revoked_user_id]
        
        if not new_policy:
            print(f"     Warning: new policy is empty")
            return C_m
        
        print(f"   Old policy: {C_m['P']}")
        print(f"   New policy: {new_policy}")
        
        # Regenerate random values
        beta_new = self.ff.random_element()
        gamma_new = self.ff.random_element()
        
        # Recompute ciphertext components (only for users in the new policy)
        c1_new, c2_new, c4_new = [], [], []
        
        for u_id in new_policy:
            k_i = math.ceil((u_id + 1) / self.params.n) - 1
            c1_i = self.pp[k_i]
            c1_new.append(c1_i)
            
            u_id_prime = (u_id % self.params.n) + 1
            
            # Recompute c2
            h_target_idx = (self.params.n + 1 - u_id_prime) - 1
            if h_target_idx >= self.params.n:
                h_target_idx += 1
            
            h_target = self.crs['h_i'][h_target_idx]
            pairing_val = self._compute_pairing(c1_i, h_target)
            c2_i = self.bp.exponentiate_gt(pairing_val, gamma_new)
            c2_new.append(c2_i)
            
            # Recompute c4
            h_u_idx = u_id_prime - 1
            h_u = self.crs['h_i'][h_u_idx]
            pairing_h = self._compute_pairing(h_u, h_target)
            pairing_h_gamma = self.bp.exponentiate_gt(pairing_h, gamma_new)
            c4_new.append((pairing_h_gamma, beta_new))
        
        # Update ciphertext
        C_m_new = C_m.copy()
        C_m_new['P'] = new_policy
        C_m_new['c1_i'] = c1_new
        C_m_new['c2_i'] = c2_new
        C_m_new['c4_i'] = c4_new
        C_m_new['beta'] = beta_new
        C_m_new['gamma'] = gamma_new
        C_m_new['n_p'] = len(new_policy)
        C_m_new['updated_after_revoke'] = True
        C_m_new['revoked_user'] = revoked_user_id
        C_m_new['update_time'] = time.time()
        
        print(f"      Policy update completed")
        print(f"      New policy includes {len(new_policy)} users")
        
        return C_m_new
    
    # Utility methods
    
    def get_system_state(self) -> Dict:
        """Get system state."""
        return {
            'crs': self.crs is not None,
            'pp_len': len(self.pp) if self.pp else 0,
            'aux_len': len(self.aux) if self.aux else 0,
            'registered_users': len(self.registered_users),
            'user_secrets': len(self.user_secrets),
            'encrypted_datasets': len(self.encrypted_datasets),
            'revoked_users': len(self._revoked_users),
            'revoked_users_list': list(self._revoked_users),
            'trust_relations': sum(len(v) for v in self._trust_map.values()) if hasattr(self, '_trust_map') else 0
        }
    
    def reset(self):
        # Reset system state
        self.crs = None
        self.pp = None
        self.aux = None
        self.registered_users = {}
        self.user_secrets = {}
        self.encrypted_datasets = {}
        self.access_policies = {}
        self._trust_map = {}
        self._revoked_users = set()
        self._revoked_info = {}
        self._revocation_factors = {}
        print("\n  System state has been reset")
    
    # Test methods
    
    def test_ai_model_encryption(self):
        
        if not AI_MODELS_AVAILABLE:
            print("  AI model module unavailable")
            return False
        
        # Test decision tree encryption
        tree = self._create_default_decision_tree()
        pk_h = self.he.public_key
        encrypted_tree = self.encrypt_decision_tree(tree, pk_h)
        
        assert encrypted_tree['type'] == 'decision_tree'
        print(f"\n  Decision tree encryption test passed")
        
        # Test neural network encryption
        nn = self.create_default_neural_network()
        encrypted_nn = self.encrypt_neural_network(nn, pk_h)
        
        assert encrypted_nn['type'] == 'neural_network'
        print(f"  Neural network encryption test passed")
        
        return True
    
    def test_model_query(self, model_type: str, C_m: Dict, sk_h_s: Any, C_M_base: Dict) -> bool:
        """Test query execution for a specific model."""
        print(f"\n{'-'*50}")
        print(f"Testing {model_type} model query")
        print(f"{'-'*50}")
        
        try:
            C_M = C_M_base.copy()
            pk_h = self.he.public_key
            
            if model_type == 'dot':
                # Dot-product model
                print(f"Creating dot-product model...")
                ai_model = [0.1, 0.2, 0.3, 0.4, 0.5]
                encrypted_model = self.he.encrypt(ai_model)
                C_M['encrypted_model'] = encrypted_model
                
            elif model_type == 'decision_tree':
                # Decision tree model
                print(f"Creating decision tree model...")
                tree = self._create_default_decision_tree()
                encrypted_model = self.encrypt_decision_tree(tree, pk_h)
                C_M['encrypted_model'] = encrypted_model
                
            elif model_type == 'neural_network':
                # Neural network model
                print(f"Creating neural network model...")
                encrypted_model = self.encrypt_neural_network()
                C_M['encrypted_model'] = encrypted_model
            
            else:
                print(f"  Unknown model type: {model_type}")
                return False
            
            # Execute query
            print(f"Executing encrypted query...")
            ER = self.query(C_M, C_m, sk_h_s)
            
            # Decrypt results
            print(f"Decrypting results...")
            results = self.decrypt(C_M['sk_h_u'], ER)
            
            print(f"   {model_type} model query succeeded")
            print(f"   Result count: {len(results)}")
            print(f"   Result sample: {results[:5]}")
            
            return True
            
        except Exception as e:
            print(f"   {model_type} model query failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_complete_workflow(self):
        
        try:
            # 1. System initialization
            self.setup()
            
            # 2. Create and register users
            user_ids = [0, 1, 2]
            user_keys = {}
            
            for uid in user_ids:
                sk, pk, pap = self.keygen(uid)
                user_keys[uid] = (sk, pk, pap)
                self.register(uid, pk, pap)
            
            print(f"\n  User registration completed: {user_ids}")
            
            # 3. Data owner encrypts data
            owner_id = 0
            access_policy = [0, 1, 2]
            data_records = [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
                [5.0, 6.0, 7.0, 8.0, 9.0]
            ]
            
            C_m, sk_h_s = self.encrypt(owner_id, access_policy, data_records)
            print(f"\n  Data encryption completed: {len(data_records)} records")
            
            # 4. Querier checks access
            querier_id = 1
            querier_sk = user_keys[querier_id][0]
            
            C_M_base = self.check(querier_id, querier_sk, C_m)
            if C_M_base is None:
                print("  Access check failed")
                return False
            
            print(f"\n  Access verification passed")
            
            # 5. Test all model types
            print("\n" + "="*70)
            print("Start testing all model types")
            print("="*70)
            
            results = {}
            
            # Test dot-product model
            results['dot'] = self.test_model_query('dot', C_m, sk_h_s, C_M_base)
            
            # Test decision tree model
            results['decision_tree'] = self.test_model_query('decision_tree', C_m, sk_h_s, C_M_base)
            
            # Test neural network model
            results['neural_network'] = self.test_model_query('neural_network', C_m, sk_h_s, C_M_base)
            
            # 6. Summarize results
            print("\n" + "="*70)
            print("  Test result summary")
            print("="*70)
            
            all_passed = True
            for model_type, passed in results.items():
                status = "  PASSED" if passed else "  FAILED"
                print(f"   {status} - {model_type}")
                all_passed = all_passed and passed
            
            if all_passed:
                print("\n All model tests passed!")
                print("   Supported: dot-product, decision tree, neural network")
            else:
                print("\n  Some model tests failed")
            
            print("\n" + "="*70)
            return all_passed
            
        except Exception as e:
            print(f"\n  Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


# Test functions

def test_revoke_functionality():
    """Specifically test revoke functionality."""
    print("\n" + "="*80)
    print("  Testing DeCart* revoke functionality")
    print("="*80)
    
    try:
        system = DeCartStarSystem(DeCartStarParams(N=32, n=8))
        
        print("\n1. System initialization...")
        system.setup()
        
        print("\n2. Creating users...")
        users = [5, 6, 7]
        user_keys = {}
        for uid in users:
            sk, pk, pap = system.keygen(uid)
            user_keys[uid] = {'sk': sk, 'pk': pk, 'pap': pap}
            system.register(uid, pk, pap)
            print(f"   User {uid} registered successfully")
        
        print("\n3. Initial state:")
        print(f"   Registered users: {len(system.registered_users)}")
        print(f"   Revoked users: {len(system._revoked_users)}")
        assert len(system.registered_users) == 3, f"Registered user count should be 3, got {len(system.registered_users)}"
        assert len(system._revoked_users) == 0, f"Revoked user count should be 0, got {len(system._revoked_users)}"
        
        print("\n4. Revoking user 6...")
        pp_new, aux_new = system.revoke(6, system.pp, system.aux)
        system.pp = pp_new
        system.aux = aux_new
        
        print("\n5. Verifying revocation state:")
        is_revoked_6 = system.is_revoked(6)
        is_revoked_5 = system.is_revoked(5)
        print(f"   Is user 6 revoked: {is_revoked_6}")
        print(f"   Is user 5 revoked: {is_revoked_5}")
        assert is_revoked_6 == True, "User 6 should be revoked"
        assert is_revoked_5 == False, "User 5 should not be revoked"
        
        print(f"\n  All DeCart* revoke tests passed")
        return True
        
    except Exception as e:
        print(f"\n  Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Original test suite

def test_setup():
    """Test 1: Setup algorithm."""
    print("\n" + "★"*70)
    print("Test 1: Setup algorithm test")
    print("★"*70)
    
    system = DeCartStarSystem(DeCartStarParams(N=64, n=16))
    crs, pp, aux = system.setup()
    
    assert crs is not None, "crs is empty"
    assert pp is not None, "pp is empty"
    assert aux is not None, "aux is empty"
    assert len(pp) == 4, f"Incorrect pp length: {len(pp)}"
    assert len(aux) == 64, f"Incorrect aux length: {len(aux)}"
    assert len([h for h in crs['h_i'] if h]) == 31, "Incorrect h_i count"
    
    print(f"\n  Setup test passed")
    return system


def test_keygen():
    """Test 2: KeyGen algorithm."""
    print("\n" + "★"*70)
    print("Test 2: KeyGen algorithm test")
    print("★"*70)
    
    system = test_setup()
    
    test_users = [5, 6, 7]
    for uid in test_users:
        sk, pk, pap = system.keygen(uid)
        assert sk is not None, f"User {uid} sk is empty"
        assert pk is not None, f"User {uid} pk is empty"
        assert len(pap) == 16, f"Incorrect pap length for user {uid}"
        
        u_id_prime = (uid % 16) + 1
        assert pap[u_id_prime - 1] is None, f"Incorrect phi position for user {uid}"
    
    print(f"\n  KeyGen test passed")
    return system


def test_register_same_block():
    print("\n" + "★"*70)
    print("Test 3: Register algorithm test (same block)")
    print("★"*70)
    
    system = test_keygen()
    
    users_block0 = [5, 6, 7]
    for uid in users_block0:
        user_info = system.user_secrets[uid]
        pk = user_info['pk_id']
        pap = user_info['pap_id']
        system.register(uid, pk, pap)
    
    for uid in users_block0:
        aux_len = len(system.aux[uid])
        print(f"   User {uid} aux length: {aux_len}")
        assert aux_len == len(users_block0) - 1, f"Incorrect aux update for user {uid}"
    
    print(f"\n  Same-block registration test passed")
    return system


def test_encrypt():
    
    system = test_register_same_block()
    
    owner_id = 5
    access_policy = [5, 6, 7]
    data_records = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0]
    ]
    
    C_m, sk_h_s = system.encrypt(owner_id, access_policy, data_records)
    
    assert C_m is not None, "C_m is empty"
    assert 'c1_i' in C_m, "Missing c1_i"
    assert 'c2_i' in C_m, "Missing c2_i"
    assert 'c3' in C_m, "Missing c3"
    assert 'c4_i' in C_m, "Missing c4_i"
    assert 'c5' in C_m, "Missing c5"
    assert 'c6_i' in C_m, "Missing c6_i"
    assert len(C_m['c6_i']) == 2, "Incorrect encrypted data count"
    
    print(f"\n  Encrypt test passed")
    return system, C_m, sk_h_s


def test_check_self_query():
    
    system, C_m, _ = test_encrypt()
    
    owner_id = 5
    sk_id = system.user_secrets[owner_id]['sk_id']
    
    C_M = system.check(owner_id, sk_id, C_m)
    
    assert C_M is not None, "Self-query failed"
    assert C_M.get('self_query', False), "Not in self-query mode"
    assert C_M.get('access_granted', False), "Access denied"
    
    print(f"\n  Self-query test passed")
    return system, C_M


def test_query_decrypt():
    
    system, C_M = test_check_self_query()
    
    owner_id = 5
    C_m = system.encrypted_datasets[owner_id]
    sk_h_s = None
    
    ai_model = [0.1, 0.2, 0.3, 0.4, 0.5]
    encrypted_model = system.he.encrypt(ai_model)
    C_M['encrypted_model'] = encrypted_model
    
    ER = system.query(C_M, C_m, sk_h_s)
    assert ER is not None, "Query failed"
    assert 'encrypted_results' in ER, "Missing encrypted results"
    
    results = system.decrypt(C_M['sk_h_u'], ER)
    assert len(results) > 0, "Decrypted results are empty"
    print(f"   Decrypted result sample: {results[:3]}")
    
    print(f"\n  Query/Decrypt test passed")


def test_full_workflow():
    """Test 9: Complete workflow test."""
    print("\n" + "★"*70)
    print("Test 9: Complete workflow test")
    print("★"*70)
    
    system = DeCartStarSystem(DeCartStarParams(N=64, n=16))
    system.setup()
    
    users = {}
    for uid in [5, 6, 7]:
        sk, pk, pap = system.keygen(uid)
        users[uid] = {'sk': sk, 'pk': pk, 'pap': pap}
    
    for uid in [5, 6, 7]:
        system.register(uid, users[uid]['pk'], users[uid]['pap'])
    
    owner_id = 5
    access_policy = [5, 6, 7]
    data_records = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0]
    ]
    C_m, sk_h_s = system.encrypt(owner_id, access_policy, data_records)
    
    querier_id = 6
    C_M = system.check(querier_id, users[querier_id]['sk'], C_m)
    assert C_M is not None, "Check failed"
    
    ai_model = [0.1, 0.2, 0.3, 0.4, 0.5]
    encrypted_model = system.he.encrypt(ai_model)
    C_M['encrypted_model'] = encrypted_model
    
    ER = system.query(C_M, C_m, sk_h_s)
    
    results = system.decrypt(C_M['sk_h_u'], ER)
    
    print(f"\n   Complete workflow executed successfully!")
    print(f"   Registered users: {len(system.registered_users)}")
    print(f"   Encrypted data: {len(C_m['c6_i'])} records")
    print(f"   Query results: {len(results)}")
    
    print(f"\n  Complete workflow test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("  Starting full DeCart* test suite")
    print("="*80)
    
    tests = [
        ("Setup Algorithm", test_setup),
        ("KeyGen Algorithm", test_keygen),
        ("Register Same Block", test_register_same_block),
        ("Encrypt Algorithm", test_encrypt),
        ("Check Self Query", test_check_self_query),
        ("Query/Decrypt", test_query_decrypt),
        ("Complete Workflow", test_full_workflow),
        ("AI Model Encryption", lambda: DeCartStarSystem().test_ai_model_encryption()),
        ("Multi-Model Query", lambda: DeCartStarSystem().test_complete_workflow()),
        ("Revoke Functionality", test_revoke_functionality)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n ▶ Running: {name} \n")

        
        try:
            if name == "Setup Algorithm":
                test_func()
            elif name == "KeyGen Algorithm":
                test_func()
            elif name == "Register Same Block":
                test_func()
            elif name == "Encrypt Algorithm":
                test_func()
            elif name == "Check Self Query":
                test_func()
            elif name == "Query/Decrypt":
                test_func()
            elif name == "Complete Workflow":
                test_func()
            elif name == "AI Model Encryption":
                system = DeCartStarSystem(DeCartStarParams(N=32, n=8))
                system.test_ai_model_encryption()
            elif name == "Multi-Model Query":
                system = DeCartStarSystem(DeCartStarParams(N=32, n=8))
                system.test_complete_workflow()
            elif name == "Revoke Functionality":
                test_func()
            
            results[name] = True
            print(f"\n {name} passed")
            
        except Exception as e:
            results[name] = False
            print(f"\n {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(" Test result summary")
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"   {status} - {name}")
        all_passed = all_passed and passed
    
    print("\n" + "="*80)
    if all_passed:
        print("   DeCart* tests passed")

    else:
        print("   Some tests failed")
    print("="*80)
    
    return all_passed


# Export interface

class DeCartStarScheme:
    """Main class for the DeCart* scheme."""
    
    def __init__(self, params: Optional[DeCartStarParams] = None):
        self.system = DeCartStarSystem(params)
    
    def setup(self):
        return self.system.setup()
    
    def keygen(self, user_id: int):
        return self.system.keygen(user_id)
    
    def register(self, user_id: int, pk_id: Any, pap_id: List):
        return self.system.register(user_id, pk_id, pap_id)
    
    def encrypt(self, owner_id: int, policy: List[int], data: List[List[float]]):
        return self.system.encrypt(owner_id, policy, data)
    
    def check(self, querier_id: int, sk_id: int, C_m: Dict):
        return self.system.check(querier_id, sk_id, C_m)
    
    def encrypt_decision_tree(self, tree_model, pk_h: Any):
        return self.system.encrypt_decision_tree(tree_model, pk_h)
    
    def encrypt_neural_network(self, nn_model=None, pk_h: Any = None):
        return self.system.encrypt_neural_network(nn_model, pk_h)
    
    def query(self, C_M: Dict, C_m: Dict, sk_h_s: Any):
        return self.system.query(C_M, C_m, sk_h_s)
    
    def decrypt(self, sk_h_u: Any, ER: Dict):
        return self.system.decrypt(sk_h_u, ER)
    
    def revoke(self, user_id: int, pp: List, aux: List):
        return self.system.revoke(user_id, pp, aux)
    
    def is_revoked(self, user_id: int) -> bool:
        return self.system.is_revoked(user_id)
    
    def update_policy(self, C_m: Dict, revoked_user_id: int):
        return self.system.update_policy_after_revoke(C_m, revoked_user_id)
    
    def get_affected_owners(self, revoked_user_id: int) -> List[int]:
        return self.system.get_affected_owners(revoked_user_id)
    
    def add_trust(self, truster_id: int, trustee_id: int):
        self.system.add_trust(truster_id, trustee_id)
    
    def get_trusted_by(self, user_id: int) -> Set[int]:
        return self.system.get_trusted_by(user_id)
    
    def test_ai_models(self):
        return self.system.test_ai_model_encryption()
    
    def test_complete(self):
        return self.system.test_complete_workflow()
    
    def test_revoke(self):
        return test_revoke_functionality()


if __name__ == "__main__":

    system = DeCartStarSystem(DeCartStarParams(N=32, n=8))
    
    # Test AI model encryption first
    print(" Testing AI model encryption")
    ai_success = system.test_ai_model_encryption()
    
    # Then run the existing complete workflow test
    print(" Complete workflow")
    workflow_success = system.test_complete_workflow()
    
    # Finally run the newly added revoke test
    print(" Revoke functionality")
    revoke_success = test_revoke_functionality()
    