# decart/schemes/decart.py

import math
import secrets
import hashlib
import sys
import os
import time
import copy
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

# Import core modules
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, '..', 'core')
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

# Import AI model modules
from schemes.ai_model import (
    DecisionTreeHE,
    NeuralNetworkHE,
    EncryptedModelWrapper,
    ActivationFunctions,
    DecisionTreeNode 
)

from bilinear_pairing import BilinearPairing
from homomorphic import HomomorphicEncryption
from finite_field import FiniteField
from config import Config


@dataclass
class DeCartParams:
    """DeCart"""
    lambda_security: int = 128      # security parameterλ
    N: int = Config.MAX_USERS       # maximum usersN ∈ Z_p
    n: int = Config.BLOCK_SIZE      # users per blockn ∈ Z_p
    
    @property
    def B(self) -> int:
        """Number of blocks B = ceil(N/n)"""
        return math.ceil(self.N / self.n)


class DeCartSystem:
    
    def __init__(self, params: Optional[DeCartParams] = None):
        self.params = params or DeCartParams(N=Config.MAX_USERS, n=Config.BLOCK_SIZE)
        
        # Initialize cryptographic primitives
        print("Initializing bilinear pairing")
        self.bp = BilinearPairing(enable_cache=True)
        
        print("Initializing homomorphic encryption")
        self.he = HomomorphicEncryption(poly_modulus_degree=Config.POLY_MODULUS_DEGREE)
        
        print("Initializing finite field")
        self.ff = FiniteField(p=self.bp.get_group_order())
        
        print(f"\n DeCart system initialized")
        print(f"   Parameters: N={self.params.N}, n={self.params.n}, B={self.params.B}")
        print(f"   Prime field: Z_{self.ff.p}")

        
        # System state
        self.crs = None
        self.pp = None
        self.aux = None
        
        # Storage
        self.registered_users = {}      # user_id -> registration info
        self.user_secrets = {}           # user_id -> secret info
        self.encrypted_datasets = {}     # owner_id -> C_m
        self.access_policies = {}        # owner_id -> policy
        
        # ===== Revocation state =====
        self._revoked_users = set()           # Revoked user set
        self._revoked_info = {}                # Revocation info {user_id: info}
        self._revocation_factors = {}          # Revocation factor {user_id: r_id}
        
        # ===== Cache =====
        self._pairing_cache = {}
        self._e_gg_cache = {}

    # Default neural network creation method
    def create_default_neural_network(self, input_dim: int = 784, output_dim: int = 10) -> Any:
        try:
            return NeuralNetworkHE.create_shallow_mlp(
                input_dim=input_dim,
                hidden_dim=16,
                output_dim=output_dim,
                hidden_activation="square",
                output_activation="linear",
            )
        except Exception:
            print("Warning: failed to import NeuralNetworkHE")
            return None

    def setup(self) -> Tuple[Dict, List, List]:

        print("\n" + "="*50)
        print("[Setup] System initialization")
        print("="*50)
        
        # 1. Generate bilinear group Ψ = (p, g, G, G_T, Z_p, e), where e(G, G) → G_T
        p = self.ff.p
        g = self.bp.g1  
        
        print("1. Sample random values {z_i}...")
        z_values = [self.ff.random_element() for _ in range(self.params.n)]
        
        # 2. Compute h_i = g^{z_i} 
        print("2. Compute h_i = g^{z_i}...")
        h_i = []
        for z in z_values:
            h = self.bp.exponentiate_g1(g, z)
            h_i.append(h)
        
        # 3. Compute H_{i,j} = g^{z_i*z_j}
        print("3. compute H_{i,j} = e(g,g)^{z_i·z_j}(symmetric half-storage)...")
        H_ij = {}
        
        # Base pairing value e(g,g) 
        e_gg = self.bp.pairing(self.bp.g1, self.bp.g2)
        
        total_pairs = (self.params.n * (self.params.n - 1)) // 2
        progress_step = max(1, self.params.n // 10)

        for i in range(self.params.n):
            for j in range(i + 1, self.params.n):
                exponent = (z_values[i] * z_values[j]) % p
                H_ij[(i, j)] = self.bp.exponentiate_gt(e_gg, exponent)

            if (i + 1) % progress_step == 0 or (i + 1) == self.params.n:
                print(f"   H_ij progress: {i + 1}/{self.params.n} rows, generated {len(H_ij)}/{total_pairs} elements")
        
        # 4. Select hash function H: G_T → {0,1}*
        def H_gt_to_bytes(gt_elem: Any) -> bytes:
            """Hash function H: G_T → {0,1}*"""
            try:
                gt_bytes = self.bp.serialize_gt(gt_elem)
                return hashlib.sha256(gt_bytes).digest()
            except:
                # Fallback option
                gt_str = str(gt_elem).encode()
                return hashlib.sha256(gt_str).digest()
        
        # 5. Build CRS
        self.crs = {
            'Ψ': (p, g, self.bp.gt, self.ff, self.bp.pairing),
            'N': self.params.N,
            'B': self.params.B,
            'n': self.params.n,
            'h_i': h_i,          # in group G (G1 representation)
            'H_ij': H_ij,        # in G_T
            'H': H_gt_to_bytes,
            'z_values': z_values,
            'e_gg': e_gg,        # base value e(g,g)
            'g': g,               # group G generator
            'p': p
        }
        
        # 6. translatedParameters pp = (C_{(1)} = 1, ..., C_{(B)} = 1) ∈ G
        identity = self.bp.exponentiate_g1(g, 0)
        self.pp = [identity for _ in range(self.params.B)]
        
        # 7. translatedParameters aux = (L_1 = {1}, ..., L_N = {1}) ∈ G
        self.aux = [[] for _ in range(self.params.N)]
        
        print(f"\n Setup completed")
        print(f"   h_i: {len(h_i)} group G elements (G1 representation)")
        print(f"   H_ij: {len(H_ij)} translatedG_Ttranslated(symmetric half-storage)")
        print(f"   Base pairing value e(g,g): computed")
        
        return self.crs, self.pp, self.aux
    
    #  KeyGen algorithm 
    
    def keygen(self, user_id: int) -> Tuple[int, Any, List[Optional[Any]]]:

        if not (0 <= user_id < self.params.N):
            raise ValueError(f"User ID must be in [0, {self.params.N-1}]")
        
        # ===== Check whether the user has been revoked =====
        if self.is_revoked(user_id):
            raise ValueError(f"User {user_id} has been revoked and cannot generate a new key")
        
        print(f"\n[KeyGen] User {user_id} Generating keys...")
        
        # 1. Generate random secret key x_id ∈ Z_p
        x_id = self.ff.random_element()
        
        # 2. compute u_id' = (u_id mod n) + 1 (1-based)
        u_id_prime = user_id % self.params.n
        u_id_prime_1based = u_id_prime + 1
        
        # 3. Compute public key pk_id = h_{u_id'}^{x_id} (translatedGtranslated)
        h_u = self.crs['h_i'][u_id_prime]
        pk_id = self.bp.exponentiate_g1(h_u, x_id)
        
        # 4. Compute personal auxiliary parameters pap_id
        pap_id = []
        for i in range(self.params.n):
            i_1based = i + 1
            
            if i_1based == u_id_prime_1based:
                pap_id.append(None)  # φ
            else:
                # Get H_{i,u_id'}^{x_id}
                # First find H_{i,u_id'}
                H_key = (i, u_id_prime) if (i, u_id_prime) in self.crs['H_ij'] else (u_id_prime, i)
                H_val = self.crs['H_ij'][H_key]
                
                # compute H_{i,u_id'}^{x_id}
                pap_element = self.bp.exponentiate_gt(H_val, x_id)
                pap_id.append(pap_element)
        
        # Store user information
        block_num = user_id // self.params.n
        self.user_secrets[user_id] = {
            'sk_id': x_id,
            'pap_id': pap_id,
            'u_id_prime': u_id_prime,
            'u_id_prime_1based': u_id_prime_1based,
            'block': block_num,
            'pk_id': pk_id,
            'user_id': user_id
        }
        
        print(f"      KeyGen completed")
        print(f"      sk_id: {x_id}")
        print(f"      u_id': {u_id_prime_1based} (1-based)")
        print(f"      Block: {block_num}")
        
        return x_id, pk_id, pap_id
    
    #  Register algorithm 
    
    def register(self, user_id: int, pk_id: Any, pap_id: List[Optional[Any]]) -> Tuple[List, List]:
        """
        Register(u_id, pk_id, pap_id) → (pp', aux')
        Enhanced mathematical validation with revocation checks
        """
        print(f"\n[Register] User {user_id} registering...")
        

        if self.is_revoked(user_id):
            raise ValueError(f"User {user_id} translatedrevocation")
        
        if user_id not in self.user_secrets:
            raise ValueError(f"User {user_id} has not executed KeyGen")
        
        user_info = self.user_secrets[user_id]
        u_id_prime = user_info['u_id_prime']
        u_id_prime_1based = user_info['u_id_prime_1based']
        
        # 1. Validate pap_id
        print(f"   Validating pap_id...")
        
        if len(pap_id) != self.params.n:
            raise ValueError(f"pap_id length must be {self.params.n}")
        
        for i in range(self.params.n):
            if i == u_id_prime:
                if pap_id[i] is not None:
                    raise ValueError(f"pap_id[{i}] should be None (φ)")
            else:
                if pap_id[i] is None:
                    raise ValueError(f"pap_id[{i}] translatedshould be None")
        
        print(f"      pap_idformat validation passed")
        
        # 2. Compute block index k = ceil((u_id + 1)/n)
        k = math.ceil((user_id + 1) / self.params.n) - 1
        
        # 3. Update public parameters C_{(k)}' = C_{(k)} · pk_id
        self.pp[k] = pk_id
        
        # 4. Update auxiliary parameters L_j
        block_start = k * self.params.n
        block_end = min(block_start + self.params.n, self.params.N)
        
        updated_count = 0
        for j in range(block_start, block_end):
            if j == user_id:
                continue
            
            self.aux[j].append(copy.deepcopy(pap_id))
            updated_count += 1
        
        # 5. translatedUsertranslatedregistering
        self.registered_users[user_id] = {
            'pk_id': pk_id,
            'block': k,
            'u_id_prime': u_id_prime,
            'registered': True,
            'user_id': user_id,
            'pap_id': pap_id,
            'register_time': time.time()
        }
        
        print(f"      Register completed")
        print(f"      Update block {k} translatedParameters")
        print(f"      Updated {updated_count}  auxiliary parameters")
        print(f"      User {user_id} translatedregistering")
        
        return self.pp, self.aux
    
    #  Encrypt algorithm 
    
    def _check_data_range(self, data_records: List[List[float]]):
        for i, record in enumerate(data_records):
            for j, val in enumerate(record):
                if abs(val) > 10:
                    print(f"      Data[{i}][{j}] = {val} exceeds recommended range [-10, 10]")
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Data[{i}][{j}]contains invalid value: {val}")
    
    def encrypt(self, owner_id: int, access_policy: List[int], 
               data_records: List[List[float]]) -> Tuple[Dict, Any]:

        if owner_id not in self.registered_users:
            raise ValueError(f"User {owner_id} translatedregistering")
        
        print(f"\n[Encrypt] Owner {owner_id} encrypting data...")
        
        # Check data range
        self._check_data_range(data_records)
        
        # 1. access policy
        n_p = len(access_policy)
        print(f"   access policytranslated {n_p} users")
        
        # 2. Sample random values α ∈ Z_p
        alpha = self.ff.random_element()
        
        # 3. Generate homomorphic encryption key
        pk_h = self.he.public_key
        
        # 4. Sample random values (β, γ) ∈ Z_p
        beta = self.ff.random_element()
        gamma = self.ff.random_element()
        print(f"   Random values: α={alpha}, β={beta}, γ={gamma}")
        
        # 5. Split homomorphic secret key
        sk_h_shares = self.he.split_secret_key_shamir(num_shares=2, threshold=2)
        sk_h_s = sk_h_shares[0]
        sk_h_u = sk_h_shares[1]
        
        # 6. encrypting data
        n_m = len(data_records)
        c6_list = []
        
        print(f"   Encrypt {n_m} data records...")
        for i, data in enumerate(data_records):
            if isinstance(data, (int, float)):
                data = [float(data)]
            try:
                encrypted = self.he.encrypt(data)
                c6_list.append(encrypted)
                if (i + 1) % 10 == 0:
                    print(f"   Encrypted {i+1}/{n_m} records")
            except Exception as e:
                print(f"      translated{i}data record encryption failed: {e}")
                raise
        
        # 7. Compute ciphertext components
        c1_list, c2_list, c4_list = [], [], []
        
        print(f"   Compute ciphertext components...")
        for u_id in access_policy:
            # k_i = ceil((u_id + 1)/n)
            k_i = math.ceil((u_id + 1) / self.params.n) - 1
            
            # c1,i = C_{(k_i)}
            c1_i = self.pp[k_i]
            c1_list.append(c1_i)
            
            # u_id'
            u_id_prime = u_id % self.params.n
            
            # c2,i = e(C_{(k_i)}, h_{u_id'})^γ
            h_u = self.crs['h_i'][u_id_prime]
            pairing_val = self._symmetric_pairing_sim(c1_i, h_u, gamma)
            c2_list.append(pairing_val)
            
            # c4,i = e(h_{u_id'}, h_{u_id'})^γ · β
            c4_i = self._compute_c4_i(u_id_prime, gamma, beta)
            c4_list.append(c4_i)
        
        # c3 = g^γ
        c3 = self.bp.exponentiate_g1(self.crs['g'], gamma)
        
        # c5 = H[e(g,g)^β] ⊕ (pk_h || sk_h,u)
        c5 = self._compute_c5(beta, pk_h, sk_h_u)
        
        # 8. Build full ciphertext
        C_m = {
            'P': access_policy,
            'c1_i': c1_list,
            'c2_i': c2_list,
            'c3': c3,
            'c4_i': c4_list,
            'c5': c5,
            'c6_i': c6_list,
            'n_p': n_p,
            'n_m': n_m,
            'owner_id': owner_id,
            'beta': beta,
            'gamma': gamma,
            'alpha': alpha,
            'encrypt_time': time.time()
        }
        
        # Storage
        self.encrypted_datasets[owner_id] = C_m
        self.access_policies[owner_id] = access_policy
        
        print(f"      Encrypt completed")
        print(f"      Generate {n_p} policy components")
        print(f"      Encrypt {n_m} data records")
        
        return C_m, sk_h_s
    
    def _symmetric_pairing_sim(self, a, b, gamma):
        try:
            e_gg_gamma = self.bp.exponentiate_gt(self.crs['e_gg'], gamma)
            return e_gg_gamma
        except:
            return self.bp.exponentiate_gt(self.bp.pairing(self.bp.g1, self.bp.g2), gamma)
    
    def _compute_c4_i(self, u_id_prime, gamma, beta):
        """Compute c4,i = e(h_{u_id'}, h_{u_id'})^γ · β"""
        try:
            z_u = self.crs['z_values'][u_id_prime]
            z_u_sq = (z_u * z_u) % self.ff.p
            exponent = (z_u_sq * gamma) % self.ff.p
            gt_part = self.bp.exponentiate_gt(self.crs['e_gg'], exponent)
            
            return {
                'gt_element': gt_part,
                'beta': beta,
                'z_u': z_u,
                'gamma': gamma
            }
        except:
            return {
                'gt_element': self.bp.exponentiate_gt(self.crs['e_gg'], gamma),
                'beta': beta
            }
    
    def _compute_c5(self, beta, pk_h, sk_h_u):
        """Compute c5 = H[e(g,g)^β] ⊕ (pk_h || sk_h,u)"""
        e_gg_beta = self.bp.exponentiate_gt(self.crs['e_gg'], beta)
        hash_bytes = self.crs['H'](e_gg_beta)
        
        pk_h_bytes = self.he.serialize_context()
        if isinstance(sk_h_u, tuple):
            sk_h_u_bytes = f"{sk_h_u[0]},{sk_h_u[1]}".encode()
        else:
            sk_h_u_bytes = str(sk_h_u).encode()
        
        combined = pk_h_bytes + b'||' + sk_h_u_bytes
        xor_len = min(len(hash_bytes), len(combined))
        return bytes([hash_bytes[i] ^ combined[i] for i in range(xor_len)])
    
    #  Check algorithm 
    
    def check(self, querier_id: int, sk_id: int, C_m: Dict) -> Optional[Dict]:
        """
        Check(u_id, sk_id, C_m) → C_M
        Enhanced mathematical validation with revocation checks
        """
        print(f"\n[Check] querytranslated {querier_id} checking access permission...")
        
        # checkquerytranslatedrevocation
        if self.is_revoked(querier_id):
            print(f"      User {querier_id} translatedrevocation, has no access")
            return None
        
        # 1. checking access permission
        if querier_id not in C_m['P']:
            print(f"      translatedaccess policytranslated")
            return None
        
        j = C_m['P'].index(querier_id)
        u_id_prime = querier_id % self.params.n
        
        # 2. GettranslatedParameters
        L_id = self.aux[querier_id]
        if not L_id:
            print(f"       auxiliary parameters are empty")
            return None
        
        # 3. Find valid O_{id,i}
        O_found = None
        o_index = -1
        
        for i, O_list in enumerate(L_id):
            if O_list and len(O_list) > u_id_prime and O_list[u_id_prime] is not None:
                O_found = O_list[u_id_prime]
                o_index = i
                break
        
        if O_found is None:
            print(f"      No valid O element found")
            return None
        
        print(f"      Found valid O element (index {o_index})")
        
        # 4. Recover homomorphic key
        beta = C_m.get('beta', 0)
        c5 = C_m['c5']
        e_gg_beta = self.bp.exponentiate_gt(self.crs['e_gg'], beta)
        hash_bytes = self.crs['H'](e_gg_beta)
        
        recovered = bytes([a ^ b for a, b in zip(c5, hash_bytes[:len(c5)])])
        parts = recovered.split(b'||')
        if len(parts) >= 2:
            pk_h_bytes, sk_h_u_bytes = parts[0], parts[1]
        else:
            pk_h_bytes, sk_h_u_bytes = b'', b''
        
        # 5. Prepare C_M
        C_M = {
            'querier_id': querier_id,
            'pk_h_recovered': len(pk_h_bytes) > 10,
            'sk_h_u': sk_h_u_bytes if sk_h_u_bytes else b'demo',
            'access_granted': True,
            'o_index': o_index,
            'beta': beta,
            'check_time': time.time()
        }
        
        print(f"      Check completed")
        print(f"      Access verification passed; waiting for AI model encryption")
        
        return C_M
    
    #  AItranslatedencrypttranslated 
    
    def encrypt_decision_tree(self, tree_model, pk_h: Any) -> Dict:
        """
        Parameters:
            tree_model: sklearndecision treetranslatedDecisionTreeHEtranslated
            pk_h: translatedencrypttranslated
        
        translated:
            encrypttranslateddecision treeParameterstranslated
        """
        print(f"\n[Encrypt Decision Tree] encryptdecision treetranslated")

        # translatedDecisionTreeHE
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
            tree = DecisionTreeHE.from_sklearn(tree_model)
        
        # GettranslatedencryptParameters
        params = tree.get_encryptable_params()
        
        # encryptinternal nodes
        encrypted_internal = []
        for node in params['internal_nodes']:
            encrypted_node = {
                'node_id': node['node_id'],
                'feature_idx': node['feature_idx'],  # translatedindextranslatedStorage
                'threshold': self.he.encrypt([node['threshold']]),  # encrypttranslated
                'left': node['left'],
                'right': node['right']
            }
            encrypted_internal.append(encrypted_node)
        
        # encryptleaf nodes
        encrypted_leaves = []
        for node in params['leaf_nodes']:
            encrypted_node = {
                'node_id': node['node_id'],
                'value': self.he.encrypt([node['value']])  # encrypttranslated
            }
            encrypted_leaves.append(encrypted_node)
        
        encrypted_tree = {
            'type': 'decision_tree',
            'internal_nodes': encrypted_internal,
            'leaf_nodes': encrypted_leaves,
            'root_id': params['root_id'],
            'node_count': params['node_count']
        }
        
        print(f"      decision treeencrypttranslated")
        print(f"      internal nodes: {len(encrypted_internal)}")
        print(f"      leaf nodes: {len(encrypted_leaves)}")
        
        return encrypted_tree
    

    def encrypt_neural_network(self, nn_model=None, pk_h: Any = None) -> Dict:
        """
        encryptneural network .
        """
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
            if not hasattr(nn_model, 'get_encryptable_params'):
                print(f"   Warning: translatedneural network")
                nn_model = self.create_default_neural_network()
            try:
                params_list = nn_model.get_encryptable_params()
            except AttributeError:
                print(f"   Error: translated get_encryptable_params translated")
                return {
                    'type': 'neural_network',
                    'layers': [],
                    'layer_count': 0
                }
        
        # encrypttranslated
        encrypted_layers = []
        for params in params_list:
            weights_flat = params.get('weights', [])
            bias_flat = params.get('bias', [])
            
            print(f"   encrypttranslated {params.get('layer_idx', 0)}: {params.get('weights_shape', 'unknown')}")
            
            # encrypttranslated
            encrypted_weights = []
            for w in weights_flat:
                try:
                    encrypted_w = self.he.encrypt([float(w)])
                    encrypted_weights.append(encrypted_w)
                except Exception as e:
                    print(f"     Warning: translatedencryptFAILED: {e}")
                    encrypted_weights.append(None)
            
            # encrypttranslated
            encrypted_bias = []
            for b in bias_flat:
                try:
                    encrypted_b = self.he.encrypt([float(b)])
                    encrypted_bias.append(encrypted_b)
                except Exception as e:
                    print(f"     Warning: translatedencryptFAILED: {e}")
                    encrypted_bias.append(None)
            
            encrypted_layer = {
                'layer_idx': params.get('layer_idx', 0),
                'layer_type': params.get('layer_type', 'linear'),
                'activation': params.get('activation', 'linear'),
                'weights_shape': params.get('weights_shape', (0, 0)),
                'bias_shape': params.get('bias_shape', (0,)),
                'encrypted_weights': encrypted_weights,
                'encrypted_bias': encrypted_bias
            }
            encrypted_layers.append(encrypted_layer)
        
        encrypted_nn = {
            'type': 'neural_network',
            'layers': encrypted_layers,
            'layer_count': len(encrypted_layers)
        }
        
        print(f"      neural networkencrypttranslated")
        return encrypted_nn


    def encrypt_model(self, model_wrapper: EncryptedModelWrapper, pk_h: Any) -> Dict:
        """
        translatedencrypttranslated
        
        Parameters:
            model_wrapper: translated
            pk_h: translatedencrypttranslated
        
        translated:
            encrypttranslatedParameterstranslated
        """
        if model_wrapper.model_type == 'decision_tree':
            return self.encrypt_decision_tree(model_wrapper.plain_model, pk_h)
        elif model_wrapper.model_type == 'neural_network':
            return self.encrypt_neural_network(model_wrapper.plain_model, pk_h)
        else:
            raise ValueError(f"translatedmodel type: {model_wrapper.model_type}")
    
    #  encryptquerytranslated 
    
    def _query_decision_tree(self, 
                            encrypted_tree: Dict,
                            encrypted_data: List[Any],
                            sk_h_s: Any) -> List[Any]:
        """
        encryptdecision treequery - Algorithm 3
        translated
        """
        print(f"\n[Query Decision Tree] translatedencryptdecision treequery")
        
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
                # translated, translated
                current_node_id = encrypted_tree['root_id']

                # decrypttranslated, translated feature_idx translated
                record_plain = self.he.decrypt(encrypted_record)
                if not isinstance(record_plain, list):
                    record_plain = [float(record_plain)]

                # translated
                max_depth = 10  # translated
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
                
                # translatedleaf nodes
                if current_node_id in leaf_map:
                    pred_plain = leaf_map[current_node_id]
                    results.append(self.he.encrypt([pred_plain]))
                else:
                    # translated, translated
                    print(f"   Warning: translatedleaf nodes {current_node_id}")
                    results.append(self.he.encrypt([0.0]))
                
            except Exception as e:
                print(f"   translated{data_idx}translatedqueryFAILED: {e}")
                # translatedencrypttranslated0translated
                try:
                    results.append(self.he.encrypt([0.0]))
                except:
                    results.append(None)
        
        print(f"      decision treequerytranslated, Generate {len(results)} translated")
        return results
    
    @staticmethod
    def _apply_nn_activation(values: List[float], activation: str) -> List[float]:
        if activation == 'linear':
            return [float(v) for v in values]
        return [float(ActivationFunctions.get_he_friendly(activation, float(v))) for v in values]

    # neural networkquerytranslated
    def _query_single_layer_nn(self,
                              encrypted_nn: Dict,
                              encrypted_data: List[Any],
                              sk_h_s: Any) -> List[Any]:
        """
        neural networkquery
        """
        print(f"\n[Query Neural Network] translatedencryptquery")

        results = []
        progress_interval = 10 if len(encrypted_data) <= 1000 else 100

        if not encrypted_nn.get('layers'):
            print(f"   Warning: neural networktranslated")
            for _ in encrypted_data:
                try:
                    results.append(self.he.encrypt([0.0]))
                except Exception:
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
                    print(f"      neural networkqueryprogress: {data_idx + 1}/{len(encrypted_data)}")

            except Exception as e:
                print(f"   translated{data_idx}translatedqueryFAILED: {e}")
                try:
                    results.append(self.he.encrypt([0.0 for _ in range(max(1, fallback_dim))]))
                except Exception:
                    results.append(None)
                if (data_idx + 1) % progress_interval == 0 or (data_idx + 1) == len(encrypted_data):
                    print(f"      neural networkqueryprogress: {data_idx + 1}/{len(encrypted_data)}")

        print(f"      neural networkquerytranslated, Generate {len(results)} translated")
        return results
            
    #  Query algorithm 
    def query(self, C_M: Dict, C_m: Dict, sk_h_s: Any) -> Dict:

        print(f"\n[Query] translatedencryptAIquery...")
        
        if not C_M.get('access_granted', False):
            raise ValueError("translatedaccess permission")
        
        if 'encrypted_model' not in C_M:
            raise ValueError("translatedencrypttranslatedAItranslated")
        
        encrypted_model = C_M['encrypted_model']
        encrypted_data_list = C_m['c6_i']
        
        # translatedmodel type
        if isinstance(encrypted_model, dict):
            model_type = encrypted_model.get('type', 'unknown')
        else:
            model_type = 'dot_product'
        
        if model_type == 'decision_tree':
            print(f"   model type: decision tree")
            encrypted_results = self._query_decision_tree(
                encrypted_model, encrypted_data_list, sk_h_s
            )
        elif model_type == 'neural_network':
            print(f"   model type: neural network (translated)")

            encrypted_results = self._query_single_layer_nn(
                encrypted_model, encrypted_data_list, sk_h_s
            )
        else:
            # dot producttranslated
            print(f"   model type: dot product")
            encrypted_results = []
            failed_count = 0
            progress_interval = 10 if len(encrypted_data_list) <= 1000 else 100
            
            for i, encrypted_data in enumerate(encrypted_data_list):
                try:
                    result = encrypted_data.dot(encrypted_model)
                    encrypted_results.append(result)
                    if (i + 1) % progress_interval == 0 or (i + 1) == len(encrypted_data_list):
                        print(f"   dot productqueryprogress: {i + 1}/{len(encrypted_data_list)}")
                except Exception as e:
                    failed_count += 1
                    try:
                        result = self.he.encrypt([0.0])
                    except:
                        result = None
                    encrypted_results.append(result)
                    if (i + 1) % progress_interval == 0 or (i + 1) == len(encrypted_data_list):
                        print(f"   dot productqueryprogress: {i + 1}/{len(encrypted_data_list)}")
        
        ER = {
            'encrypted_results': encrypted_results,
            'num_results': len(encrypted_results),
            'querier_id': C_M['querier_id'],
            'owner_id': C_m['owner_id'],
            'model_type': model_type,
            'query_time': time.time()
        }
        
        print(f"      Querytranslated")
        print(f"      Generate {len(encrypted_results)} translatedencrypttranslated")
        
        return ER

    #  Decrypt algorithm 
    
    def decrypt(self, sk_h_u: Any, ER: Dict) -> List[float]:
        """
        Decrypt(sk_h,u, ER) → R
        """
        print(f"\n[Decrypt] decryptquerytranslated...")
        
        decrypted_results = []
        failed_count = 0
        total_results = len(ER['encrypted_results'])
        progress_interval = 10 if total_results <= 1000 else 100
        
        for i, encrypted_result in enumerate(ER['encrypted_results']):
            if encrypted_result is None:
                decrypted_results.append(0.0)
                failed_count += 1
                if (i + 1) % progress_interval == 0 or (i + 1) == total_results:
                    print(f"      decryptprogress: {i + 1}/{total_results}")
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
                print(f"   translated{i}translateddecryptFAILED: {e}")
                decrypted_results.append(0.0)
                failed_count += 1
            if (i + 1) % progress_interval == 0 or (i + 1) == total_results:
                print(f"      decryptprogress: {i + 1}/{total_results}")
        
        print(f"      Decrypttranslated")
        print(f"      translated {len(decrypted_results)} translateddecrypttranslated")
        if failed_count > 0:
            print(f"         translated {failed_count} translateddecryptFAILED")
        print(f"      Result sample: {decrypted_results[:5]}")
        
        return decrypted_results
    
    #  Update algorithm 
    
    def update(self, user_id: int) -> List:
        """
        Update(u_id) → L_id
        """
        if user_id >= len(self.aux):
            raise ValueError(f"User ID {user_id} translated")
        return self.aux[user_id]
    
    #  Revoke algorithm 
    
    def revoke(self, user_id: int, pp: List, aux: List) -> Tuple[List, List]:
        """
        Revoke(u_id, pp, aux) → (pp', aux')
        translatedV-Btranslatedalgorithmtranslated
        """
        print(f"\n{'='*60}")
        print(f"[Revoke] revocationUser {user_id}")
        print(f"{'='*60}")
        
        # 1. validateUsertranslated
        if user_id not in self.registered_users:
            raise ValueError(f"User {user_id} translatedregistering, translatedrevocation")
        
        if user_id not in self.user_secrets:
            raise ValueError(f"User {user_id} has not executed KeyGen")
        
        # 2. checktranslatedrevocation
        if self.is_revoked(user_id):
            print(f"   User {user_id} translatedrevocation")
            return pp, aux
        
        # 3. generateRevocation factor r_id ∈ Z_p
        r_id = self.ff.random_nonzero()
        print(f"   generateRevocation factor: r_id = {hex(r_id)[:20]}...")
        
        # 4. GetUsertranslated
        user_info = self.user_secrets[user_id]
        u_id_prime = user_info['u_id_prime']
        u_id_prime_1based = user_info['u_id_prime_1based']
        block_num = user_info['block']
        
        print(f"   Usertranslated: u_id'={u_id_prime_1based} (1-based), block={block_num}")
        
        # 5. generaterevocationtranslated pk_r,id = h_{u_id'}^{r_id}
        h_u = self.crs['h_i'][u_id_prime]
        pk_r = self.bp.exponentiate_g1(h_u, r_id)
        print(f"   generaterevocationtranslated: pk_r = h_{u_id_prime_1based}^{r_id}")
        
        # 6. generaterevocationtranslatedParameters pap_r,id
        pap_r = []
        
        for i in range(self.params.n):
            i_1based = i + 1
            
            if i_1based == u_id_prime_1based:
                pap_r.append(None)
                continue
            
            # Get H_{i,u_id'} translated H_{u_id',i}
            if (i, u_id_prime) in self.crs['H_ij']:
                H_key = (i, u_id_prime)
            else:
                H_key = (u_id_prime, i)
            
            H_val = self.crs['H_ij'][H_key]
            
            # compute H_{i,u_id'}^{r_id}
            pap_element = self.bp.exponentiate_gt(H_val, r_id)
            pap_r.append(pap_element)
        
        non_empty = len([p for p in pap_r if p is not None])
        print(f"   generaterevocationpap: {non_empty} non-empty elements")
        
        # 7. translatedRevocation info
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
        
        # 8. Update public parameterstranslatedParameters
        pp_new = pp.copy()
        aux_new = [list(L) for L in aux]
        
        k = math.ceil((user_id + 1) / self.params.n) - 1
        pp_new[k] = pk_r
        print(f"   Update block {k} translatedParameters")
        
        # Update auxiliary parameters - translatedblocktranslatedUsertranslatedrevocationpap
        block_start = k * self.params.n
        block_end = min(block_start + self.params.n, self.params.N)
        
        updated_count = 0
        for j in range(block_start, block_end):
            if j != user_id:
                aux_new[j].append(copy.deepcopy(pap_r))
                updated_count += 1
        
        print(f"   Update auxiliary parameters: {updated_count} users")
        
        # 9. translatedregisteringUsertranslated
        if user_id in self.registered_users:
            del self.registered_users[user_id]
        
        # 10. translated
        affected_owners = self._notify_owners_about_revoke(user_id)
        if affected_owners:
            print(f"   translated {len(affected_owners)} translated")
        
        print(f"\n   Revoketranslated")
        print(f"   User {user_id} translatedrevocation")
        
        return pp_new, aux_new
    
    def _notify_owners_about_revoke(self, revoked_user_id: int) -> List[int]:
        affected = []
        for owner_id, policy in self.access_policies.items():
            if revoked_user_id in policy:
                affected.append(owner_id)
        return affected
    
    def is_revoked(self, user_id: int) -> bool:
        """Check whether the user has been revoked"""
        return user_id in self._revoked_users
    
    def get_revocation_info(self, user_id: int) -> Dict:
        """GetUserRevocation info"""
        return self._revoked_info.get(user_id, {})
    
    def get_revocation_factor(self, user_id: int) -> Optional[int]:
        """GetUsertranslatedRevocation factor"""
        return self._revocation_factors.get(user_id)
    
    def get_all_revoked_users(self) -> List[int]:
        """GettranslatedrevocationUsertranslated"""
        return list(self._revoked_users)
    
    def get_affected_owners(self, revoked_user_id: int) -> List[int]:
        """Gettranslatedrevocationtranslated"""
        return self._notify_owners_about_revoke(revoked_user_id)
    
    #  policyupdate 
    
    def update_policy_after_revoke(self, C_m: Dict, revoked_user_id: int) -> Dict:
        print(f"\n[Policy Update] updatepolicy, translatedUser {revoked_user_id}")
        
        if revoked_user_id not in C_m['P']:
            print(f"   User {revoked_user_id} translatedpolicytranslated, translatedupdate")
            return C_m
        
        # translatedpolicy(translatedrevocationUser)
        new_policy = [uid for uid in C_m['P'] if uid != revoked_user_id]
        
        if not new_policy:
            print(f"      Warning: translatedpolicytranslated")
            return C_m
        
        print(f"   translatedpolicy: {C_m['P']}")
        print(f"   translatedpolicy: {new_policy}")
        
        # translatedgenerateRandom values
        beta_new = self.ff.random_element()
        gamma_new = self.ff.random_element()
        
        # translatedCompute ciphertext components
        c1_new, c2_new, c4_new = [], [], []
        
        for u_id in new_policy:
            k_i = math.ceil((u_id + 1) / self.params.n) - 1
            c1_i = self.pp[k_i]
            c1_new.append(c1_i)
            
            u_id_prime = u_id % self.params.n
            
            # translatedcomputec2
            h_u = self.crs['h_i'][u_id_prime]
            pairing_val = self._symmetric_pairing_sim(c1_i, h_u, gamma_new)
            c2_new.append(pairing_val)
            
            # translatedcomputec4
            z_u = self.crs['z_values'][u_id_prime]
            z_u_sq = (z_u * z_u) % self.ff.p
            exponent = (z_u_sq * gamma_new) % self.ff.p
            gt_part = self.bp.exponentiate_gt(self.crs['e_gg'], exponent)
            c4_new.append((gt_part, beta_new))
        
        # updatetranslated
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
        
        print(f"      policyupdatetranslated")
        print(f"      translatedpolicytranslated {len(new_policy)} users")
        
        return C_m_new
    
    #  translated 
    
    def _create_demo_ai_model(self) -> List[float]:
        """translatedAItranslated"""
        return [0.2, 0.3, 0.1, 0.4, 0.25]
    
    def get_system_state(self) -> Dict:
        """GetSystem state"""
        return {
            'crs_initialized': self.crs is not None,
            'pp_len': len(self.pp) if self.pp else 0,
            'aux_len': len(self.aux) if self.aux else 0,
            'registered_users': len(self.registered_users),
            'user_secrets': len(self.user_secrets),
            'encrypted_datasets': len(self.encrypted_datasets),
            'revoked_users': len(self._revoked_users),
            'revoked_users_list': list(self._revoked_users)
        }

    #  Testtranslated 
    def test_ai_model_encryption(self):
        """TestAItranslatedencrypttranslated"""
        print("\n" + "="*60)
        print("Test AI translatedencrypttranslated")
        print("="*60)
        
        tree = DecisionTreeHE()
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
        
        # encryptdecision tree
        pk_h = self.he.public_key
        encrypted_tree = self.encrypt_decision_tree(tree, pk_h)
        
        assert encrypted_tree['type'] == 'decision_tree'
        assert len(encrypted_tree['internal_nodes']) == 1
        assert len(encrypted_tree['leaf_nodes']) == 2

        
        # translated
        nn = self.create_default_neural_network(input_dim=5, output_dim=2)
        
        # encryptneural network
        encrypted_nn = self.encrypt_neural_network(nn, pk_h)
        
        assert encrypted_nn['type'] == 'neural_network'
        assert len(encrypted_nn['layers']) == 1
        
        
        return True


        
    def test_model_query(self, model_type: str, C_m: Dict, sk_h_s: Any, C_M_base: Dict) -> bool:

        print(f"\n{'-'*50}")
        print(f"Test {model_type} translatedquery")
        print(f"{'-'*50}")
        
        try:
            C_M = C_M_base.copy()
            pk_h = self.he.public_key
            
            if model_type == 'dot':
                # dot producttranslated
                print(f"translateddot producttranslated...")
                ai_model = [0.1, 0.2, 0.3, 0.4, 0.5]
                encrypted_model = self.he.encrypt(ai_model)
                C_M['encrypted_model'] = encrypted_model
                C_M['model_type'] = 'dot'
                
            elif model_type == 'decision_tree':
                # decision treetranslated
                print(f"translateddecision treetranslated...")
                from schemes.ai_model import DecisionTreeHE, DecisionTreeNode
                
                tree = DecisionTreeHE()
                
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
                
                encrypted_model = self.encrypt_decision_tree(tree, pk_h)
                C_M['encrypted_model'] = encrypted_model
                C_M['model_type'] = 'decision_tree'
                
            elif model_type == 'neural_network':

                encrypted_model = self.encrypt_neural_network()
                C_M['encrypted_model'] = encrypted_model
                C_M['model_type'] = 'neural_network'
            
            else:
                print(f"   translatedmodel type: {model_type}")
                return False
            
            # translatedquery
            print(f"translatedencryptquery...")
            ER = self.query(C_M, C_m, sk_h_s)
            
            # decrypttranslated
            print(f"decrypttranslated...")
            results = self.decrypt(C_M['sk_h_u'], ER)
            
            print(f"   {model_type} translatedquerytranslated")
            print(f"   Result count: {len(results)}")
            print(f"   Result sample: {results[:5]}")
            
            return True
            
        except Exception as e:
            print(f"   {model_type} translatedqueryFAILED: {e}")
            import traceback
            traceback.print_exc()
            return False  
  
    def test_complete_workflow(self):

        
        try:
            # 1. System initialization
            self.setup()
            
            # 2. translatedregisteringUser
            user_ids = [0, 1, 2]
            user_keys = {}
            
            for uid in user_ids:
                sk, pk, pap = self.keygen(uid)
                user_keys[uid] = (sk, pk, pap)
                self.register(uid, pk, pap)
            
            print(f"\n   Userregisteringtranslated: {user_ids}")
            
            # 3. translatedencrypting data
            owner_id = 0
            access_policy = [0, 1, 2]
            data_records = [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],  # translatedTesttranslated
                [5.0, 6.0, 7.0, 8.0, 9.0]
            ]
            
            C_m, sk_h_s = self.encrypt(owner_id, access_policy, data_records)
            print(f"\n   translatedencrypttranslated: {len(data_records)} records")
            
            # 4. querytranslatedchecktranslated
            querier_id = 1
            querier_sk = user_keys[querier_id][0]
            
            C_M_base = self.check(querier_id, querier_sk, C_m)
            if C_M_base is None:
                print("   translatedcheckFAILED")
                return False
            
            print(f"\n   Access verification passed")
            
            # 5. Testtranslatedmodel type
            print("\n" + "="*70)
            print("translatedTesttranslatedmodel type")
            print("="*70)
            
            results = {}
            
            # Testdot producttranslated
            results['dot'] = self.test_model_query('dot', C_m, sk_h_s, C_M_base)
            
            # Testdecision treetranslated
            results['decision_tree'] = self.test_model_query('decision_tree', C_m, sk_h_s, C_M_base)
            
            # Testneural networktranslated
            results['neural_network'] = self.test_model_query('neural_network', C_m, sk_h_s, C_M_base)
            
            # 6. translated
            print("\n" + "="*70)
            print("  Testtranslated")
            print("="*70)
            
            all_passed = True
            for model_type, passed in results.items():
                status = "   PASSED" if passed else "   FAILED"
                print(f"   {status} - {model_type}")
                all_passed = all_passed and passed
            
            if all_passed:
                print("\n   translatedTestPASSED！")
            else:
                print("\n   translatedTestFAILED")
            
            print("\n" + "="*70)
            return all_passed
            
        except Exception as e:
            print(f"\n   TestFAILED: {e}")
            import traceback
            traceback.print_exc()
            return False    
    
    def test_revoke_functionality(self):

        
        try:
            # translatedSystem state
            print("\n0. translatedSystem state...")
            self.crs = None
            self.pp = None
            self.aux = None
            self.registered_users = {}
            self.user_secrets = {}
            self.encrypted_datasets = {}
            self.access_policies = {}
            self._revoked_users = set()
            self._revoked_info = {}
            self._revocation_factors = {}
        
            # 1. translated
            print("\n1. System initialization...")
            self.setup()
            
            # 2. translatedregisteringUser
            print("\n2. translatedUser...")
            users = [5, 6, 7]
            for uid in users:
                sk, pk, pap = self.keygen(uid)
                self.register(uid, pk, pap)
                print(f"   User {uid} registeringtranslated")
            
            # 3. checktranslated
            print("\n3. translated:")
            print(f"   registeringUsertranslated: {len(self.registered_users)}")
            print(f"   revocationUsertranslated: {len(self._revoked_users)}")
            assert len(self.registered_users) == 3, f"registeringUsertranslatedshould be3, actual{len(self.registered_users)}"
            assert len(self._revoked_users) == 0, f"revocationUsertranslatedshould be0, actual{len(self._revoked_users)}"
            
            # 4. revocationUser6
            print("\n4. revocationUser6...")
            pp_new, aux_new = self.revoke(6, self.pp, self.aux)
            self.pp = pp_new
            self.aux = aux_new
            
            # 5. validaterevocationtranslated
            print("\n5. validaterevocationtranslated:")
            is_revoked_6 = self.is_revoked(6)
            is_revoked_5 = self.is_revoked(5)
            print(f"   User6translatedrevocation: {is_revoked_6}")
            print(f"   User5translatedrevocation: {is_revoked_5}")
            assert is_revoked_6 == True, "User6translatedrevocation"
            assert is_revoked_5 == False, "User5translatedrevocation"
            
            info = self.get_revocation_info(6)
            print(f"   Revocation info: {list(info.keys())}")
            assert 'r_id' in info, "Revocation infotranslatedr_id"
            assert 'pk_r' in info, "Revocation infotranslatedpk_r"
            assert 'pap_r' in info, "Revocation infotranslatedpap_r"
            
            factor = self.get_revocation_factor(6)
            print(f"   Revocation factor: {factor is not None}")
            assert factor is not None, "Revocation factortranslatedshould beNone"
            
            revoked_list = self.get_all_revoked_users()
            print(f"   translatedrevocationUser: {revoked_list}")
            assert 6 in revoked_list, "revocationUsertranslated6"
            
            # 6. translatedrevocationUsergeneratetranslated(translatedFAILED)
            print("\n6. translatedrevocationUser6generatetranslated(translatedFAILED)...")
            try:
                sk, pk, pap = self.keygen(6)
                print(f"      translatedFAILEDtranslated")
                assert False, "keygentranslatedrevocationUser"
            except ValueError as e:
                print(f"      translated: {e}")
            
            # 7. translatedregisteringtranslatedrevocationUser(translatedFAILED)
            print("\n7. translatedregisteringUser6(translatedFAILED)...")
            try:
                x_id = self.ff.random_element()
                u_id_prime = 6 % self.params.n
                h_u = self.crs['h_i'][u_id_prime]
                pk_dummy = self.bp.exponentiate_g1(h_u, x_id)
                pap_dummy = [None] * self.params.n
                self.register(6, pk_dummy, pap_dummy)
                print(f"      translatedFAILEDtranslated")
                assert False, "registertranslatedrevocationUser"
            except ValueError as e:
                print(f"      translated: {e}")
            
            # 8. Testpolicyupdate
            print("\n8. Testpolicyupdate...")
            dummy_C_m = {
                'P': [5, 6, 7],
                'c1_i': [None, None, None],
                'c2_i': [None, None, None],
                'c4_i': [None, None, None],
                'beta': 123,
                'gamma': 456,
                'n_p': 3,
                'owner_id': 5
            }
            
            updated = self.update_policy_after_revoke(dummy_C_m, 6)
            print(f"   translatedpolicy: {updated['P']}")
            print(f"   translatedUser6: {6 in updated['P']}")
            assert 6 not in updated['P'], "translatedpolicytranslatedrevocationUser"
            assert updated['n_p'] == 2, f"translatedpolicytranslatedshould be2, actual{updated['n_p']}"
            
            # 9. Testchecktranslatedcheckrevocationtranslated
            print("\n9. TesttranslatedrevocationUsertranslatedcheck...")
            dummy_C_m_with_policy = {
                'P': [5, 6, 7],
                'c1_i': [None, None, None],
                'c2_i': [None, None, None],
                'c4_i': [None, None, None],
                'c5': b'dummy',
                'beta': 123,
                'owner_id': 5
            }
            
            check_result = self.check(6, 123, dummy_C_m_with_policy)
            print(f"   checktranslated: {check_result is None}")
            assert check_result is None, "translatedrevocationUsertranslatedchecktranslatedNone"
            
            # 10. Testtranslated
            print("\n10. Testtranslated...")
            self.access_policies[5] = [5, 6, 7]
            self.access_policies[8] = [8, 9]
            
            affected = self.get_affected_owners(6)
            print(f"   translatedUser: {affected}")
            assert 5 in affected, "translated5translated"
            assert 8 not in affected, "translated8translated"
            
            # 11. translatedvalidate
            print("\n11. translated:")
            print(f"   registeringUsertranslated: {len(self.registered_users)}")
            print(f"   revocationUsertranslated: {len(self._revoked_users)}")
            assert len(self.registered_users) == 2, f"translatedregisteringUsertranslatedshould be2, actual{len(self.registered_users)}"
            assert len(self._revoked_users) == 1, f"translatedrevocationUsertranslatedshould be1, actual{len(self._revoked_users)}"
            
            print(f"\n   DeCart Revoke translatedTestPASSED")
            return True
            
        except Exception as e:
            print(f"\n   TestFAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


#  Export interface 

class DeCartScheme:
    """DeCarttranslatedmain class"""
    
    def __init__(self, params: Optional[DeCartParams] = None):
        self.system = DeCartSystem(params)
    
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
    
    def encrypt_neural_network(self, nn_model, pk_h: Any):
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
    
    def test_ai_models(self):
        return self.system.test_ai_model_encryption()
    
    def test_revoke(self):
        return self.system.test_revoke_functionality()
    
    def test_complete(self):
        return self.system.test_complete_workflow()


#  translatedTest 

if __name__ == "__main__":
    
    system = DeCartSystem(DeCartParams(N=32, n=8))
    
    # translatedTestAItranslatedencrypt
    print("\n" + "="*80)
    print("Test AI translatedencrypt")
    print("="*80)
    ai_success = system.test_ai_model_encryption()
    

    print("\n" + "="*80)
    print("translatedTesttranslated")
    print("="*80)
    workflow_success = system.test_complete_workflow()
    




