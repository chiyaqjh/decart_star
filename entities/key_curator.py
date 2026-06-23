# decart/entities/key_curator.py

import sys
import os
import copy
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import both schemes
from schemes.decart import DeCartSystem, DeCartParams
from schemes.decart_star import DeCartStarSystem, DeCartStarParams
from core.bilinear_pairing import BilinearPairing
from core.finite_field import FiniteField
from config import Config


class KeyCurator:
    """
    Key Curator

    1. System initialization - Setup(λ) → (crs, pp, aux)
    2. User registration - Register(u_id, pk_id, pap_id) → (pp', aux')
    3. User revocation - Revoke(u_id, pp, aux) → (pp', aux')
    4. Maintain public parameters C_{(k)}
    5. Maintain auxiliary parameters L_j
    """
    
    def __init__(self, 
                 scheme: str = "decart_star",  
                 params: Optional[Union[DeCartParams, DeCartStarParams]] = None):
        self.scheme = scheme.lower()
        
        # ===== Select the corresponding system core based on the scheme =====
        if self.scheme == "decart":
            from schemes.decart import DeCartSystem, DeCartParams
            self.params = params or DeCartParams(N=Config.MAX_USERS, n=Config.BLOCK_SIZE)
            self.system = DeCartSystem(self.params)
            self.scheme_name = "DeCart (O(n²))"
            
        elif self.scheme == "decart_star":
            from schemes.decart_star import DeCartStarSystem, DeCartStarParams
            self.params = params or DeCartStarParams(N=Config.MAX_USERS, n=Config.BLOCK_SIZE)
            self.system = DeCartStarSystem(self.params)
            self.scheme_name = "DeCart* (O(n) optimization)"
            
        else:
            raise ValueError(f"Unknown scheme: {scheme}. Please use 'decart' or 'decart_star'")
        
        # ===== System state - shared by all schemes =====
        self.crs = None
        self.pp = None
        self.aux = None
        
        # ===== Registry =====
        self.user_public_keys = {}      # user_id -> pk_id
        self.user_blocks = {}           # user_id -> block
        self.user_id_prime = {}         # user_id -> u_id'
        self.user_pap = {}             # user_id -> pap_id
        self.registered_users = set()   # Registered users
        self.registration_time = {}    # user_id -> timestamp
        
        # ===== Cross-block trust management (entity-layer enhancement) =====
        self._trust_map = {}            # trustee_id -> Set[truster_id]
        self._trust_time = {}           # Trust creation time
        
        # ===== Revocation-related state =====
        self._revoked_users = set()           # Revoked users
        self._revoked_info = {}                # Revocation info {user_id: info}
        self._revocation_time = {}             # Revocation time
        
        # ===== Statistics =====
        self.stats = {
            'scheme': self.scheme_name,
            'total_users': 0,
            'total_revoked': 0,
            'total_blocks': self.params.B,
            'setup_complete': False,
            'start_time': time.time(),
            'trust_relations': 0,
            'cross_block_updates': 0,
            'revoke_operations': 0
        }
        
        print(f"\n  Key Curator entity initialized")
        print(f"   Scheme: {self.scheme_name}")
        print(f"   Parameters: N={self.params.N}, n={self.params.n}, B={self.params.B}")
        print(f"   Supports Revoke functionality")
    
    #  Setup
    
    def setup(self) -> Tuple[Dict, List, List]:

        print("\n" + "="*60)
        print(f"[Key Curator] System initialization - {self.scheme_name}")
        print("="*60)
        
        self.crs, self.pp, self.aux = self.system.setup()
        self.stats['setup_complete'] = True
        self.stats['setup_time'] = time.time()
        
        # Scheme-specific statistics
        if self.scheme == "decart":
            h_count = len(self.crs['h_i'])
            H_count = len(self.crs['H_ij'])
            print(f"   crs: h_i={h_count}, H_ij={H_count}, total={h_count + H_count}")
        else:  # decart_star
            h_count = len([h for h in self.crs['h_i'] if h])
            print(f"   crs: h_i={h_count} (O(n) optimization)")
        
        print(f"   pp: {len(self.pp)} block parameters")
        print(f"   aux: {len(self.aux)} user slots")
        
        return self.crs, self.pp, self.aux
    
    #  Key generation
    
    def generate_user_key(self, user_id: int) -> Tuple[int, Any, List[Optional[Any]]]:

        if self.crs is None:
            raise ValueError("Please run setup() first")
        
        if not (0 <= user_id < self.params.N):
            raise ValueError(f"User ID must be in [0, {self.params.N-1}]")
        
        print(f"\n[KeyGen] User {user_id} generating keys - {self.scheme_name}")
        return self.system.keygen(user_id)
    
    #  Register
    
    def register(self, user_id: int, pk_id: Any, pap_id: List[Optional[Any]]) -> bool:

        print(f"\n[Key Curator] Handling registration request for user {user_id} - {self.scheme_name}")
        
        try:
            # ===== Check whether the user has been revoked =====
            if self.is_revoked(user_id):
                print(f"    User {user_id} has been revoked and cannot be re-registered")
                return False
            
            # Verify whether the user is already registered
            if user_id in self.registered_users:
                print(f"    User {user_id} is already registered")
                return False
            
            # Verify whether the user has executed KeyGen
            if user_id not in self.system.user_secrets:
                print(f"    User {user_id} has not executed KeyGen")
                return False
            
            # ===== Step 1: call the paper's Register algorithm (same-block updates only) =====
            print(f"   [Algorithm layer] Running {self.scheme_name} Register algorithm...")
            self.pp, self.aux = self.system.register(user_id, pk_id, pap_id)
            
            # ===== Step 2: entity-layer enhancement - cross-block trust updates =====
            trusted_by = self.get_trusted_by(user_id)
            if trusted_by:
                print(f"   [Entity layer] Found {len(trusted_by)} users trusting user {user_id}")
                
                cross_block_count = 0
                for truster_id in trusted_by:
                    if truster_id != user_id and truster_id < len(self.aux):
                        self.aux[truster_id].append(copy.deepcopy(pap_id))
                        cross_block_count += 1
                        self.stats['cross_block_updates'] += 1
                
                print(f"   [Entity layer] Completed {cross_block_count} cross-block aux updates")
            
            # ===== Step 3: fetch user info and update the registry =====
            user_info = self.system.user_secrets.get(user_id)
            if not user_info:
                print(f"    User {user_id} key information does not exist")
                return False
            
            block_num = user_info['block']
            u_id_prime = user_info['u_id_prime']
            
            self.user_public_keys[user_id] = pk_id
            self.user_blocks[user_id] = block_num
            self.user_id_prime[user_id] = u_id_prime
            self.user_pap[user_id] = pap_id
            self.registered_users.add(user_id)
            self.registration_time[user_id] = time.time()
            self.stats['total_users'] += 1
            
            print(f"     User {user_id} registered successfully")
            print(f"      Block: {block_num}, u_id'={u_id_prime}")
            print(f"      Registered users: {self.stats['total_users']}")
            
            return True
            
        except Exception as e:
            print(f"    Registration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    #  Revoke
    
    def revoke_user(self, user_id: int) -> bool:

        print(f"\n[Key Curator] Revoking user {user_id} - {self.scheme_name}")
        
        try:
            # 1. Verify whether the user exists
            if user_id not in self.registered_users:
                print(f"    User {user_id} is not registered and cannot be revoked")
                return False
            
            # 2. Check whether the user is already revoked
            if self.is_revoked(user_id):
                print(f"    User {user_id} has already been revoked")
                return True
            
            # 3. Call the Revoke algorithm of the selected scheme
            self.pp, self.aux = self.system.revoke(user_id, self.pp, self.aux)
            
            # 4. Update local revocation state
            self._revoked_users.add(user_id)
            self._revocation_time[user_id] = time.time()
            
            # 5. Remove from the registry
            if user_id in self.registered_users:
                self.registered_users.remove(user_id)
                self.stats['total_users'] -= 1
            
            self.stats['total_revoked'] += 1
            self.stats['revoke_operations'] += 1
            
            # 6. Retrieve revocation info (for debugging)
            revoke_info = self.system.get_revocation_info(user_id)
            self._revoked_info[user_id] = revoke_info
            
            # 7. Retrieve affected owners
            affected_owners = self.system.get_affected_owners(user_id)
            if affected_owners:
                print(f"    Notifying {len(affected_owners)} owners to update policies")
            
            print(f"\n  User {user_id} revoked successfully")
            print(f"   Registered users: {self.stats['total_users']}")
            print(f"   Total revoked users: {self.stats['total_revoked']}")
            
            return True
            
        except Exception as e:
            print(f"    Revocation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_revoked(self, user_id: int) -> bool:

        # Check local cache first
        if user_id in self._revoked_users:
            return True
        # Then check the system core
        if hasattr(self.system, 'is_revoked'):
            return self.system.is_revoked(user_id)
        return False
    
    def get_revoked_users(self) -> List[int]:

        revoked = list(self._revoked_users)
        if hasattr(self.system, 'get_all_revoked_users'):
            system_revoked = self.system.get_all_revoked_users()
            # Merge and deduplicate
            revoked = list(set(revoked + system_revoked))
        return revoked
    
    def get_revocation_info(self, user_id: int) -> Dict:

        if user_id in self._revoked_info:
            return self._revoked_info[user_id]
        if hasattr(self.system, 'get_revocation_info'):
            return self.system.get_revocation_info(user_id)
        return {}
    
    def get_affected_owners(self, revoked_user_id: int) -> List[int]:

        if hasattr(self.system, 'get_affected_owners'):
            return self.system.get_affected_owners(revoked_user_id)
        return []
    
    def update_policy_after_revoke(self, owner_id: int, C_m: Dict, revoked_user_id: int) -> Optional[Dict]:

        if hasattr(self.system, 'update_policy_after_revoke'):
            return self.system.update_policy_after_revoke(C_m, revoked_user_id)
        return None
    
    #  Entity-layer enhancement: cross-block trust management
    
    def add_trust(self, truster_id: int, trustee_id: int) -> bool:

        if truster_id == trustee_id:
            print(f"    A user cannot trust themselves")
            return False
        
        if trustee_id not in self._trust_map:
            self._trust_map[trustee_id] = set()
        
        if truster_id not in self._trust_map[trustee_id]:
            self._trust_map[trustee_id].add(truster_id)
            self._trust_time[f"{truster_id}->{trustee_id}"] = time.time()
            self.stats['trust_relations'] += 1
            
            print(f"   [Trust] User {trustee_id} ← User {truster_id}")
            
            # If the trustee is already registered, update immediately
            if trustee_id in self.registered_users and trustee_id in self.user_pap:
                pap_id = self.user_pap[trustee_id]
                if truster_id < len(self.aux):
                    self.aux[truster_id].append(copy.deepcopy(pap_id))
                    self.stats['cross_block_updates'] += 1
            
            return True
        return False
    
    def get_trusted_by(self, user_id: int) -> Set[int]:

        return self._trust_map.get(user_id, set())
    
    #  Query interface
    
    def get_user_aux(self, user_id: int) -> List:

        if not self.aux or user_id >= len(self.aux):
            return []
        return self.aux[user_id]
    
    def get_block_public_key(self, block_id: int) -> Any:

        if not self.pp or block_id >= len(self.pp):
            raise ValueError(f"Block {block_id} does not exist")
        return self.pp[block_id]
    
    def get_system_info(self) -> Dict:

        return {
            'scheme': self.scheme_name,
            'N': self.params.N,
            'n': self.params.n,
            'B': self.params.B,
            'registered_users': len(self.registered_users),
            'revoked_users': len(self._revoked_users),
            'setup_complete': self.stats['setup_complete'],
            'trust_relations': self.stats['trust_relations'],
            'cross_block_updates': self.stats['cross_block_updates'],
            'revoke_operations': self.stats['revoke_operations']
        }
    
    def switch_scheme(self, scheme: str) -> bool:

        if scheme.lower() == self.scheme:
            print(f"   Already using {self.scheme_name}")
            return True
        
        print(f"\n Switching scheme: {self.scheme_name} → ", end="")
        self.__init__(scheme, self.params)
        print(f"{self.scheme_name}")
        return True


#  Test code

def test_key_curator_with_both_schemes():
    
    # 1. Test the DeCart scheme
    print("\n Testing DeCart scheme...")
    curator1 = KeyCurator(scheme="decart", params=DeCartParams(N=64, n=16))
    curator1.setup()
    
    # 2. Test the DeCart* scheme
    print("\n Testing DeCart* scheme...")
    curator2 = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator2.setup()
    
    # 3. Comparison test
    print("\n" + "="*80)
    print(" Scheme comparison")
    print("="*80)
    
    info1 = curator1.get_system_info()
    info2 = curator2.get_system_info()
    
    print(f"\n   DeCart  : {info1['scheme']}")
    print(f"   DeCart* : {info2['scheme']}")
    print(f"\n   Both schemes initialized successfully ✓")
    
    return curator1, curator2

def test_revoke_functionality():
    
    print("\n" + "="*80)
    print(" Testing Key Curator Revoke functionality")
    print("="*80)
    
    # Test with the DeCart* scheme
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 1. Create and register users
    print("\n1. Create users...")
    users = [5, 6, 7, 8]  # Add one more user as data owner
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)
    
    assert len(curator.registered_users) == 4, "Registered user count should be 4"
    assert len(curator.get_revoked_users()) == 0, "Revoked user count should be 0"
    
    # 2. Establish trust relationships
    print("\n2. Establish trust relationships...")
    curator.add_trust(6, 5)  # 6 trusts 5
    curator.add_trust(7, 5)  # 7 trusts 5
    
    # 3. Create encrypted dataset mocks (simulate owner 8 creating a policy containing user 5)
    print("\n3. Create simulated encrypted datasets...")
    # Directly operate on system.access_policies
    curator.system.access_policies[8] = [5, 6, 7]  # Owner 8's policy contains user 5
    curator.system.access_policies[5] = [5, 6]     # Owner 5's policy contains self
    curator.system.access_policies[6] = [6, 7]     # Owner 6's policy does not contain user 5
    print(f"   Created 3 simulated encrypted datasets")
    
    # 4. Revoke user 5
    print("\n4. Revoke user 5...")
    success = curator.revoke_user(5)
    assert success, "Revocation failed"
    
    # 5. Verify state
    print("\n5. Verify state...")
    assert curator.is_revoked(5), "User 5 should be marked as revoked"
    assert not curator.is_revoked(6), "User 6 should not be revoked"
    
    revoked_list = curator.get_revoked_users()
    print(f"   Revoked users: {revoked_list}")
    assert 5 in revoked_list, "Revocation list should contain 5"
    
    info = curator.get_revocation_info(5)
    print(f"   Revocation info: {list(info.keys())}")
    
    # 6. Try generating new keys for revoked user 5 (expected to fail)
    print("\n6. Try generating new keys for revoked user 5 (expected failure)...")
    try:
        sk, pk, pap = curator.generate_user_key(5)
        print(f"     Error: should have failed but succeeded")
        assert False, "generate_user_key should reject revoked users"
    except ValueError as e:
        print(f"     Correctly rejected: {e}")
    
    # 7. Try re-registering revoked user 5 (expected to fail)
    print("\n7. Try re-registering user 5 (expected failure)...")
    try:
        success = curator.register(5, "dummy_pk", [None] * curator.params.n)
        assert not success, "register should return False"
        print(f"     register returned False and correctly rejected")
    except Exception as e:
        print(f"     Correctly rejected (raised exception): {e}")
    
    # 8. Get affected owners
    print("\n8. Get affected owners...")
    affected = curator.get_affected_owners(5)
    print(f"   Affected owners: {affected}")
    assert 8 in affected, "Owner 8 should be affected (policy contains user 5)"
    assert 5 in affected, "Owner 5 should be affected (policy contains self)"
    assert 6 not in affected, "Owner 6 should not be affected (policy does not contain user 5)"
    
    # 9. Policy update example
    print("\n9. Test policy update...")
    if 8 in affected:
        # Simulate owner 8 updating the policy
        dummy_C_m = {
            'P': [5, 6, 7],
            'c1_i': [None, None, None],
            'c2_i': [None, None, None],
            'c4_i': [None, None, None],
            'beta': 123,
            'gamma': 456,
            'n_p': 3,
            'owner_id': 8
        }
        updated = curator.update_policy_after_revoke(8, dummy_C_m, 5)
        if updated:
            print(f"   Policy updated successfully, new policy: {updated.get('P', [])}")
            assert 5 not in updated.get('P', []), "The new policy should not contain the revoked user"
    
    # 10. Final state
    print("\n10. Final state:")
    print(f"    Registered user count: {len(curator.registered_users)}")
    print(f"    Revoked user count: {len(curator.get_revoked_users())}")
    print(f"    Revoke operation count: {curator.stats['revoke_operations']}")
    print(f"    Affected owner count: {len(affected)}")
    
    print(f"\n  Key Curator Revoke test passed")
    
    return curator

if __name__ == "__main__":
    
    # Run tests
    test_key_curator_with_both_schemes()
    test_revoke_functionality()
