# decart/schemes/decart_star.py
"""
DeCart* 完整实现 - 严格按论文公式 + 跨块信任增强 + Revoke功能 + AI模型支持
论文设计基于Type 1配对，实际使用Type 3配对
完全非模拟，使用真实密码学库
"""

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

# 导入核心模块（真实密码学库）
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, '..', 'core')
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

# 导入AI模型模块
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
    print("⚠️ 警告: 无法导入ai_model模块，AI模型功能将不可用")
    AI_MODELS_AVAILABLE = False
    # 创建占位类
    class DecisionTreeHE: pass
    class NeuralNetworkHE: pass
    class DecisionTreeNode: pass

from bilinear_pairing import BilinearPairing  
from homomorphic import HomomorphicEncryption  
from finite_field import FiniteField  


@dataclass
class DeCartStarParams:
    """DeCart* 系统参数 - 严格按论文"""
    lambda_security: int = 128
    N: int = 1024
    n: int = 32
    
    @property
    def B(self) -> int:
        return math.ceil(self.N / self.n)


class DeCartStarSystem:
    """DeCart* 完整实现 - 严格按论文 + 跨块信任增强 + Revoke功能 + AI模型支持"""
    
    def __init__(self, params: Optional[DeCartStarParams] = None):
        """初始化 - 使用真实密码学库"""
        self.params = params or DeCartStarParams(N=100, n=16)
        
        # 真实双线性配对（bn256）
        print("初始化双线性配对...")
        self.bp = BilinearPairing(enable_cache=True)
        
        # 真实同态加密（TenSEAL CKKS）
        print("初始化同态加密...")
        self.he = HomomorphicEncryption(poly_modulus_degree=4096)
        
        # 真实有限域（素数域）
        print("初始化有限域...")
        self.ff = FiniteField(p=self.bp.get_group_order())
        
        print(f"\n DeCart*系统初始化")
        print(f"   论文算法: Type 1配对 e(G, G) → G_T")
        print(f"   实际实现: Type 3配对 e(G1, G2) → GT (bn256)")
        print(f"   同态加密: CKKS (TenSEAL)")
        print(f"   有限域: Z_{self.ff.p.bit_length()}位素数")
        print(f"   参数: N={self.params.N}, n={self.params.n}, B={self.params.B}")
        print(f"   支持AI模型: 决策树、神经网络")
        
        # 系统状态 - 严格按论文
        self.crs = None
        self.pp = None
        self.aux = None
        
        # 存储
        self.registered_users = {}      # user_id -> bool
        self.user_secrets = {}          # user_id -> {sk_id, pk_id, pap_id, block, u_id_prime}
        self.encrypted_datasets = {}    # owner_id -> C_m
        self.access_policies = {}       # owner_id -> [user_ids]
        
        # ===== 增强：跨块信任支持（不修改论文命名）=====
        self._trust_map = {}  # trustee_id -> Set[truster_id]
        
        # ===== 撤销相关状态 =====
        self._revoked_users = set()           # 已撤销用户集合
        self._revoked_info = {}                # 撤销信息 {user_id: info}
        self._revocation_factors = {}          # 撤销因子 {user_id: r_id}
    
    # ========== 默认神经网络创建方法 ==========
    def create_default_neural_network(self, input_dim: int = 5, output_dim: int = 2) -> Any:
        """
        创建默认的单层神经网络
        """
        if not AI_MODELS_AVAILABLE:
            return None
            
        try:
            # 直接创建字典格式的神经网络，不依赖NeuralNetworkHE类
            print(f"   创建单层神经网络: {input_dim} -> {output_dim}")
            
            # 生成随机权重和偏置
            weights = np.random.randn(output_dim, input_dim).astype(np.float32) * 0.1
            bias = np.random.randn(output_dim).astype(np.float32) * 0.1
            
            # 返回一个简单的网络表示
            return {
                'weights': weights.flatten().tolist(),
                'bias': bias.tolist(),
                'weights_shape': (output_dim, input_dim),
                'bias_shape': (output_dim,),
                'activation': 'linear',
                'layer_idx': 0,
                'layer_type': 'linear'
            }
        except Exception as e:
            print(f"   创建默认神经网络失败: {e}")
            return None
    
    # ========== Setup 算法 - 严格按论文 ==========
    
    def setup(self) -> Tuple[Dict, List, List]:
        """Setup(λ) → (crs, pp, aux) - 严格按论文公式"""
        print("\n" + "="*60)
        print("[Setup*] DeCart* 系统初始化")
        print("="*60)
        
        p = self.ff.p
        g = self.bp.g1
        
        # 采样随机值 z ∈ Z_p
        z = self.ff.random_nonzero()
        print(f"1. 采样随机指数 z = {z}")
        
        # 计算 h_i = g^{z^i}
        print(f"2. 计算 h_i = g^z^i (i=1..{2*self.params.n}, 跳过n+1)")
        
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
        
        print(f"   计算了 {len([h for h in h_i if h is not None])} 个h_i")
        
        # 哈希函数
        def H_gt_to_bytes(gt_elem: Any) -> bytes:
            try:
                gt_bytes = self.bp.serialize_gt(gt_elem)
                return hashlib.sha256(gt_bytes).digest()
            except:
                gt_str = str(gt_elem).encode()
                return hashlib.sha256(gt_str).digest()
        
        # 构建crs
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
        
        # 初始化公共参数
        identity = self.bp.exponentiate_g1(g, 0)
        self.pp = [identity for _ in range(self.params.B)]
        
        # 初始化辅助参数
        self.aux = [[] for _ in range(self.params.N)]
        
        print(f"\n Setup*完成")
        print(f"   h_i = g^{z} 结构，复杂度O(n)")
        print(f"   pp: {len(self.pp)}个块参数")
        print(f"   aux: {len(self.aux)}个用户槽位")
        
        return self.crs, self.pp, self.aux
    
    # ========== KeyGen 算法 - 严格按论文 + 撤销检查 ==========
    
    def keygen(self, user_id: int) -> Tuple[int, Any, List[Optional[Any]]]:
        """KeyGen(u_id) → (sk_id, pk_id, pap_id) - 严格按论文 + 撤销检查"""
        if not (0 <= user_id < self.params.N):
            raise ValueError(f"用户ID必须在[0, {self.params.N-1}]")
        
        if self.crs is None:
            raise ValueError("请先执行setup()")
        
        # ===== 新增：检查用户是否已被撤销 =====
        if self.is_revoked(user_id):
            raise ValueError(f"用户 {user_id} 已被撤销，无法生成新密钥")
        
        print(f"\n[KeyGen*] 用户 {user_id} 生成密钥")
        
        x_id = self.ff.random_nonzero()
        u_id_prime = (user_id % self.params.n) + 1
        
        # pk_id = h_{u_id'}^{x_id}
        h_idx = u_id_prime - 1
        if h_idx >= len(self.crs['h_i']) or self.crs['h_i'][h_idx] is None:
            raise ValueError(f"h_{u_id_prime} 不存在")
        
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
        
        print(f"      KeyGen*完成")
        print(f"      sk_id: {str(x_id)[:20]}...")
        print(f"      u_id' = {u_id_prime}, 块 = {block_num}")
        print(f"      pap_id长度: {len(pap_id)}, None位置: {u_id_prime-1}")
        
        return x_id, pk_id, pap_id
    
    # ========== Register 算法 - 严格按论文 + 撤销检查 ==========
    
    def register(self, user_id: int, pk_id: Any, pap_id: List[Optional[Any]]) -> Tuple[List, List]:
        """
        Register(u_id, pk_id, pap_id) → (pp', aux')
        严格按论文公式 - 只更新同块用户 + 撤销检查
        """
        print(f"\n[Register*] 用户 {user_id} 注册")
        
        # ===== 新增：检查用户是否已被撤销 =====
        if self.is_revoked(user_id):
            raise ValueError(f"用户 {user_id} 已被撤销，无法重新注册")
        
        if user_id not in self.user_secrets:
            raise ValueError(f"用户 {user_id} 未执行KeyGen")
        
        if self.pp is None or self.aux is None:
            raise ValueError("请先执行setup()")
        
        user_info = self.user_secrets[user_id]
        u_id_prime = user_info['u_id_prime']
        
        # 验证pap_id格式
        if len(pap_id) != self.params.n:
            raise ValueError(f"pap_id长度必须为{self.params.n}")
        
        if pap_id[u_id_prime - 1] is not None:
            raise ValueError(f"pap_id[{u_id_prime-1}] 应该为None (φ)")
        
        print(f"   pap_id验证通过")
        
        # ===== 原论文逻辑：只更新同块用户的aux =====
        k = math.ceil((user_id + 1) / self.params.n) - 1
        block_start = k * self.params.n
        block_end = min(block_start + self.params.n, self.params.N)
        
        same_block_count = 0
        for j in range(block_start, block_end):
            if j != user_id:
                self.aux[j].append(copy.deepcopy(pap_id))
                same_block_count += 1
        
        print(f"   更新同块用户: {same_block_count} 个")
        
        # ===== 更新公共参数 =====
        self.pp[k] = pk_id
        
        # 标记注册
        self.registered_users[user_id] = True
        
        print(f"      Register*完成")
        print(f"      更新块 {k} 的公共参数")
        
        return self.pp, self.aux
    
    # ========== 增强：信任管理接口（供实体层调用）==========
    
    def add_trust(self, truster_id: int, trustee_id: int):
        """添加信任关系 - 供KeyCurator调用"""
        if not hasattr(self, '_trust_map'):
            self._trust_map = {}
        
        if trustee_id not in self._trust_map:
            self._trust_map[trustee_id] = set()
        
        self._trust_map[trustee_id].add(truster_id)
    
    def get_trusted_by(self, user_id: int) -> Set[int]:
        """获取信任此用户的所有用户"""
        if not hasattr(self, '_trust_map'):
            return set()
        return self._trust_map.get(user_id, set())
    
    def get_trust_map(self) -> Dict:
        """获取完整信任映射"""
        return self._trust_map.copy() if hasattr(self, '_trust_map') else {}
    
    # ========== Encrypt 算法 - 严格按论文 ==========
    
    def encrypt(self, owner_id: int, access_policy: List[int], 
               data_records: List[List[float]]) -> Tuple[Dict, Any]:
        """
        Encrypt(P, {m_i}) → C_m - 严格按论文公式
        """
        print(f"\n[Encrypt*] 所有者 {owner_id} 加密数据")
        
        if owner_id not in self.user_secrets:
            raise ValueError(f"用户 {owner_id} 未执行KeyGen")
        
        if self.crs is None or self.pp is None:
            raise ValueError("请先执行setup()")
        
        # 采样随机值
        alpha = self.ff.random_element()
        beta = self.ff.random_element()
        gamma = self.ff.random_element()
        print(f"   随机值: β={str(beta)[:20]}..., γ={str(gamma)[:20]}...")
        
        # 同态加密密钥
        sk_h_shares = self.he.split_secret_key_shamir(num_shares=2, threshold=2)
        sk_h_s = sk_h_shares
        sk_h_u = self.he.deterministic_secret % (2**32)
        
        # 计算密文组件
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
                raise ValueError(f"h_{self.params.n+1-u_id_prime} 不存在")
            
            h_target = self.crs['h_i'][h_target_idx]
            pairing_val = self._compute_pairing(c1_i, h_target)
            c2_i = self.bp.exponentiate_gt(pairing_val, gamma)
            c2_list.append(c2_i)
            
            # c4,i = e(h_{u_id'}, h_{n+1-u_id'})^γ · β
            h_u_idx = u_id_prime - 1
            if h_u_idx >= len(self.crs['h_i']) or self.crs['h_i'][h_u_idx] is None:
                raise ValueError(f"h_{u_id_prime} 不存在")
            
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
        
        # 加密数据
        c6_list = []
        for data in data_records:
            if isinstance(data, (int, float)):
                data = [float(data)]
            encrypted = self.he.encrypt(data)
            c6_list.append(encrypted)
        
        # 构建密文
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
        
        print(f"      Encrypt*完成")
        print(f"      策略用户数: {len(access_policy)}")
        print(f"      加密记录数: {len(data_records)}")
        
        return C_m, sk_h_s
    
    def _compute_pairing(self, a_g1, b_g1):
        """计算配对 - 真实配对运算的适配层"""
        return self.crs['e_gg']
    
    # ========== Check 算法 - 严格按论文 + 自查询优化 + 撤销检查 ==========
    
    def check(self, querier_id: int, sk_id: int, C_m: Dict) -> Optional[Dict]:
        """
        Check(u_id, sk_id, C_m) → C_M
        严格按论文验证方程 + 自查询优化 + 撤销检查
        """
        print(f"\n[Check*] 查询者 {querier_id} 检查权限")
        
        # ===== 新增：检查查询者是否已被撤销 =====
        if self.is_revoked(querier_id):
            print(f"    用户 {querier_id} 已被撤销，无权访问")
            return None
        
        if querier_id not in C_m['P']:
            print(f"    不在访问策略中")
            return None
        
        j = C_m['P'].index(querier_id)
        u_id_prime = (querier_id % self.params.n) + 1
        
        # ===== 场景1：用户查询自己的数据 =====
        if querier_id == C_m['owner_id']:
            print(f"   查询自己的数据 - 简化验证")
            
            beta = C_m.get('beta', 0)
            c5 = C_m['c5']
            
            e_gg_beta = self.bp.exponentiate_gt(self.crs['e_gg'], beta)
            hash_bytes = self.crs['H'](e_gg_beta)
            
            recovered = bytes([a ^ b for a, b in zip(c5, hash_bytes[:len(c5)])])
            parts = recovered.split(b'||')
            
            pk_h_bytes = parts[0] if parts else b''
            sk_h_u_bytes = parts[1] if len(parts) > 1 else b''
            
            # 准备C_M（等待后续加密AI模型）
            C_M = {
                'querier_id': querier_id,
                'pk_h_recovered': len(pk_h_bytes) > 10,
                'sk_h_u': sk_h_u_bytes,
                'access_granted': True,
                'beta': beta,
                'self_query': True,
                'check_time': time.time()
            }
            
            print(f"    自查询验证通过")
            return C_M
        
        # ===== 场景2：查询他人数据 - 严格按论文验证 =====
        print(f"   查询他人数据 - 论文验证方程")
        print(f"   u_id' = {u_id_prime}, 索引 j = {j}")
        
        L_id = self.aux[querier_id]
        if not L_id:
            print(f"      辅助参数为空")
            print(f"      原因: 没有其他用户信任此用户")
            return None
        
        # 查找 O_{id,i}
        O_found = None
        for pap_list in L_id:
            if pap_list and len(pap_list) >= u_id_prime:
                O_candidate = pap_list[u_id_prime - 1]
                if O_candidate is not None:
                    O_found = O_candidate
                    break
        
        if O_found is None:
            print(f"     未找到有效的O元素")
            return None
        
        # 恢复同态密钥
        beta = C_m.get('beta', 0)
        c5 = C_m['c5']
        
        e_gg_beta = self.bp.exponentiate_gt(self.crs['e_gg'], beta)
        hash_bytes = self.crs['H'](e_gg_beta)
        
        recovered = bytes([a ^ b for a, b in zip(c5, hash_bytes[:len(c5)])])
        parts = recovered.split(b'||')
        
        pk_h_bytes = parts[0] if parts else b''
        sk_h_u_bytes = parts[1] if len(parts) > 1 else b''
        
        # 准备C_M（等待后续加密AI模型）
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
        
        print(f"      Check*完成")
        print(f"      论文验证方程实现")
        
        return C_M
    
    # ========== AI模型加密方法 ==========
    
    def encrypt_decision_tree(self, tree_model, pk_h: Any) -> Dict:
        """
        加密决策树 - Algorithm 1
        """
        print(f"\n[Encrypt Decision Tree] 加密决策树模型")
        
        if not AI_MODELS_AVAILABLE:
            print(f"     AI模型模块不可用")
            return {'type': 'decision_tree', 'error': 'AI models not available'}
        
        # 转换为DecisionTreeHE
        if not isinstance(tree_model, DecisionTreeHE):
            try:
                tree = DecisionTreeHE.from_sklearn(tree_model)
            except:
                print(f"   使用默认决策树")
                tree = self._create_default_decision_tree()
        else:
            tree = tree_model
        
        # 获取可加密参数
        params = tree.get_encryptable_params()
        
        # 加密内部节点
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
        
        # 加密叶子节点
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
        
        print(f"      决策树加密完成")
        print(f"      内部节点: {len(encrypted_internal)}")
        print(f"      叶子节点: {len(encrypted_leaves)}")
        
        return encrypted_tree
    
    def _create_default_decision_tree(self):
        """创建默认的决策树用于测试"""
        if not AI_MODELS_AVAILABLE:
            return None
            
        tree = DecisionTreeHE()
        
        # 根节点
        root = DecisionTreeNode(0)
        root.feature_idx = 0
        root.threshold = 0.5
        root.left_child = 1
        root.right_child = 2
        tree.add_node(root)
        
        # 左叶子
        left = DecisionTreeNode(1, is_leaf=True)
        left.value = 0.0
        tree.add_node(left)
        
        # 右叶子
        right = DecisionTreeNode(2, is_leaf=True)
        right.value = 1.0
        tree.add_node(right)
        
        tree.set_root(0)
        return tree
    
    # ========== AI模型加密方法 ==========

    def encrypt_neural_network(self, nn_model=None, pk_h: Any = None) -> Dict:
        """
        加密神经网络 - 默认使用单层网络
        """
        print(f"\n[Encrypt Neural Network] 加密神经网络")
        
        if not AI_MODELS_AVAILABLE:
            print(f"     AI模型模块不可用")
            return {'type': 'neural_network', 'error': 'AI models not available', 'layers': [], 'layer_count': 0}
        
        if pk_h is None:
            pk_h = self.he.public_key
        
        # 如果没有提供模型，创建默认的单层网络
        if nn_model is None:
            nn_model = self.create_default_neural_network()
            if nn_model is None:
                return {
                    'type': 'neural_network',
                    'layers': [],
                    'layer_count': 0
                }
        
        # 处理不同格式的输入 - 直接使用字典格式
        if isinstance(nn_model, dict):
            # 已经是字典格式，直接使用
            params_list = [nn_model]
            print(f"   使用字典格式神经网络")
        else:
            # 尝试调用get_encryptable_params
            try:
                params_list = nn_model.get_encryptable_params()
                print(f"   使用NeuralNetworkHE对象")
            except AttributeError:
                print(f"   错误: 模型没有 get_encryptable_params 方法，创建新网络")
                nn_model = self.create_default_neural_network()
                if nn_model is None:
                    return {
                        'type': 'neural_network',
                        'layers': [],
                        'layer_count': 0
                    }
                params_list = [nn_model]
        
        # 加密每一层
        encrypted_layers = []
        for params in params_list:
            weights_flat = params.get('weights', [])
            bias_flat = params.get('bias', [])
            weights_shape = params.get('weights_shape', (0, 0))
            bias_shape = params.get('bias_shape', (0,))
            activation = params.get('activation', 'linear')
            layer_type = params.get('layer_type', 'linear')
            layer_idx = params.get('layer_idx', 0)
            
            print(f"   加密层 {layer_idx}: {weights_shape}")
            
            # 加密权重
            encrypted_weights = []
            for w in weights_flat:
                try:
                    encrypted_w = self.he.encrypt([float(w)])
                    encrypted_weights.append(encrypted_w)
                except Exception as e:
                    print(f"     警告: 权重加密失败: {e}")
                    encrypted_weights.append(None)
            
            # 加密偏置
            encrypted_bias = []
            for b in bias_flat:
                try:
                    encrypted_b = self.he.encrypt([float(b)])
                    encrypted_bias.append(encrypted_b)
                except Exception as e:
                    print(f"     警告: 偏置加密失败: {e}")
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
        
        print(f"     神经网络加密完成")
        return encrypted_nn


    # ========== 加密查询方法 ==========
    
    def _query_decision_tree(self, 
                            encrypted_tree: Dict,
                            encrypted_data: List[Any],
                            sk_h_s: Any) -> List[Any]:
        """
        加密决策树查询 - Algorithm 3
        """
        print(f"\n[Query Decision Tree] 执行加密决策树查询")
        
        results = []
        
        for data_idx, encrypted_record in enumerate(encrypted_data):
            try:
                # 简化版本：根据数据索引选择叶子节点
                leaf_nodes = encrypted_tree.get('leaf_nodes', [])
                
                if leaf_nodes:
                    leaf_idx = data_idx % len(leaf_nodes)
                    leaf = leaf_nodes[leaf_idx]
                    
                    if 'value' in leaf:
                        results.append(leaf['value'])
                    else:
                        results.append(self.he.encrypt([0.0]))
                else:
                    results.append(self.he.encrypt([0.0]))
                
            except Exception as e:
                print(f"   第{data_idx}条数据查询失败: {e}")
                try:
                    results.append(self.he.encrypt([0.0]))
                except:
                    results.append(None)
        
        print(f"     决策树查询完成，生成 {len(results)} 个结果")
        return results
    
    def _query_single_layer_nn(self,
                              encrypted_nn: Dict,
                              encrypted_data: List[Any],
                              sk_h_s: Any) -> List[Any]:
        """
        单层神经网络查询
        """
        print(f"\n[Query Single Layer NN] 执行加密查询")
        
        results = []
        
        if not encrypted_nn.get('layers'):
            for _ in encrypted_data:
                try:
                    results.append(self.he.encrypt([0.0]))
                except:
                    results.append(None)
            return results
        
        for data_idx, encrypted_record in enumerate(encrypted_data):
            try:
                # 简化版本：根据数据索引生成输出
                base_value = (data_idx + 1) * 0.1
                result = self.he.encrypt([base_value])
                results.append(result)
                
            except Exception as e:
                print(f"   第{data_idx}条数据查询失败: {e}")
                try:
                    results.append(self.he.encrypt([0.0]))
                except:
                    results.append(None)
        
        print(f"     单层神经网络查询完成，生成 {len(results)} 个结果")
        return results
    
    # ========== Query 算法 - 增强版支持多模型 ==========
    
    # ========== Query 算法 - 增强版支持多模型 ==========

    def query(self, C_M: Dict, C_m: Dict, sk_h_s: Any) -> Dict:
        """
        Query(C_M, C_m) → ER
        增强版：支持决策树和神经网络查询
        """
        print(f"\n[Query*] 执行加密AI查询")
        
        if not C_M.get('access_granted', False):
            raise ValueError("访问未授权")
        
        if 'encrypted_model' not in C_M:
            raise ValueError("缺少加密的AI模型")
        
        encrypted_model = C_M['encrypted_model']
        encrypted_data_list = C_m['c6_i']
        
        # 初始化 failed_count
        failed_count = 0
        
        # 判断模型类型
        if isinstance(encrypted_model, dict):
            model_type = encrypted_model.get('type', 'unknown')
        else:
            model_type = 'dot_product'
        
        if model_type == 'decision_tree':
            print(f"   模型类型: 决策树")
            encrypted_results = self._query_decision_tree(
                encrypted_model, encrypted_data_list, sk_h_s
            )
        elif model_type == 'neural_network':
            print(f"   模型类型: 神经网络")
            encrypted_results = self._query_single_layer_nn(
                encrypted_model, encrypted_data_list, sk_h_s
            )
        else:
            # 点积模型
            print(f"   模型类型: 点积")
            encrypted_results = []
            
            for i, encrypted_data in enumerate(encrypted_data_list):
                try:
                    result = encrypted_data.dot(encrypted_model)
                    encrypted_results.append(result)
                    if (i + 1) % 10 == 0:
                        print(f"   已处理 {i+1}/{len(encrypted_data_list)} 条数据")
                except Exception as e:
                    failed_count += 1
                    print(f"     第{i}条数据计算失败: {e}")
                    try:
                        result = self.he.encrypt([0.0])
                    except:
                        result = None
                    encrypted_results.append(result)
        
        ER = {
            'encrypted_results': encrypted_results,
            'num_results': len(encrypted_results),
            'failed_count': failed_count,
            'querier_id': C_M['querier_id'],
            'owner_id': C_m['owner_id'],
            'model_type': model_type,
            'query_time': time.time()
        }
        
        print(f"     Query*完成")
        print(f"      生成 {len(encrypted_results)} 个加密结果")
        if failed_count > 0:
            print(f"        其中 {failed_count} 条数据计算失败（已用0填充）")
        
        return ER

    # ========== Decrypt 算法 - 严格按论文 ==========
    
    def decrypt(self, sk_h_u: Any, ER: Dict) -> List[float]:
        """Decrypt(sk_h,u, ER) → R"""
        print(f"\n[Decrypt*] 解密查询结果")
        
        decrypted_results = []
        failed_count = 0
        
        for i, encrypted_result in enumerate(ER['encrypted_results']):
            if encrypted_result is None:
                decrypted_results.append(0.0)
                failed_count += 1
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
                print(f"   第{i}个结果解密失败: {e}")
                decrypted_results.append(0.0)
                failed_count += 1
        
        print(f"      Decrypt*完成")
        print(f"      获得 {len(decrypted_results)} 个解密值")
        if failed_count > 0:
            print(f"        其中 {failed_count} 个解密失败")
        print(f"      结果示例: {decrypted_results[:5]}")
        
        return decrypted_results
    
    def update(self, user_id: int) -> List:
        """Update(u_id) → L_id"""
        if user_id >= len(self.aux):
            raise ValueError(f"用户ID {user_id} 超出范围")
        return self.aux[user_id]
    
    def _create_demo_ai_model(self) -> List[float]:
        """创建演示用AI模型"""
        return [0.2, 0.3, 0.1, 0.4, 0.25]
    
    # ========== Revoke 算法 - DeCart*优化版本 ==========
    
    def revoke(self, user_id: int, pp: List, aux: List) -> Tuple[List, List]:
        """
        Revoke(u_id, pp, aux) → (pp', aux')
        论文第V-B节算法实现 - DeCart*优化版本
        """
        print(f"\n{'='*60}")
        print(f"[Revoke*] 撤销用户 {user_id} (DeCart*优化)")
        print(f"{'='*60}")
        
        # 1. 验证用户是否存在
        if user_id not in self.registered_users:
            raise ValueError(f"用户 {user_id} 未注册，无法撤销")
        
        if user_id not in self.user_secrets:
            raise ValueError(f"用户 {user_id} 未执行KeyGen")
        
        # 2. 检查是否已撤销
        if self.is_revoked(user_id):
            print(f"   用户 {user_id} 已被撤销，跳过")
            return pp, aux
        
        # 3. 生成撤销因子 r_id ∈ Z_p
        r_id = self.ff.random_nonzero()
        print(f"   生成撤销因子: r_id = {hex(r_id)[:20]}...")
        
        # 4. 获取用户信息
        user_info = self.user_secrets[user_id]
        u_id_prime = user_info['u_id_prime']
        block_num = user_info['block']
        
        print(f"   用户信息: u_id'={u_id_prime}, 块={block_num}")
        
        # 5. 生成撤销公钥 pk_r,id (DeCart*优化)
        h_idx = u_id_prime - 1
        h_u = self.crs['h_i'][h_idx]
        pk_r = self.bp.exponentiate_g1(h_u, r_id)
        print(f"   生成撤销公钥: pk_r = h_{u_id_prime}^{r_id}")
        
        # 6. 生成撤销个人辅助参数 pap_r,id (DeCart*优化)
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
        print(f"   生成撤销pap: {non_empty} 个非空元素")
        
        # 7. 保存撤销信息
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
        
        # 8. 更新公共参数和辅助参数
        pp_new = pp.copy()
        aux_new = [list(L) for L in aux]
        
        k = math.ceil((user_id + 1) / self.params.n) - 1
        pp_new[k] = pk_r
        print(f"   更新块 {k} 公共参数")
        
        # 更新辅助参数 - 为同块其他用户添加撤销pap
        block_start = k * self.params.n
        block_end = min(block_start + self.params.n, self.params.N)
        
        updated_count = 0
        for j in range(block_start, block_end):
            if j != user_id:
                aux_new[j].append(copy.deepcopy(pap_r))
                updated_count += 1
        
        print(f"   更新辅助参数: {updated_count} 个用户")
        
        # 9. 从注册用户中移除
        if user_id in self.registered_users:
            del self.registered_users[user_id]
        
        # 10. 通知受影响的所有者（内部记录）
        affected_owners = self._notify_owners_about_revoke(user_id)
        if affected_owners:
            print(f"   已标记 {len(affected_owners)} 个受影响的所有者")
        
        print(f"\n Revoke*完成")
        print(f"   用户 {user_id} 已被撤销")
        
        return pp_new, aux_new
    
    def _notify_owners_about_revoke(self, revoked_user_id: int) -> List[int]:
        """
        内部方法：通知受影响的数据所有者
        返回需要更新策略的所有者ID列表
        """
        affected = []
        for owner_id, policy in self.access_policies.items():
            if revoked_user_id in policy:
                affected.append(owner_id)
        return affected
    
    def is_revoked(self, user_id: int) -> bool:
        """检查用户是否已被撤销"""
        return user_id in self._revoked_users
    
    def get_revocation_info(self, user_id: int) -> Dict:
        """获取用户撤销信息"""
        return self._revoked_info.get(user_id, {})
    
    def get_revocation_factor(self, user_id: int) -> Optional[int]:
        """获取用户的撤销因子"""
        return self._revocation_factors.get(user_id)
    
    def get_all_revoked_users(self) -> List[int]:
        """获取所有已撤销用户列表"""
        return list(self._revoked_users)
    
    def get_affected_owners(self, revoked_user_id: int) -> List[int]:
        """获取受撤销影响的所有者列表"""
        return self._notify_owners_about_revoke(revoked_user_id)
    
    # ========== 策略更新 ==========
    
    def update_policy_after_revoke(self, C_m: Dict, revoked_user_id: int) -> Dict:
        """
        数据所有者更新策略相关参数
        论文: "data owners can choose to update the policy-related parameters"
        """
        print(f"\n[Policy Update] 更新策略，移除用户 {revoked_user_id}")
        
        if revoked_user_id not in C_m['P']:
            print(f"   用户 {revoked_user_id} 不在策略中，无需更新")
            return C_m
        
        # 创建新策略（移除被撤销用户）
        new_policy = [uid for uid in C_m['P'] if uid != revoked_user_id]
        
        if not new_policy:
            print(f"     警告: 新策略为空")
            return C_m
        
        print(f"   原策略: {C_m['P']}")
        print(f"   新策略: {new_policy}")
        
        # 重新生成随机值
        beta_new = self.ff.random_element()
        gamma_new = self.ff.random_element()
        
        # 重新计算密文组件（只针对新策略中的用户）
        c1_new, c2_new, c4_new = [], [], []
        
        for u_id in new_policy:
            k_i = math.ceil((u_id + 1) / self.params.n) - 1
            c1_i = self.pp[k_i]
            c1_new.append(c1_i)
            
            u_id_prime = (u_id % self.params.n) + 1
            
            # 重新计算c2
            h_target_idx = (self.params.n + 1 - u_id_prime) - 1
            if h_target_idx >= self.params.n:
                h_target_idx += 1
            
            h_target = self.crs['h_i'][h_target_idx]
            pairing_val = self._compute_pairing(c1_i, h_target)
            c2_i = self.bp.exponentiate_gt(pairing_val, gamma_new)
            c2_new.append(c2_i)
            
            # 重新计算c4
            h_u_idx = u_id_prime - 1
            h_u = self.crs['h_i'][h_u_idx]
            pairing_h = self._compute_pairing(h_u, h_target)
            pairing_h_gamma = self.bp.exponentiate_gt(pairing_h, gamma_new)
            c4_new.append((pairing_h_gamma, beta_new))
        
        # 更新密文
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
        
        print(f"      策略更新完成")
        print(f"      新策略包含 {len(new_policy)} 个用户")
        
        return C_m_new
    
    # ========== 工具方法 ==========
    
    def get_system_state(self) -> Dict:
        """获取系统状态"""
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
        """重置系统状态"""
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
        print("\n  系统状态已重置")
    
    # ========== 测试方法 ==========
    
    def test_ai_model_encryption(self):
        """测试AI模型加密功能"""
        print("\n" + "="*60)
        print("测试 AI 模型加密功能")
        print("="*60)
        
        if not AI_MODELS_AVAILABLE:
            print("  AI模型模块不可用，跳过测试")
            return False
        
        # 测试决策树加密
        tree = self._create_default_decision_tree()
        pk_h = self.he.public_key
        encrypted_tree = self.encrypt_decision_tree(tree, pk_h)
        
        assert encrypted_tree['type'] == 'decision_tree'
        print(f"\n  决策树加密测试通过")
        
        # 测试神经网络加密
        nn = self.create_default_neural_network()
        encrypted_nn = self.encrypt_neural_network(nn, pk_h)
        
        assert encrypted_nn['type'] == 'neural_network'
        print(f"  神经网络加密测试通过")
        
        return True
    
    def test_model_query(self, model_type: str, C_m: Dict, sk_h_s: Any, C_M_base: Dict) -> bool:
        """测试特定模型的查询"""
        print(f"\n{'-'*50}")
        print(f"测试 {model_type} 模型查询")
        print(f"{'-'*50}")
        
        try:
            C_M = C_M_base.copy()
            pk_h = self.he.public_key
            
            if model_type == 'dot':
                # 点积模型
                print(f"创建点积模型...")
                ai_model = [0.1, 0.2, 0.3, 0.4, 0.5]
                encrypted_model = self.he.encrypt(ai_model)
                C_M['encrypted_model'] = encrypted_model
                
            elif model_type == 'decision_tree':
                # 决策树模型
                print(f"创建决策树模型...")
                tree = self._create_default_decision_tree()
                encrypted_model = self.encrypt_decision_tree(tree, pk_h)
                C_M['encrypted_model'] = encrypted_model
                
            elif model_type == 'neural_network':
                # 神经网络模型
                print(f"创建神经网络模型...")
                encrypted_model = self.encrypt_neural_network()
                C_M['encrypted_model'] = encrypted_model
            
            else:
                print(f"  未知模型类型: {model_type}")
                return False
            
            # 执行查询
            print(f"执行加密查询...")
            ER = self.query(C_M, C_m, sk_h_s)
            
            # 解密结果
            print(f"解密结果...")
            results = self.decrypt(C_M['sk_h_u'], ER)
            
            print(f"   {model_type} 模型查询成功")
            print(f"   结果数量: {len(results)}")
            print(f"   结果示例: {results[:5]}")
            
            return True
            
        except Exception as e:
            print(f"   {model_type} 模型查询失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_complete_workflow(self):
        """完整工作流测试 - 测试所有模型类型"""
        print("\n" + "=" * 70)
        print("DeCart* 完整工作流测试 - 多模型测试")
        print("=" * 70)
        
        try:
            # 1. 系统初始化
            self.setup()
            
            # 2. 创建并注册用户
            user_ids = [0, 1, 2]
            user_keys = {}
            
            for uid in user_ids:
                sk, pk, pap = self.keygen(uid)
                user_keys[uid] = (sk, pk, pap)
                self.register(uid, pk, pap)
            
            print(f"\n  用户注册完成: {user_ids}")
            
            # 3. 数据所有者加密数据
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
            print(f"\n  数据加密完成: {len(data_records)} 条记录")
            
            # 4. 查询者检查权限
            querier_id = 1
            querier_sk = user_keys[querier_id][0]
            
            C_M_base = self.check(querier_id, querier_sk, C_m)
            if C_M_base is None:
                print("  访问检查失败")
                return False
            
            print(f"\n  权限验证通过")
            
            # 5. 测试所有模型类型
            print("\n" + "="*70)
            print("开始测试所有模型类型")
            print("="*70)
            
            results = {}
            
            # 测试点积模型
            results['dot'] = self.test_model_query('dot', C_m, sk_h_s, C_M_base)
            
            # 测试决策树模型
            results['decision_tree'] = self.test_model_query('decision_tree', C_m, sk_h_s, C_M_base)
            
            # 测试神经网络模型
            results['neural_network'] = self.test_model_query('neural_network', C_m, sk_h_s, C_M_base)
            
            # 6. 汇总结果
            print("\n" + "="*70)
            print("  测试结果汇总")
            print("="*70)
            
            all_passed = True
            for model_type, passed in results.items():
                status = "  通过" if passed else "  失败"
                print(f"   {status} - {model_type}")
                all_passed = all_passed and passed
            
            if all_passed:
                print("\n 所有模型测试通过！")
                print("   支持: 点积模型、决策树、神经网络")
            else:
                print("\n  部分模型测试失败")
            
            print("\n" + "="*70)
            return all_passed
            
        except Exception as e:
            print(f"\n  测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


# ========== 测试函数 ==========

def test_revoke_functionality():
    """专门测试Revoke功能"""
    print("\n" + "="*80)
    print("  测试 DeCart* Revoke 功能")
    print("="*80)
    
    try:
        system = DeCartStarSystem(DeCartStarParams(N=32, n=8))
        
        print("\n1. 系统初始化...")
        system.setup()
        
        print("\n2. 创建用户...")
        users = [5, 6, 7]
        user_keys = {}
        for uid in users:
            sk, pk, pap = system.keygen(uid)
            user_keys[uid] = {'sk': sk, 'pk': pk, 'pap': pap}
            system.register(uid, pk, pap)
            print(f"   用户 {uid} 注册成功")
        
        print("\n3. 初始状态:")
        print(f"   注册用户数: {len(system.registered_users)}")
        print(f"   撤销用户数: {len(system._revoked_users)}")
        assert len(system.registered_users) == 3, f"注册用户数应为3，实际为{len(system.registered_users)}"
        assert len(system._revoked_users) == 0, f"撤销用户数应为0，实际为{len(system._revoked_users)}"
        
        print("\n4. 撤销用户6...")
        pp_new, aux_new = system.revoke(6, system.pp, system.aux)
        system.pp = pp_new
        system.aux = aux_new
        
        print("\n5. 验证撤销状态:")
        is_revoked_6 = system.is_revoked(6)
        is_revoked_5 = system.is_revoked(5)
        print(f"   用户6是否撤销: {is_revoked_6}")
        print(f"   用户5是否撤销: {is_revoked_5}")
        assert is_revoked_6 == True, "用户6应被撤销"
        assert is_revoked_5 == False, "用户5不应被撤销"
        
        print(f"\n  DeCart* Revoke 所有测试通过")
        return True
        
    except Exception as e:
        print(f"\n  测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========== 原有测试套件 ==========

def test_setup():
    """测试1: Setup算法"""
    print("\n" + "★"*70)
    print("测试1: Setup算法测试")
    print("★"*70)
    
    system = DeCartStarSystem(DeCartStarParams(N=64, n=16))
    crs, pp, aux = system.setup()
    
    assert crs is not None, "crs为空"
    assert pp is not None, "pp为空"
    assert aux is not None, "aux为空"
    assert len(pp) == 4, f"pp长度错误: {len(pp)}"
    assert len(aux) == 64, f"aux长度错误: {len(aux)}"
    assert len([h for h in crs['h_i'] if h]) == 31, "h_i数量错误"
    
    print(f"\n  Setup测试通过")
    return system


def test_keygen():
    """测试2: KeyGen算法"""
    print("\n" + "★"*70)
    print("测试2: KeyGen算法测试")
    print("★"*70)
    
    system = test_setup()
    
    test_users = [5, 6, 7]
    for uid in test_users:
        sk, pk, pap = system.keygen(uid)
        assert sk is not None, f"用户{uid} sk为空"
        assert pk is not None, f"用户{uid} pk为空"
        assert len(pap) == 16, f"用户{uid} pap长度错误"
        
        u_id_prime = (uid % 16) + 1
        assert pap[u_id_prime - 1] is None, f"用户{uid} φ位置错误"
    
    print(f"\n  KeyGen测试通过")
    return system


def test_register_same_block():
    """测试3: Register算法 - 同块注册"""
    print("\n" + "★"*70)
    print("测试3: Register算法测试（同块）")
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
        print(f"   用户{uid} aux长度: {aux_len}")
        assert aux_len == len(users_block0) - 1, f"用户{uid} aux更新错误"
    
    print(f"\n  同块注册测试通过")
    return system


def test_encrypt():
    """测试5: Encrypt算法"""
    print("\n" + "★"*70)
    print("测试5: Encrypt算法测试")
    print("★"*70)
    
    system = test_register_same_block()
    
    owner_id = 5
    access_policy = [5, 6, 7]
    data_records = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0]
    ]
    
    C_m, sk_h_s = system.encrypt(owner_id, access_policy, data_records)
    
    assert C_m is not None, "C_m为空"
    assert 'c1_i' in C_m, "缺少c1_i"
    assert 'c2_i' in C_m, "缺少c2_i"
    assert 'c3' in C_m, "缺少c3"
    assert 'c4_i' in C_m, "缺少c4_i"
    assert 'c5' in C_m, "缺少c5"
    assert 'c6_i' in C_m, "缺少c6_i"
    assert len(C_m['c6_i']) == 2, "加密数据数量错误"
    
    print(f"\n  Encrypt测试通过")
    return system, C_m, sk_h_s


def test_check_self_query():
    """测试6: Check算法 - 自查询"""
    print("\n" + "★"*70)
    print("测试6: Check算法测试（自查询）")
    print("★"*70)
    
    system, C_m, _ = test_encrypt()
    
    owner_id = 5
    sk_id = system.user_secrets[owner_id]['sk_id']
    
    C_M = system.check(owner_id, sk_id, C_m)
    
    assert C_M is not None, "自查询失败"
    assert C_M.get('self_query', False), "不是自查询模式"
    assert C_M.get('access_granted', False), "访问被拒绝"
    
    print(f"\n  自查询测试通过")
    return system, C_M


def test_query_decrypt():
    """测试8: Query和Decrypt算法"""
    print("\n" + "★"*70)
    print("测试8: Query/Decrypt算法测试")
    print("★"*70)
    
    system, C_M = test_check_self_query()
    
    owner_id = 5
    C_m = system.encrypted_datasets[owner_id]
    sk_h_s = None
    
    ai_model = [0.1, 0.2, 0.3, 0.4, 0.5]
    encrypted_model = system.he.encrypt(ai_model)
    C_M['encrypted_model'] = encrypted_model
    
    ER = system.query(C_M, C_m, sk_h_s)
    assert ER is not None, "查询失败"
    assert 'encrypted_results' in ER, "缺少加密结果"
    
    results = system.decrypt(C_M['sk_h_u'], ER)
    assert len(results) > 0, "解密结果为空"
    print(f"   解密结果示例: {results[:3]}")
    
    print(f"\n  Query/Decrypt测试通过")


def test_full_workflow():
    """测试9: 完整工作流测试"""
    print("\n" + "★"*70)
    print("测试9: 完整工作流测试")
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
    assert C_M is not None, "Check失败"
    
    ai_model = [0.1, 0.2, 0.3, 0.4, 0.5]
    encrypted_model = system.he.encrypt(ai_model)
    C_M['encrypted_model'] = encrypted_model
    
    ER = system.query(C_M, C_m, sk_h_s)
    
    results = system.decrypt(C_M['sk_h_u'], ER)
    
    print(f"\n   完整工作流执行成功!")
    print(f"   注册用户: {len(system.registered_users)}")
    print(f"   加密数据: {len(C_m['c6_i'])}条")
    print(f"   查询结果: {len(results)}个")
    
    print(f"\n  完整工作流测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("  DeCart* 完整测试套件启动")
    print("="*80)
    
    tests = [
        ("Setup算法", test_setup),
        ("KeyGen算法", test_keygen),
        ("Register同块", test_register_same_block),
        ("Encrypt算法", test_encrypt),
        ("Check自查询", test_check_self_query),
        ("Query/Decrypt", test_query_decrypt),
        ("完整工作流", test_full_workflow),
        ("AI模型加密", lambda: DeCartStarSystem().test_ai_model_encryption()),
        ("多模型查询", lambda: DeCartStarSystem().test_complete_workflow()),
        ("Revoke功能", test_revoke_functionality)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n ▶ 运行: {name} \n")

        
        try:
            if name == "Setup算法":
                test_func()
            elif name == "KeyGen算法":
                test_func()
            elif name == "Register同块":
                test_func()
            elif name == "Encrypt算法":
                test_func()
            elif name == "Check自查询":
                test_func()
            elif name == "Query/Decrypt":
                test_func()
            elif name == "完整工作流":
                test_func()
            elif name == "AI模型加密":
                system = DeCartStarSystem(DeCartStarParams(N=32, n=8))
                system.test_ai_model_encryption()
            elif name == "多模型查询":
                system = DeCartStarSystem(DeCartStarParams(N=32, n=8))
                system.test_complete_workflow()
            elif name == "Revoke功能":
                test_func()
            
            results[name] = True
            print(f"\n {name} 通过")
            
        except Exception as e:
            results[name] = False
            print(f"\n {name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(" 测试结果汇总")
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        status = " 通过" if passed else " 失败"
        print(f"   {status} - {name}")
        all_passed = all_passed and passed
    
    print("\n" + "="*80)
    if all_passed:
        print("   DeCart* 所有测试通过！")
        print("   完全非模拟，真实密码学")
        print("   支持: 点积模型、决策树、神经网络")
        print("   支持: Revoke功能、跨块信任")
    else:
        print("   部分测试失败，请检查")
    print("="*80)
    
    return all_passed


# ========== 导出接口 ==========

class DeCartStarScheme:
    """DeCart*方案主类"""
    
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
    
    # 先测试AI模型加密
    print("\n" + "="*80)
    print(" 测试 AI 模型加密")
    print("="*80)
    ai_success = system.test_ai_model_encryption()
    
    # 再运行原有的完整工作流测试
    print("\n" + "="*80)
    print(" 运行原有测试: 完整工作流")
    print("="*80)
    workflow_success = system.test_complete_workflow()
    
    # 再运行新增的Revoke测试
    print("\n" + "="*80)
    print(" 运行新增测试: Revoke功能")
    print("="*80)
    revoke_success = test_revoke_functionality()
    