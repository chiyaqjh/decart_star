# decart/entities/key_curator.py
"""
Key Curator 实体 - 支持DeCart和DeCart*双方案 + Revoke功能
论文第I.A节系统模型实现
"""

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

# 导入两个方案
from schemes.decart import DeCartSystem, DeCartParams
from schemes.decart_star import DeCartStarSystem, DeCartStarParams
from core.bilinear_pairing import BilinearPairing
from core.finite_field import FiniteField


class KeyCurator:
    """
    密钥管理者 (Key Curator)
    
    支持双方案:
    1. DeCart  - 原始方案，O(n²)复杂度
    2. DeCart* - 优化方案，O(n)复杂度，20倍性能提升
    
    论文职责（与方案无关）:
    1. 系统初始化 - Setup(λ) → (crs, pp, aux)
    2. 用户注册 - Register(u_id, pk_id, pap_id) → (pp', aux')
    3. 用户撤销 - Revoke(u_id, pp, aux) → (pp', aux')
    4. 维护公共参数 C_{(k)}
    5. 维护辅助参数 L_j
    """
    
    def __init__(self, 
                 scheme: str = "decart_star",  # 默认使用优化方案
                 params: Optional[Union[DeCartParams, DeCartStarParams]] = None):
        """
        初始化密钥管理者
        
        参数:
            scheme: "decart" 或 "decart_star"
            params: 对应方案的参数对象
        """
        self.scheme = scheme.lower()
        
        # ===== 根据方案选择对应的系统内核 =====
        if self.scheme == "decart":
            from schemes.decart import DeCartSystem, DeCartParams
            self.params = params or DeCartParams(N=1024, n=32)
            self.system = DeCartSystem(self.params)
            self.scheme_name = "DeCart (O(n²))"
            
        elif self.scheme == "decart_star":
            from schemes.decart_star import DeCartStarSystem, DeCartStarParams
            self.params = params or DeCartStarParams(N=1024, n=32)
            self.system = DeCartStarSystem(self.params)
            self.scheme_name = "DeCart* (O(n)优化)"
            
        else:
            raise ValueError(f"未知方案: {scheme}，请使用 'decart' 或 'decart_star'")
        
        # ===== 系统状态 - 所有方案通用 =====
        self.crs = None
        self.pp = None
        self.aux = None
        
        # ===== 注册表 =====
        self.user_public_keys = {}      # user_id -> pk_id
        self.user_blocks = {}           # user_id -> block
        self.user_id_prime = {}         # user_id -> u_id'
        self.user_pap = {}             # user_id -> pap_id
        self.registered_users = set()   # 已注册用户集合
        self.registration_time = {}    # user_id -> timestamp
        
        # ===== 跨块信任管理（实体层增强）=====
        self._trust_map = {}            # trustee_id -> Set[truster_id]
        self._trust_time = {}           # 信任建立时间
        
        # ===== 撤销相关状态 =====
        self._revoked_users = set()           # 已撤销用户集合
        self._revoked_info = {}                # 撤销信息 {user_id: info}
        self._revocation_time = {}             # 撤销时间
        
        # ===== 统计信息 =====
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
        
        print(f"\n  Key Curator 实体初始化")
        print(f"   方案: {self.scheme_name}")
        print(f"   参数: N={self.params.N}, n={self.params.n}, B={self.params.B}")
        print(f"   支持 Revoke 功能")
    
    # ========== 论文算法：Setup ==========
    
    def setup(self) -> Tuple[Dict, List, List]:
        """
        系统初始化 - 论文Setup(λ)算法
        方案无关，委托给对应系统内核
        """
        print("\n" + "="*60)
        print(f"[Key Curator] 系统初始化 - {self.scheme_name}")
        print("="*60)
        
        self.crs, self.pp, self.aux = self.system.setup()
        self.stats['setup_complete'] = True
        self.stats['setup_time'] = time.time()
        
        # 方案特定的统计
        if self.scheme == "decart":
            h_count = len(self.crs['h_i'])
            H_count = len(self.crs['H_ij'])
            print(f"   crs: h_i={h_count}, H_ij={H_count}, 总计={h_count + H_count}")
        else:  # decart_star
            h_count = len([h for h in self.crs['h_i'] if h])
            print(f"   crs: h_i={h_count} (O(n)优化)")
        
        print(f"   pp: {len(self.pp)}个块参数")
        print(f"   aux: {len(self.aux)}个用户槽位")
        
        return self.crs, self.pp, self.aux
    
    # ========== 密钥生成 ==========
    
    def generate_user_key(self, user_id: int) -> Tuple[int, Any, List[Optional[Any]]]:
        """
        为用户生成密钥对 - 调用对应方案的KeyGen算法
        
        返回:
            sk_id: 用户私钥（保密，仅返回给用户）
            pk_id: 用户公钥（公开，注册时提交）
            pap_id: 个人辅助参数（公开，注册时提交）
        """
        if self.crs is None:
            raise ValueError("请先执行setup()")
        
        if not (0 <= user_id < self.params.N):
            raise ValueError(f"用户ID必须在[0, {self.params.N-1}]")
        
        print(f"\n[KeyGen] 用户 {user_id} 生成密钥 - {self.scheme_name}")
        return self.system.keygen(user_id)
    
    # ========== 论文算法：Register + 实体层跨块增强 + 撤销检查 ==========
    
    def register(self, user_id: int, pk_id: Any, pap_id: List[Optional[Any]]) -> bool:
        """
        用户注册 - 论文Register算法 + 实体层跨块增强 + 撤销检查
        
        算法流程:
        1. 检查用户是否已被撤销（不能重新注册）
        2. 调用对应方案的Register算法 - 只更新同块用户的aux
        3. 实体层增强 - 根据信任关系更新跨块信任者的aux
        4. 更新注册表
        
        此逻辑对DeCart和DeCart*方案通用
        """
        print(f"\n[Key Curator] 处理用户 {user_id} 注册请求 - {self.scheme_name}")
        
        try:
            # ===== 检查：用户是否已被撤销 =====
            if self.is_revoked(user_id):
                print(f"    用户 {user_id} 已被撤销，无法重新注册")
                return False
            
            # 验证用户是否已注册
            if user_id in self.registered_users:
                print(f"    用户 {user_id} 已注册")
                return False
            
            # 验证用户是否执行了KeyGen
            if user_id not in self.system.user_secrets:
                print(f"    用户 {user_id} 未执行KeyGen")
                return False
            
            # ===== 步骤1: 调用论文Register算法（只更新同块）=====
            print(f"   [算法层] 执行{self.scheme_name} Register算法...")
            self.pp, self.aux = self.system.register(user_id, pk_id, pap_id)
            
            # ===== 步骤2: 实体层增强 - 跨块信任更新 =====
            trusted_by = self.get_trusted_by(user_id)
            if trusted_by:
                print(f"   [实体层] 发现 {len(trusted_by)} 个用户信任用户 {user_id}")
                
                cross_block_count = 0
                for truster_id in trusted_by:
                    if truster_id != user_id and truster_id < len(self.aux):
                        self.aux[truster_id].append(copy.deepcopy(pap_id))
                        cross_block_count += 1
                        self.stats['cross_block_updates'] += 1
                
                print(f"   [实体层] 完成 {cross_block_count} 个跨块aux更新")
            
            # ===== 步骤3: 获取用户信息并更新注册表 =====
            user_info = self.system.user_secrets.get(user_id)
            if not user_info:
                print(f"    用户 {user_id} 密钥信息不存在")
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
            
            print(f"     用户 {user_id} 注册成功")
            print(f"      所属块: {block_num}, u_id'={u_id_prime}")
            print(f"      当前注册用户: {self.stats['total_users']}")
            
            return True
            
        except Exception as e:
            print(f"    注册失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========== 论文算法：Revoke ==========
    
    def revoke_user(self, user_id: int) -> bool:
        """
        撤销用户 - 论文Revoke算法
        
        参数:
            user_id: 要撤销的用户ID
        
        返回:
            撤销是否成功
        """
        print(f"\n[Key Curator] 撤销用户 {user_id} - {self.scheme_name}")
        
        try:
            # 1. 验证用户是否存在
            if user_id not in self.registered_users:
                print(f"    用户 {user_id} 未注册，无法撤销")
                return False
            
            # 2. 检查是否已撤销
            if self.is_revoked(user_id):
                print(f"    用户 {user_id} 已被撤销")
                return True
            
            # 3. 调用对应方案的Revoke算法
            self.pp, self.aux = self.system.revoke(user_id, self.pp, self.aux)
            
            # 4. 更新本地撤销状态
            self._revoked_users.add(user_id)
            self._revocation_time[user_id] = time.time()
            
            # 5. 从注册表中移除
            if user_id in self.registered_users:
                self.registered_users.remove(user_id)
                self.stats['total_users'] -= 1
            
            self.stats['total_revoked'] += 1
            self.stats['revoke_operations'] += 1
            
            # 6. 获取撤销信息（用于调试）
            revoke_info = self.system.get_revocation_info(user_id)
            self._revoked_info[user_id] = revoke_info
            
            # 7. 获取受影响的所有者
            affected_owners = self.system.get_affected_owners(user_id)
            if affected_owners:
                print(f"    通知 {len(affected_owners)} 个所有者更新策略")
            
            print(f"\n  用户 {user_id} 撤销成功")
            print(f"   当前注册用户: {self.stats['total_users']}")
            print(f"   累计撤销用户: {self.stats['total_revoked']}")
            
            return True
            
        except Exception as e:
            print(f"    撤销失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_revoked(self, user_id: int) -> bool:
        """检查用户是否已被撤销"""
        # 先检查本地缓存
        if user_id in self._revoked_users:
            return True
        # 再检查系统内核
        if hasattr(self.system, 'is_revoked'):
            return self.system.is_revoked(user_id)
        return False
    
    def get_revoked_users(self) -> List[int]:
        """获取所有已撤销用户列表"""
        revoked = list(self._revoked_users)
        if hasattr(self.system, 'get_all_revoked_users'):
            system_revoked = self.system.get_all_revoked_users()
            # 合并去重
            revoked = list(set(revoked + system_revoked))
        return revoked
    
    def get_revocation_info(self, user_id: int) -> Dict:
        """获取用户撤销信息"""
        if user_id in self._revoked_info:
            return self._revoked_info[user_id]
        if hasattr(self.system, 'get_revocation_info'):
            return self.system.get_revocation_info(user_id)
        return {}
    
    def get_affected_owners(self, revoked_user_id: int) -> List[int]:
        """获取受撤销影响的所有者列表"""
        if hasattr(self.system, 'get_affected_owners'):
            return self.system.get_affected_owners(revoked_user_id)
        return []
    
    def update_policy_after_revoke(self, owner_id: int, C_m: Dict, revoked_user_id: int) -> Optional[Dict]:
        """
        为数据所有者更新策略
        
        参数:
            owner_id: 数据所有者ID
            C_m: 原加密数据集
            revoked_user_id: 被撤销的用户ID
        
        返回:
            更新后的加密数据集，如果不需要更新则返回None
        """
        if hasattr(self.system, 'update_policy_after_revoke'):
            return self.system.update_policy_after_revoke(C_m, revoked_user_id)
        return None
    
    # ========== 实体层增强：跨块信任管理 ==========
    
    def add_trust(self, truster_id: int, trustee_id: int) -> bool:
        """建立跨块信任关系（方案无关）"""
        if truster_id == trustee_id:
            print(f"    不能信任自己")
            return False
        
        if trustee_id not in self._trust_map:
            self._trust_map[trustee_id] = set()
        
        if truster_id not in self._trust_map[trustee_id]:
            self._trust_map[trustee_id].add(truster_id)
            self._trust_time[f"{truster_id}->{trustee_id}"] = time.time()
            self.stats['trust_relations'] += 1
            
            print(f"   [信任] 用户 {trustee_id} ← 用户 {truster_id}")
            
            # 如果被信任者已注册，立即更新
            if trustee_id in self.registered_users and trustee_id in self.user_pap:
                pap_id = self.user_pap[trustee_id]
                if truster_id < len(self.aux):
                    self.aux[truster_id].append(copy.deepcopy(pap_id))
                    self.stats['cross_block_updates'] += 1
            
            return True
        return False
    
    def get_trusted_by(self, user_id: int) -> Set[int]:
        """获取信任此用户的所有用户"""
        return self._trust_map.get(user_id, set())
    
    # ========== 查询接口 ==========
    
    def get_user_aux(self, user_id: int) -> List:
        """获取用户的辅助参数 L_id"""
        if not self.aux or user_id >= len(self.aux):
            return []
        return self.aux[user_id]
    
    def get_block_public_key(self, block_id: int) -> Any:
        """获取块公钥 C_{(k)}"""
        if not self.pp or block_id >= len(self.pp):
            raise ValueError(f"块 {block_id} 不存在")
        return self.pp[block_id]
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
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
        """
        切换密码学方案（重置系统）
        用于对比测试
        """
        if scheme.lower() == self.scheme:
            print(f"   已经是 {self.scheme_name}")
            return True
        
        print(f"\n 切换方案: {self.scheme_name} → ", end="")
        self.__init__(scheme, self.params)
        print(f"{self.scheme_name}")
        return True


# ========== 测试代码 ==========

def test_key_curator_with_both_schemes():
    """测试Key Curator对两种方案的支持"""
    
    print("\n" + "="*80)
    print(" 测试 Key Curator 双方案支持")
    print("="*80)
    
    # 1. 测试DeCart方案
    print("\n 测试 DeCart 方案...")
    curator1 = KeyCurator(scheme="decart", params=DeCartParams(N=64, n=16))
    curator1.setup()
    
    # 2. 测试DeCart*方案
    print("\n 测试 DeCart* 方案...")
    curator2 = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator2.setup()
    
    # 3. 对比测试
    print("\n" + "="*80)
    print(" 方案对比")
    print("="*80)
    
    info1 = curator1.get_system_info()
    info2 = curator2.get_system_info()
    
    print(f"\n   DeCart  : {info1['scheme']}")
    print(f"   DeCart* : {info2['scheme']}")
    print(f"\n   两个方案都已成功初始化 ✓")
    
    return curator1, curator2

# decart/entities/key_curator.py (只修改测试函数部分)

# decart/entities/key_curator.py (修改整个测试函数)

def test_revoke_functionality():
    """测试撤销功能"""
    
    print("\n" + "="*80)
    print(" 测试 Key Curator Revoke 功能")
    print("="*80)
    
    # 使用DeCart*方案测试
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 1. 创建并注册用户
    print("\n1. 创建用户...")
    users = [5, 6, 7, 8]  # 增加一个用户作为数据所有者
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)
    
    assert len(curator.registered_users) == 4, "注册用户数应为4"
    assert len(curator.get_revoked_users()) == 0, "撤销用户数应为0"
    
    # 2. 建立信任关系
    print("\n2. 建立信任关系...")
    curator.add_trust(6, 5)  # 6信任5
    curator.add_trust(7, 5)  # 7信任5
    
    # 3. 创建加密数据集（模拟数据所有者8创建了包含用户5的策略）
    print("\n3. 创建模拟加密数据集...")
    # 直接操作system的access_policies
    curator.system.access_policies[8] = [5, 6, 7]  # 所有者8的策略包含用户5
    curator.system.access_policies[5] = [5, 6]     # 所有者5的策略包含自己
    curator.system.access_policies[6] = [6, 7]     # 所有者6的策略不包含用户5
    print(f"   已创建3个模拟加密数据集")
    
    # 4. 撤销用户5
    print("\n4. 撤销用户5...")
    success = curator.revoke_user(5)
    assert success, "撤销失败"
    
    # 5. 验证状态
    print("\n5. 验证状态...")
    assert curator.is_revoked(5), "用户5应被标记为已撤销"
    assert not curator.is_revoked(6), "用户6不应被撤销"
    
    revoked_list = curator.get_revoked_users()
    print(f"   已撤销用户: {revoked_list}")
    assert 5 in revoked_list, "撤销列表应包含5"
    
    info = curator.get_revocation_info(5)
    print(f"   撤销信息: {list(info.keys())}")
    
    # 6. 尝试为被撤销用户生成新密钥（应失败）
    print("\n6. 尝试为被撤销用户5生成新密钥（预期失败）...")
    try:
        sk, pk, pap = curator.generate_user_key(5)
        print(f"     错误：应该失败但成功了")
        assert False, "generate_user_key 应该拒绝被撤销用户"
    except ValueError as e:
        print(f"     正确拒绝: {e}")
    
    # 7. 尝试重新注册被撤销用户（应失败）
    print("\n7. 尝试重新注册用户5（预期失败）...")
    try:
        success = curator.register(5, "dummy_pk", [None] * curator.params.n)
        assert not success, "register应该返回False"
        print(f"     register返回False，正确拒绝")
    except Exception as e:
        print(f"     正确拒绝（抛出异常）: {e}")
    
    # 8. 获取受影响的所有者
    print("\n8. 获取受影响的所有者...")
    affected = curator.get_affected_owners(5)
    print(f"   受影响的所有者: {affected}")
    assert 8 in affected, "所有者8应受影响（策略包含用户5）"
    assert 5 in affected, "所有者5应受影响（策略包含自己）"
    assert 6 not in affected, "所有者6不应受影响（策略不包含用户5）"
    
    # 9. 更新策略示例
    print("\n9. 测试策略更新...")
    if 8 in affected:
        # 模拟所有者8更新策略
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
            print(f"   策略更新成功，新策略: {updated.get('P', [])}")
            assert 5 not in updated.get('P', []), "新策略不应包含被撤销用户"
    
    # 10. 最终状态
    print("\n10. 最终状态:")
    print(f"    注册用户数: {len(curator.registered_users)}")
    print(f"    撤销用户数: {len(curator.get_revoked_users())}")
    print(f"    撤销操作次数: {curator.stats['revoke_operations']}")
    print(f"    受影响的所有者数量: {len(affected)}")
    
    print(f"\n  Key Curator Revoke 测试通过")
    
    return curator

if __name__ == "__main__":
    
    # 运行双方案测试
    test_key_curator_with_both_schemes()
    
    # 运行撤销功能测试
    test_revoke_functionality()
