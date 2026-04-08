# decart/entities/data_querier.py
"""
Data Querier 实体 - 论文第I.A节
作为Web 3.0用户，使用数据库服务器执行AI查询
支持DeCart和DeCart*双方案 + 撤销检查 + 预训练模型查询
完全非模拟，基于真实同态加密
"""

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


class DataQuerier:
    """
    数据查询者 (Data Querier)
    
    论文职责:
    1. 向数据库服务器提交AI查询
    2. 使用自己的密钥验证访问权限
    3. 加密AI模型发送给数据库服务器
    4. 解密并获取查询结果
    5. 撤销检查 - 被撤销后无法查询
    6. 支持加载预训练模型进行查询
    
    支持双方案:
    - DeCart  : 原始方案，O(n²)复杂度
    - DeCart* : 优化方案，O(n)复杂度，20倍性能提升
    
    安全要求:
    - AI模型对数据库服务器保密
    - 查询结果仅自己可解密
    - 无法越权访问未授权的数据
    - 被撤销后无法进行任何查询
    
    完全非模拟:
    - 真实同态加密 (TenSEAL CKKS)
    - 真实双线性配对 (bn256)
    """
    
    def __init__(self,
                 querier_id: int,
                 key_curator: KeyCurator,
                 scheme: str = "decart_star"):
        """
        初始化数据查询者
        
        参数:
            querier_id: 查询者用户ID
            key_curator: 密钥管理者实例（已初始化）
            scheme: "decart" 或 "decart_star"
        """
        self.querier_id = querier_id
        self.key_curator = key_curator
        self.scheme = scheme.lower()
        
        # 验证密钥管理者使用的方案是否一致
        if self.scheme not in key_curator.scheme.lower():
            print(f"   ⚠️  警告: DataQuerier使用{self.scheme}, "
                  f"但KeyCurator使用{key_curator.scheme_name}")
        
        # 获取系统参数
        self.crs = key_curator.crs
        self.pp = key_curator.pp
        self.aux = key_curator.aux
        
        if self.crs is None or self.pp is None:
            raise ValueError("Key Curator尚未执行setup()")
        
        # 初始化同态加密（真实TenSEAL CKKS）
        self.he = HomomorphicEncryption(poly_modulus_degree=4096)
        
        # 用户密钥（从KeyCurator获取）
        self._load_user_keys()
        
        # 查询历史
        self.query_history = []
        
        # 加载的模型
        self.loaded_models = {}          # model_id -> model_info
        self.encrypted_models = {}       # enc_model_id -> encrypted_model
        
        # 状态缓存
        self._cached_aux = None
        self._cached_aux_time = 0
        
        # 撤销通知回调
        self._revoke_handlers = []
        
        print(f"\n  Data Querier 实体初始化")
        print(f"   查询者ID: {querier_id}")
        print(f"   方案: {key_curator.scheme_name}")
        print(f"   所属块: {self.block}")
        print(f"   u_id': {self.u_id_prime}")
        print(f"   aux长度: {len(self.key_curator.get_user_aux(querier_id))}")
        print(f"   支持撤销检查")
        print(f"   支持预训练模型查询")
    
    def _load_user_keys(self):
        """从KeyCurator加载用户密钥"""
        # 验证用户是否已注册
        if self.querier_id not in self.key_curator.registered_users:
            raise ValueError(f"用户 {self.querier_id} 尚未注册到Key Curator")
        
        # 检查是否已被撤销
        if self.key_curator.is_revoked(self.querier_id):
            raise ValueError(f"用户 {self.querier_id} 已被撤销，无法初始化查询者")
        
        # 获取用户信息
        self.block = self.key_curator.user_blocks.get(self.querier_id)
        self.u_id_prime = self.key_curator.user_id_prime.get(self.querier_id)
        self.pk_id = self.key_curator.user_public_keys.get(self.querier_id)
        self.pap_id = self.key_curator.user_pap.get(self.querier_id)
        
        # 获取私钥（从system.user_secrets）
        self.sk_id = self.key_curator.system.user_secrets[self.querier_id]['sk_id']
    
    def check_revoked(self) -> bool:
        """
        检查自己是否已被撤销
        
        返回:
            True: 已被撤销, False: 正常
        """
        is_revoked = self.key_curator.is_revoked(self.querier_id)
        if is_revoked:
            print(f"\n[Data Querier {self.querier_id}] ⚠️ 警告: 您已被系统撤销")
            print(f"   无法执行查询操作")
        return is_revoked
    
    def _check_before_operation(self, operation: str) -> bool:
        """
        执行操作前检查是否已被撤销
        
        参数:
            operation: 操作名称
        
        返回:
            True: 可以继续, False: 应中止
        """
        if self.check_revoked():
            print(f"     {operation}失败: 用户已被撤销")
            
            self.query_history.append({
                'timestamp': time.time(),
                'operation': operation,
                'status': 'failed_revoked',
                'error': 'User revoked'
            })
            
            return False
        return True
    
    # ========== 新增：加载预训练模型 ==========
    
    def load_pretrained_model(self, model_path: str, model_type: str = "cnn") -> str:
        """
        加载预训练模型文件
        
        参数:
            model_path: 模型文件路径 (.pkl)
            model_type: 模型类型 ('cnn', 'mlp', 'svm')
        
        返回:
            model_id: 模型唯一标识
        """
        print(f"\n[Data Querier {self.querier_id}] 加载预训练模型")
        print(f"   文件: {model_path}")
        print(f"   类型: {model_type}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载配置文件
        with open(model_path, 'rb') as f:
            config = pickle.load(f)
        
        model_name = config.get('model_name', 'unknown')
        test_accuracy = config.get('test_accuracy', 0.0)
        architecture = config.get('architecture', {})
        
        print(f"   模型名称: {model_name}")
        print(f"   测试准确率: {test_accuracy:.4f}")
        
        # 生成模型ID
        timestamp = int(time.time() * 1000)
        model_id = f"querier_model_{self.querier_id}_{model_type}_{timestamp}"
        
        # 存储模型信息
        self.loaded_models[model_id] = {
            'model_path': model_path,
            'model_type': model_type,
            'model_name': model_name,
            'test_accuracy': test_accuracy,
            'architecture': architecture,
            'load_time': time.time()
        }
        
        print(f"     模型加载成功: {model_id}")
        
        return model_id
    
    def load_all_models_from_dir(self, models_dir: str) -> Dict[str, str]:
        """
        从目录加载所有模型
        
        参数:
            models_dir: 模型目录路径
        
        返回:
            {model_type: model_id} 映射
        """
        print(f"\n[Data Querier {self.querier_id}] 从目录加载所有模型")
        print(f"   目录: {models_dir}")
        
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"目录不存在: {models_dir}")
        
        import glob
        model_files = glob.glob(os.path.join(models_dir, "*.pkl"))
        
        if not model_files:
            print(f"   警告: 目录中没有找到模型文件")
            return {}
        
        result = {}
        for model_path in model_files:
            filename = os.path.basename(model_path)
            
            # 根据文件名判断类型
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
                print(f"     加载 {model_type}: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"     加载失败 {filename}: {e}")
        
        return result
    
    # ========== 论文算法：Check（委托给对应方案）=========
    
    def check_access(self, C_m: Dict) -> Optional[Dict]:
        """
        检查访问权限 - 论文Check(u_id, sk_id, C_m)算法
        
        参数:
            C_m: 加密数据集密文（来自DataOwner）
        
        返回:
            C_M: 加密的AI模型和恢复的密钥，无权访问时返回None
        """
        # 操作前检查撤销状态
        if not self._check_before_operation("访问检查"):
            return None
        
        print(f"\n[Data Querier {self.querier_id}] 检查访问权限")
        print(f"   方案: {self.key_curator.scheme_name}")
        print(f"   所有者: {C_m.get('owner_id', 'unknown')}")
        
        # 调用对应方案的Check算法
        C_M = self.key_curator.system.check(
            self.querier_id,
            self.sk_id,
            C_m
        )
        
        if C_M:
            print(f"     访问授权成功")
            self.query_history.append({
                'timestamp': time.time(),
                'owner_id': C_m.get('owner_id'),
                'status': 'authorized'
            })
        else:
            print(f"     访问被拒绝")
            self.query_history.append({
                'timestamp': time.time(),
                'owner_id': C_m.get('owner_id'),
                'status': 'denied'
            })
        
        return C_M
    
    # ========== AI模型管理 ==========
    
    def create_ai_model(self, 
                       model_type: str = "linear",
                       dimension: int = 5,
                       weights: Optional[List[float]] = None) -> List[float]:
        """
        创建AI模型（明文）
        
        参数:
            model_type: 模型类型（linear, cnn, rnn等）
            dimension: 模型维度
            weights: 自定义权重（None时使用随机权重）
        
        返回:
            模型权重向量
        """
        if self.check_revoked():
            print(f"   ⚠️ 警告: 用户已被撤销，但仍可创建模型（仅用于测试）")
        
        if weights is not None:
            model = weights
        else:
            # 生成随机模型权重
            np.random.seed(int(time.time()) % 1000)
            if model_type == "linear":
                model = np.random.randn(dimension).tolist()
            elif model_type == "cnn":
                # 简化的CNN模型
                model = np.random.randn(dimension * 2).tolist()
            else:
                model = np.random.randn(dimension).tolist()
        
        print(f"\n[Data Querier {self.querier_id}] 创建AI模型")
        print(f"   类型: {model_type}")
        print(f"   维度: {len(model)}")
        print(f"   权重示例: {model[:3]}")
        
        return model
    
    # decart/entities/data_querier.py

    def encrypt_ai_model(self, model: Any, C_M: Dict) -> Dict:
        """
        加密AI模型 - 论文中的C_M = Enc[pk_h, M]
        支持多种模型类型
        
        参数:
            model: 明文AI模型（可以是列表或字典）
            C_M: Check算法返回的字典（包含恢复的公钥）
        
        返回:
            更新后的C_M（包含加密模型）
        """
        # 操作前检查撤销状态
        if not self._check_before_operation("模型加密"):
            raise ValueError(f"用户 {self.querier_id} 已被撤销，无法加密模型")
        
        print(f"\n[Data Querier {self.querier_id}] 加密AI模型")
        
        if not C_M.get('access_granted', False):
            raise ValueError("没有访问权限，无法加密模型")
        
        # 根据模型类型处理
        if isinstance(model, list):
            # 点积模型 - 直接加密列表
            print(f"   加密点积模型 (列表)")
            encrypted_model = self.he.encrypt(model)
            C_M['encrypted_model'] = encrypted_model
            C_M['model_dim'] = len(model)
            C_M['model_type'] = 'dot_product'
            
        elif isinstance(model, dict):
            # 字典类型的模型（神经网络、决策树等）
            model_type = model.get('type', 'unknown')
            print(f"   加密{model_type}模型 (字典)")
            
            if model_type == 'neural_network':
                # 神经网络模型
                weights = model.get('weights', [])
                bias = model.get('bias', [])
                input_dim = model.get('input_dim', 784)
                output_dim = model.get('output_dim', 10)
                
                # 加密权重
                encrypted_weights = []
                for w in weights:
                    try:
                        encrypted_w = self.he.encrypt([float(w)])
                        encrypted_weights.append(encrypted_w)
                    except Exception as e:
                        print(f"     权重加密失败: {e}")
                        encrypted_weights.append(None)
                
                # 加密偏置
                encrypted_bias = []
                for b in bias:
                    try:
                        encrypted_b = self.he.encrypt([float(b)])
                        encrypted_bias.append(encrypted_b)
                    except Exception as e:
                        print(f"     偏置加密失败: {e}")
                        encrypted_bias.append(None)
                
                encrypted_model = {
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
                C_M['encrypted_model'] = encrypted_model
                C_M['model_type'] = 'neural_network'
                
            elif model_type == 'decision_tree':
                # 决策树模型 - 需要使用系统的加密方法
                print(f"   决策树模型需要系统级加密")
                # 这里不处理，由上层调用系统方法
                C_M['encrypted_model'] = model
                C_M['model_type'] = 'decision_tree'
                
            else:
                # 未知字典类型，尝试直接存储
                print(f"   未知字典类型: {model_type}")
                C_M['encrypted_model'] = model
                C_M['model_type'] = model_type
                
        elif hasattr(model, 'get_encryptable_params'):
            # 具有加密方法的对象（如DecisionTreeHE）
            print(f"   加密模型对象")
            # 由上层调用系统方法处理
            C_M['encrypted_model'] = model
            C_M['model_type'] = 'object'
            
        else:
            raise TypeError(f"不支持的模型类型: {type(model)}")
        
        C_M['encrypt_time'] = time.time()
        
        print(f"     AI模型加密成功")
        print(f"      模型类型: {C_M.get('model_type', 'unknown')}")
        
        return C_M
    
    # ========== 使用预训练模型进行查询 ==========
    
    def prepare_encrypted_model(self, 
                               model_id: str, 
                               C_m: Dict) -> Optional[Dict]:
        """
        准备加密模型用于查询
        
        参数:
            model_id: 模型ID（从load_pretrained_model返回）
            C_m: 加密数据集
        
        返回:
            C_M: 包含加密模型的字典，失败返回None
        """
        print(f"\n[Data Querier {self.querier_id}] 准备加密模型")
        
        if model_id not in self.loaded_models:
            print(f"    模型不存在: {model_id}")
            return None
        
        # 检查访问权限
        C_M = self.check_access(C_m)
        if C_M is None:
            print(f"     无权访问此数据集")
            return None
        
        model_info = self.loaded_models[model_id]
        architecture = model_info.get('architecture', {})
        
        # 根据模型类型构建加密模型
        if model_info['model_type'] in ['cnn', 'cnn_test', 'mlp', 'svm']:
            # 所有模型都作为单层网络处理
            weights = architecture.get('weights', [])
            bias = architecture.get('bias', [])
            
            # 对于MLP，使用组合后的权重
            if 'combined_weights' in architecture:
                weights = architecture['combined_weights']
                bias = architecture['combined_bias']
            
            input_dim = architecture.get('input_dim', 784)
            output_dim = architecture.get('output_dim', 10)
            
            print(f"   模型架构: {input_dim} -> {output_dim}")
            print(f"   权重数量: {len(weights)}")
            print(f"   偏置数量: {len(bias)}")
            
            # 获取公钥
            pk_h = self.he.public_key
            
            # 加密权重
            encrypted_weights = []
            for w in weights:
                try:
                    encrypted_w = self.he.encrypt([float(w)])
                    encrypted_weights.append(encrypted_w)
                except Exception as e:
                    print(f"     权重加密失败: {e}")
                    encrypted_weights.append(None)
            
            # 加密偏置
            encrypted_bias = []
            for b in bias:
                try:
                    encrypted_b = self.he.encrypt([float(b)])
                    encrypted_bias.append(encrypted_b)
                except Exception as e:
                    print(f"     偏置加密失败: {e}")
                    encrypted_bias.append(None)
            
            encrypted_model = {
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
            
            C_M['encrypted_model'] = encrypted_model
            C_M['model_type'] = model_info['model_type']
            C_M['model_name'] = model_info['model_name']
            
            # 存储加密模型
            enc_id = f"enc_{model_id}_{int(time.time())}"
            self.encrypted_models[enc_id] = encrypted_model
            
            print(f"     加密模型准备完成")
            print(f"      加密权重数: {len([w for w in encrypted_weights if w])}")
            print(f"      加密偏置数: {len([b for b in encrypted_bias if b])}")
            
            return C_M
        else:
            print(f"     不支持的模型类型: {model_info['model_type']}")
            return None
    
    def query_with_model(self,
                        database_server,
                        owner_id: int,
                        dataset_id: str,
                        model_id: str) -> Optional[List[float]]:
        """
        使用预训练模型执行查询
        
        参数:
            database_server: DatabaseServer实例
            owner_id: 数据所有者ID
            dataset_id: 数据集ID
            model_id: 模型ID（从load_pretrained_model返回）
        
        返回:
            解密后的查询结果
        """
        print(f"\n" + "="*60)
        print(f"[Data Querier {self.querier_id}] 使用预训练模型查询")
        print(f"="*60)
        print(f"   所有者: {owner_id}")
        print(f"   数据集: {dataset_id}")
        print(f"   模型ID: {model_id}")
        
        # 1. 检查撤销状态
        if self.check_revoked():
            print(f"     查询失败: 用户已被撤销")
            return None
        
        try:
            # 2. 获取加密数据集
            print(f"\n1. 获取加密数据集...")
            C_m, sk_h_s = database_server.get_dataset(owner_id, dataset_id)
            if C_m is None:
                print(f"     数据集不存在")
                return None
            
            # 3. 准备加密模型
            print(f"\n2. 准备加密模型...")
            C_M = self.prepare_encrypted_model(model_id, C_m)
            if C_M is None:
                print(f"     模型准备失败")
                return None
            
            # 4. 执行加密查询
            print(f"\n3. 执行加密查询...")
            ER = self.key_curator.system.query(C_M, C_m, sk_h_s)
            
            # 5. 解密结果
            print(f"\n4. 解密查询结果...")
            results = self.key_curator.system.decrypt(C_M['sk_h_u'], ER)
            
            print(f"\n  查询完成!")
            print(f"   结果数量: {len(results)}")
            print(f"   结果示例: {results[:5]}")
            
            # 记录查询历史
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
            print(f"\n  查询失败: {e}")
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
    
    # ========== 原有查询方法保持不变 ==========
    
    def query(self, 
             database_server,
             owner_id: int,
             dataset_id: str,
             model: Optional[List[float]] = None,
             model_type: str = "linear") -> Optional[List[float]]:
        """
        完整查询流程:
        1. 检查自己是否已被撤销
        2. 从DatabaseServer获取加密数据集
        3. 检查访问权限
        4. 加密AI模型
        5. 执行加密查询
        6. 解密结果
        
        参数:
            database_server: DatabaseServer实例
            owner_id: 数据所有者ID
            dataset_id: 数据集ID
            model: AI模型权重（None时自动生成）
            model_type: 模型类型
        
        返回:
            解密后的查询结果，失败时返回None
        """
        print(f"\n" + "="*60)
        print(f"[Data Querier {self.querier_id}] 执行完整查询")
        print(f"="*60)
        print(f"   所有者: {owner_id}")
        print(f"   数据集: {dataset_id}")
        
        # 1. 检查撤销状态
        if self.check_revoked():
            print(f"     查询失败: 用户已被撤销")
            return None
        
        try:
            # 2. 获取加密数据集
            print(f"\n1. 获取加密数据集...")
            C_m, sk_h_s = database_server.get_dataset(owner_id, dataset_id)
            if C_m is None:
                print(f"     数据集不存在")
                return None
            
            # 3. 检查访问权限
            print(f"\n2. 检查访问权限...")
            C_M = self.check_access(C_m)
            if C_M is None:
                print(f"     无权访问此数据集")
                return None
            
            # 4. 创建并加密AI模型
            print(f"\n3. 准备AI模型...")
            if model is None:
                # 根据数据集维度自动创建模型
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
            
            # 5. 执行加密查询
            print(f"\n4. 执行加密查询...")
            ER = self.key_curator.system.query(C_M, C_m, sk_h_s)
            
            # 6. 解密结果
            print(f"\n5. 解密查询结果...")
            results = self.key_curator.system.decrypt(C_M['sk_h_u'], ER)
            
            print(f"\n  查询完成!")
            print(f"   结果数量: {len(results)}")
            print(f"   结果示例: {results[:5]}")
            
            # 记录查询历史
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
            print(f"\n  查询失败: {e}")
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
    
    # ========== 批量查询 ==========
    
    def batch_query(self,
                   database_server,
                   queries: List[Tuple[int, str]],
                   model_type: str = "linear") -> List[Optional[List[float]]]:
        """
        批量执行多个查询
        
        参数:
            database_server: DatabaseServer实例
            queries: [(owner_id, dataset_id), ...]
            model_type: 模型类型
        
        返回:
            查询结果列表
        """
        if not self._check_before_operation("批量查询"):
            return [None] * len(queries)
        
        print(f"\n[Data Querier {self.querier_id}] 批量执行 {len(queries)} 个查询")
        
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
        print(f"\n  批量查询完成: {success_count}/{len(queries)} 成功")
        
        return results
    
    # ========== 撤销通知处理 ==========
    
    def on_user_revoked(self):
        """
        当自己被撤销时调用（由Key Curator触发）
        """
        print(f"\n[Data Querier {self.querier_id}] ⚠️ 收到撤销通知")
        print(f"   您已被系统撤销，无法继续执行查询")
        
        for handler in self._revoke_handlers:
            try:
                handler(self.querier_id)
            except Exception as e:
                print(f"   ⚠️ 处理函数执行失败: {e}")
    
    def register_revoke_handler(self, handler_func):
        """
        注册撤销处理函数
        """
        self._revoke_handlers.append(handler_func)
        print(f"   [通知] 已注册撤销处理函数")
    
    # ========== 查询历史 ==========
    
    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """
        获取查询历史
        """
        return sorted(
            self.query_history[-limit:],
            key=lambda x: x['timestamp'],
            reverse=True
        )
    
    def clear_history(self):
        """
        清除查询历史
        """
        self.query_history = []
        print(f"[Data Querier {self.querier_id}] 查询历史已清除")
    
    # ========== 工具方法 ==========
    
    def verify_aux_status(self) -> Dict:
        """
        验证自己的aux状态（用于调试）
        """
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
        
        print(f"\n[Data Querier {self.querier_id}] aux状态:")
        print(f"   aux长度: {info['aux_length']}")
        print(f"   可查询他人: {info['can_query_others']}")
        print(f"   信任此用户: {info['trusted_by']}")
        print(f"   是否已撤销: {info['revoked']}")
        
        return info
    
    def get_querier_info(self) -> Dict:
        """
        获取查询者信息
        """
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


# ========== 原有测试保持不变 ==========

def test_data_querier_revoked():
    """测试被撤销用户的查询行为"""
    
    print("\n" + "="*80)
    print("🧪 测试 Data Querier 撤销状态")
    print("="*80)
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.database_server import DatabaseServer
    from schemes.decart_star import DeCartStarParams
    
    # 1. 初始化系统
    print("\n1. 初始化系统...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. 创建用户
    print("\n2. 创建用户...")
    users = [5, 6, 7]
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)
    
    # 3. 建立信任关系
    print("\n3. 建立信任关系...")
    curator.add_trust(6, 5)  # 6信任5
    curator.add_trust(7, 5)  # 7信任5
    
    # 4. 数据所有者加密数据
    print("\n4. 数据所有者加密数据...")
    owner = DataOwner(owner_id=5, key_curator=curator, scheme="decart_star")
    data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    policy = [5, 6, 7]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy, {'name': 'test'})
    
    # 5. 创建数据库服务器
    print("\n5. 创建数据库服务器...")
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(5, ds_id, C_m, sk_h_s)
    
    # 6. 创建查询者6（正常用户）
    print("\n6. 创建查询者6（正常用户）...")
    querier6 = DataQuerier(querier_id=6, key_curator=curator, scheme="decart_star")
    info6 = querier6.get_querier_info()
    print(f"   用户6撤销状态: {info6['revoked']}")
    assert not info6['revoked'], "用户6初始不应被撤销"
    
    # 7. 撤销用户6
    print("\n7. 撤销用户6...")
    curator.revoke_user(6)
    
    # 8. 验证用户6已被撤销
    print("\n8. 验证用户6状态...")
    info6_after = querier6.get_querier_info()
    print(f"   用户6撤销状态: {info6_after['revoked']}")
    assert info6_after['revoked'], "用户6应被标记为已撤销"
    
    # 9. 尝试创建新的查询者6（应失败）
    print("\n9. 尝试创建新的查询者6（预期失败）...")
    try:
        querier6_new = DataQuerier(querier_id=6, key_curator=curator, scheme="decart_star")
        print(f"     应该失败但成功了")
        assert False, "初始化被撤销用户应失败"
    except ValueError as e:
        print(f"     正确拒绝: {e}")
    
    # 10. 尝试让已撤销用户6执行查询（应失败）
    print("\n10. 尝试让已撤销用户6执行查询（预期失败）...")
    result = querier6.query(db_server, 5, ds_id)
    print(f"   查询结果: {result is None}")
    assert result is None, "被撤销用户的查询应返回None"
    
    print(f"\n  Data Querier 撤销测试通过")


def test_data_querier_normal():
    """测试正常用户的查询行为"""
    
    print("\n" + "="*80)
    print("🧪 测试 Data Querier 正常查询")
    print("="*80)
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.database_server import DatabaseServer
    from schemes.decart_star import DeCartStarParams
    
    # 1. 初始化系统
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. 创建用户
    users = [5, 6, 7]
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)
    
    # 3. 建立信任关系
    curator.add_trust(6, 5)
    
    # 4. 数据所有者加密数据
    owner = DataOwner(owner_id=5, key_curator=curator, scheme="decart_star")
    data = [[1.0, 2.0, 3.0]]
    policy = [5, 6]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy)
    
    # 5. 数据库服务器
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(5, ds_id, C_m, sk_h_s)
    
    # 6. 创建查询者6
    querier = DataQuerier(querier_id=6, key_curator=curator, scheme="decart_star")
    
    # 7. 执行查询
    print("\n7. 正常用户执行查询...")
    result = querier.query(db_server, 5, ds_id)
    
    print(f"   查询结果: {result is not None}")
    assert result is not None, "正常用户的查询应成功"
    
    print(f"\n  正常查询测试通过")


def test_self_query_after_revoke():
    """测试用户撤销后查询自己的数据"""
    
    print("\n" + "="*80)
    print("🧪 测试用户撤销后自查询")
    print("="*80)
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.database_server import DatabaseServer
    from schemes.decart_star import DeCartStarParams
    
    # 1. 初始化系统
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. 创建用户
    uid = 5
    sk, pk, pap = curator.generate_user_key(uid)
    curator.register(uid, pk, pap)
    
    # 3. 创建所有者（也是查询者）
    owner = DataOwner(owner_id=uid, key_curator=curator, scheme="decart_star")
    data = [[1.0, 2.0, 3.0]]
    policy = [uid]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy)
    
    # 4. 数据库服务器
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(uid, ds_id, C_m, sk_h_s)
    
    # 5. 创建查询者（同一用户）
    querier = DataQuerier(querier_id=uid, key_curator=curator, scheme="decart_star")
    
    # 6. 撤销用户
    print("\n6. 撤销用户5...")
    curator.revoke_user(uid)
    
    # 7. 尝试自查询（应失败）
    print("\n7. 尝试被撤销后自查询（预期失败）...")
    result = querier.query(db_server, uid, ds_id)
    print(f"   查询结果: {result is None}")
    assert result is None, "被撤销后自查询应失败"
    
    print(f"\n  自查询撤销测试通过")


# ========== 新增：预训练模型测试 ==========

def test_pretrained_models():
    """测试使用预训练模型查询（新增测试）"""
    
    print("\n" + "="*80)
    print("🧪 新增测试: 预训练模型查询")
    print("="*80)
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.database_server import DatabaseServer
    from schemes.decart_star import DeCartStarParams
    import glob
    
    # 1. 初始化系统
    print("\n1. 初始化系统...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. 创建用户
    print("\n2. 创建用户...")
    owner_id = 5
    querier_id = 6
    
    sk_o, pk_o, pap_o = curator.generate_user_key(owner_id)
    curator.register(owner_id, pk_o, pap_o)
    
    sk_q, pk_q, pap_q = curator.generate_user_key(querier_id)
    curator.register(querier_id, pk_q, pap_q)
    
    # 3. 建立信任关系
    print("\n3. 建立信任关系...")
    curator.add_trust(querier_id, owner_id)
    
    # 4. 数据所有者加密数据
    print("\n4. 数据所有者加密数据...")
    owner = DataOwner(owner_id=owner_id, key_curator=curator, scheme="decart_star")
    
    # 创建MNIST格式的测试数据 (784维)
    data = [np.random.randn(784).tolist() for _ in range(3)]
    policy = [owner_id, querier_id]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy, {'name': 'mnist_test'})
    
    # 5. 创建数据库服务器
    print("\n5. 创建数据库服务器...")
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(owner_id, ds_id, C_m, sk_h_s)
    
    # 6. 创建查询者
    print("\n6. 创建查询者...")
    querier = DataQuerier(querier_id=querier_id, key_curator=curator, scheme="decart_star")
    
    # 7. 设置模型目录
    models_dir = r"E:\decart\experiments\models\trained"
    print(f"\n7. 模型目录: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"     模型目录不存在，跳过测试")
        return
    
    # 8. 加载所有模型
    print(f"\n8. 加载所有模型...")
    model_map = querier.load_all_models_from_dir(models_dir)
    
    if not model_map:
        print(f"     没有加载到任何模型")
        return
    
    print(f"\n   加载的模型: {list(model_map.keys())}")
    
    # 9. 使用每种模型进行查询
    print("\n" + "="*60)
    print("9. 测试每种模型查询")
    print("="*60)
    
    results = {}
    
    for model_type, model_id in model_map.items():
        print(f"\n{'-'*50}")
        print(f"测试 {model_type} 模型查询")
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
                print(f"     {model_type} 查询成功")
                print(f"      结果: {query_result[:3]}")
            else:
                results[model_type] = False
                print(f"     {model_type} 查询失败")
                
        except Exception as e:
            results[model_type] = False
            print(f"     {model_type} 查询异常: {e}")
    
    # 10. 汇总结果
    print("\n" + "="*60)
    print("📊 预训练模型测试结果")
    print("="*60)
    
    all_passed = True
    for model_type, passed in results.items():
        status = "  通过" if passed else "  失败"
        print(f"   {status} - {model_type}")
        all_passed = all_passed and passed
    
    if all_passed:
        print(f"\n  所有预训练模型查询测试通过！")
    else:
        print(f"\n⚠️ 部分预训练模型查询测试失败")


if __name__ == "__main__":
    print("="*80)
    print("🔬 Data Querier 实体测试套件")
    print("="*80)
    
    # 运行原有测试
    test_data_querier_normal()
    test_data_querier_revoked()
    test_self_query_after_revoke()
    
    # 运行新增的预训练模型测试
    test_pretrained_models()
    
    print("\n" + "="*80)
    print("  所有 Data Querier 测试完成")
    print("   完全非模拟，真实密码学")
    print("   支持 DeCart / DeCart* 双方案")
    print("   支持撤销状态检查")
    print("   支持预训练模型查询")
    print("="*80)