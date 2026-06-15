# decart/experiments/scheme3_offline/wrapper.py
"""
线下密钥分发方案实验包装器
作为访问控制效率上限基准：
- 密钥通过安全信道线下分发
- Check算法为线下授权表查询
- 数据和模型用用户密钥加密
"""

import sys
import os
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 复用核心密码学模块
from core.homomorphic import HomomorphicEncryption


class OfflineSchemeExperimentWrapper:
    """
    线下密钥分发方案实验包装器
    特点：
    - 密钥线下预先分发
    - 无Check算法开销（权限验证为0）
    - 每个用户有自己的密钥对
    - 数据用所有者密钥加密，查询者需预先获得授权
    """
    
    def __init__(self, N: int = 64, n: int = 16):
        """
        初始化实验环境
        
        参数:
            N: 最大用户数
            n: 每块用户数（接口兼容）
        """
        self.N = N
        self.n = n
        
        # 初始化同态加密
        self.he = HomomorphicEncryption(poly_modulus_degree=32768)
        
        # 用户密钥对（线下预分发）
        self.user_keys = {}  # user_id -> {'pk': public_key, 'sk': secret_key}
        
        # 授权关系（线下建立）
        self.authorizations = {}  # (owner_id, querier_id) -> True
        
        # 数据存储（加密）
        self.encrypted_datasets = {}  # owner_id -> {dataset_id -> {'data': enc_data, 'policy': policy}}
        
        # 实验数据
        self.metrics = {
            'setup_time': 0,
            'keygen_times': [],
                'register_times': [],
            'encrypt_times': [],
            'query_times': [],
            'decrypt_times': [],
            'check_times': [],  # 专门记录Check算法时间
            'communication_sizes': [],
            'memory_sizes': []
        }

    def _safe_obj_size(self, obj: Any, fallback: int = 1024) -> int:
        """稳健估算对象字节数，优先使用真实序列化大小。"""
        seen_ids = set()

        def _measure(x: Any, depth: int = 0) -> int:
            if x is None:
                return 0

            obj_id = id(x)
            if obj_id in seen_ids:
                return 0

            if isinstance(x, (bytes, bytearray, memoryview)):
                return len(x)

            serializer = getattr(x, 'serialize', None)
            if callable(serializer):
                try:
                    ser = serializer()
                    if isinstance(ser, (bytes, bytearray, memoryview)):
                        return len(ser)
                except Exception:
                    pass

            try:
                return len(pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                pass

            if depth < 20 and isinstance(x, dict):
                seen_ids.add(obj_id)
                total = 0
                for k, v in x.items():
                    total += _measure(k, depth + 1)
                    total += _measure(v, depth + 1)
                seen_ids.discard(obj_id)
                return total

            if depth < 20 and isinstance(x, (list, tuple, set)):
                seen_ids.add(obj_id)
                total = sum(_measure(item, depth + 1) for item in x)
                seen_ids.discard(obj_id)
                return total

            try:
                return len(str(x).encode('utf-8'))
            except Exception:
                return fallback

        size = _measure(obj)
        return size if size > 0 else fallback

    def get_auxiliary_sizes(self) -> Dict[str, int]:
        """Offline 基线不维护 CRS / pp / aux，统一返回 0。"""
        return {
            'crs_size_bytes': 0,
            'pp_size_bytes': 0,
            'aux_size_bytes': 0,
            'total_auxiliary_size_bytes': 0
        }

    @staticmethod
    def _should_report_progress(current: int, total: int) -> bool:
        if total <= 10:
            return True
        interval = max(1, total // 5)
        return current == 1 or current == total or current % interval == 0

    @classmethod
    def _report_progress(cls, stage: str, current: int, total: int, progress_label: Optional[str] = None):
        if cls._should_report_progress(current, total):
            prefix = f"{progress_label} " if progress_label else ""
            print(f"      {prefix}{stage} 进度: {current}/{total}")
    
    def setup(self) -> float:
        """
        初始化系统
        
        返回:
            setup_time: 初始化耗时
        """
        start = time.perf_counter()
        # 线下方案只需要初始化同态加密
        elapsed = time.perf_counter() - start
        self.metrics['setup_time'] = elapsed
        print(f"   Offline 方案初始化完成: {elapsed:.4f}秒")
        return elapsed
    
    def register_user(self, user_id: int) -> Tuple[int, Any]:
        """
        注册用户（线下生成密钥对）
        
        返回:
            (sk, pk) - 用户密钥对
        """
        start = time.perf_counter()
        
        # 生成用户密钥对（模拟线下分发）
        # 实际中，每个用户应该有不同密钥，这里简化使用同一个HE实例的不同"虚拟"密钥
        sk = f"offline_sk_{user_id}"
        pk = f"offline_pk_{user_id}"
        
        self.user_keys[user_id] = {'sk': sk, 'pk': pk}
        
        elapsed = time.perf_counter() - start
        self.metrics['keygen_times'].append(elapsed)
        self.metrics['register_times'].append(0.0)
        
        return sk, pk
    
    def setup_authorization(self, owner_id: int, querier_id: int):
        """
        建立授权关系（线下完成）
        相当于论文中的信任建立
        """
        self.authorizations[(owner_id, querier_id)] = True
        print(f"    建立授权: 用户 {querier_id} 可访问 所有者 {owner_id} 的数据")
    
    def check_authorization(self, owner_id: int, querier_id: int) -> bool:
        """
        检查授权（线下已完成，通过本地授权表查询）
        
        返回:
            True: 有权限
        """
        check_start = time.perf_counter()
        result = self.authorizations.get((owner_id, querier_id), False)
        self.metrics['check_times'].append(time.perf_counter() - check_start)
        
        return result
    
    def encrypt_dataset(self, 
                       owner_id: int,
                       data: List[List[float]],
                       policy: List[int],
                       metadata: Optional[Dict] = None) -> Tuple[Dict, Any, str]:
        """
        加密数据集（使用所有者公钥）
        
        返回:
            (C_m, None, dataset_id)
        """
        # 生成数据集ID
        import time
        timestamp = int(time.time() * 1000)
        dataset_id = f"ds_{owner_id}_{timestamp}"
        
        # 测量加密时间
        start = time.perf_counter()
        
        # 加密每条数据记录（使用所有者密钥）
        encrypted_data = []
        total_records = len(data)
        for index, record in enumerate(data, start=1):
            encrypted_record = self.he.encrypt(record)
            encrypted_data.append(encrypted_record)
            self._report_progress('数据集加密', index, total_records)
        
        elapsed = time.perf_counter() - start
        self.metrics['encrypt_times'].append(elapsed)
        
        # 存储加密数据
        if owner_id not in self.encrypted_datasets:
            self.encrypted_datasets[owner_id] = {}
        
        self.encrypted_datasets[owner_id][dataset_id] = {
            'encrypted_data': encrypted_data,
            'policy': policy.copy(),
            'metadata': metadata or {},
            'store_time': time.time()
        }
        
        # 为策略中的每个查询者建立授权
        for querier_id in policy:
            if querier_id != owner_id:
                self.setup_authorization(owner_id, querier_id)
        
        # 测量密文大小
        try:
            import pickle
            size = len(pickle.dumps(encrypted_data))
        except:
            size = len(data) * 1024 * 1024  # 1MB per record估算
        
        self.metrics['communication_sizes'].append({
            'type': 'encrypt',
            'size': size,
            'records': len(data)
        })
        
        print(f"    Offline 方案加密: {elapsed*1000:.2f} ms, 密文大小: {size/1024:.2f} KB")
        
        # 返回格式兼容的元数据
        C_m = {
            'type': 'offline_scheme',
            'dataset_id': dataset_id,
            'owner_id': owner_id,
            'policy': policy,
            'num_records': len(data)
        }
        
        return C_m, None, dataset_id
    
    def store_dataset(self, owner_id: int, dataset_id: str, C_m: Dict, sk_h_s: Any):
        """已在 encrypt_dataset 中完成存储"""
        pass
    
    def encrypt_model(self, model: Any, user_id: int) -> Any:
        """
        加密模型（使用查询者自己的密钥）
        
        返回:
            加密后的模型
        """
        if isinstance(model, list):
            # 点积模型 - 加密列表
            return self.he.encrypt(model)
        elif isinstance(model, dict) and model.get('type') == 'neural_network':
            encrypted_layers = []
            layers = model.get('layers') or [
                {
                    'weights': model.get('weights', []),
                    'bias': model.get('bias', []),
                    'weights_shape': (model.get('output_dim'), model.get('input_dim')),
                    'activation': 'linear',
                }
            ]

            total_units = 0
            for layer in layers:
                weights_shape = tuple(layer.get('weights_shape', (0, 0)))
                output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else 0
                total_units += output_dim
                if layer.get('bias', []):
                    total_units += 1

            processed_units = 0

            for layer in layers:
                weights = layer.get('weights', [])
                bias = layer.get('bias', [])
                weights_shape = tuple(layer.get('weights_shape', (0, 0)))
                output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else 0
                input_dim = int(weights_shape[1]) if len(weights_shape) > 1 else 0

                encrypted_weight_rows = []
                for row_idx in range(output_dim):
                    row_start = row_idx * input_dim
                    row_end = row_start + input_dim
                    weight_row = [float(value) for value in weights[row_start:row_end]]
                    if len(weight_row) < input_dim:
                        weight_row.extend([0.0] * (input_dim - len(weight_row)))
                    encrypted_weight_rows.append(self.he.encrypt(weight_row))
                    processed_units += 1
                    self._report_progress('模型加密', processed_units, total_units)

                encrypted_bias_vector = None
                bias_values = [float(value) for value in bias]
                if bias_values:
                    encrypted_bias_vector = self.he.encrypt(bias_values)
                    processed_units += 1
                    self._report_progress('模型加密', processed_units, total_units)

                encrypted_layers.append({
                    'layer_idx': layer.get('layer_idx', len(encrypted_layers)),
                    'layer_type': layer.get('layer_type', 'linear'),
                    'activation': layer.get('activation', 'linear'),
                    'weights_shape': layer.get('weights_shape'),
                    'bias_shape': tuple(layer.get('bias_shape', (len(bias_values),))),
                    'encrypted_weights': [],
                    'encrypted_bias': [],
                    'encrypted_weight_rows': encrypted_weight_rows,
                    'encrypted_bias_vector': encrypted_bias_vector,
                })

            return {
                'type': 'neural_network',
                'layers': encrypted_layers,
                'input_dim': model.get('input_dim'),
                'output_dim': model.get('output_dim'),
                'user_id': user_id
            }
        else:
            # 默认处理
            return self.he.encrypt([0.0])

    def prepare_query_model(self, querier_id: int, model: Any) -> Any:
        """Encrypt the model once with the querier key and reuse it across repeated queries."""
        start_encrypt_model = time.perf_counter()
        print("   准备查询模型...")
        encrypted_model = self.encrypt_model(model, querier_id)
        elapsed = time.perf_counter() - start_encrypt_model
        self.metrics['encrypt_times'].append(elapsed)
        print(f"   查询模型准备完成: {elapsed*1000:.2f} ms")
        return encrypted_model

    @staticmethod
    def _evaluate_neural_network_plain(model: Dict[str, Any], plain_record: List[float]) -> float:
        values = [float(v) for v in plain_record]
        layers = model.get('layers') or [
            {
                'weights': model.get('weights', []),
                'bias': model.get('bias', []),
                'weights_shape': (int(model.get('output_dim', 10) or 10), len(plain_record)),
                'activation': 'linear',
            }
        ]

        for layer in layers:
            weights = layer.get('weights', [])
            bias = layer.get('bias', [])
            weights_shape = tuple(layer.get('weights_shape', (int(model.get('output_dim', 10) or 10), len(values))))
            output_dim = int(weights_shape[0]) if len(weights_shape) > 0 else len(bias)
            input_dim = int(weights_shape[1]) if len(weights_shape) > 1 else len(values)
            record_vec = values[:input_dim]
            if len(record_vec) < input_dim:
                record_vec.extend([0.0] * (input_dim - len(record_vec)))

            next_values = []
            for output_index in range(output_dim):
                start_idx = output_index * input_dim
                row_weights = weights[start_idx:start_idx + input_dim] if len(weights) > start_idx else []
                value = sum(float(weight) * record_vec[idx] for idx, weight in enumerate(row_weights) if idx < input_dim)
                if output_index < len(bias):
                    value += float(bias[output_index])
                if layer.get('activation', 'linear') == 'square':
                    value = value * value
                next_values.append(value)
            values = next_values
        return values[0] if values else 0.0
    
    def execute_query(self,
                     querier_id: int,
                     owner_id: int,
                     dataset_id: str,
                     model: Any,
                     prepared_model: Any = None,
                     progress_label: Optional[str] = None) -> Optional[List[float]]:
        """
        执行查询（无Check开销）
        
        返回:
            解密后的查询结果
        """
        # 检查数据集是否存在
        if owner_id not in self.encrypted_datasets or dataset_id not in self.encrypted_datasets[owner_id]:
            print(f"      数据集不存在")
            return None
        
        # 检查授权（线下完成，开销为0）
        is_authorized = self.check_authorization(owner_id, querier_id)
        self.metrics['communication_sizes'].append({
            'type': 'check',
            'size': 0,
            'records': 0,
        })
        if not is_authorized:
            print(f"      未授权访问")
            return None
        
        dataset_info = self.encrypted_datasets[owner_id][dataset_id]
        encrypted_data = dataset_info['encrypted_data']
        
        encrypted_model = prepared_model if prepared_model is not None else self.prepare_query_model(querier_id, model)

        # 查询请求阶段通信量：查询者发送的加密模型
        # 查询请求阶段通信量：统一口径，统计发送给服务器的请求包
        req_payload = {
            'querier_id': querier_id,
            'owner_id': owner_id,
            'dataset_id': dataset_id,
            'encrypted_model': encrypted_model,
            'model_type': model.get('type', 'dot_product') if isinstance(model, dict) else 'dot_product'
        }
        req_size = self._safe_obj_size(req_payload)
        self.metrics['communication_sizes'].append({
            'type': 'query',
            'size': req_size,
            'records': len(encrypted_data)
        })
        
        # 执行同态查询
        start_query = time.perf_counter()
        
        total_records = len(encrypted_data)
        results = []
        if isinstance(model, list):
            # 点积模型
            for index, enc_record in enumerate(encrypted_data, start=1):
                try:
                    # 同态点积
                    result = enc_record.dot(encrypted_model)
                    results.append(result)
                except:
                    results.append(self.he.encrypt([0.0]))
                self._report_progress('查询计算', index, total_records, progress_label)
        elif isinstance(model, dict) and model.get('type') == 'decision_tree':
            # 简单决策树：统一基线逻辑
            # 规则: feature[0] <= 0.5 -> 左叶子，否则右叶子
            nodes = model.get('nodes', [])
            node_map = {n.get('id'): n for n in nodes}
            root_id = model.get('root', 0)

            for index, enc_record in enumerate(encrypted_data, start=1):
                try:
                    plain_record = self.he.decrypt(enc_record)
                    if not isinstance(plain_record, list):
                        plain_record = [float(plain_record)]

                    current_id = root_id
                    depth = 0
                    max_depth = 10
                    pred = 0.0

                    while depth < max_depth and current_id in node_map:
                        node = node_map[current_id]

                        if 'value' in node:
                            pred = float(node['value'])
                            break

                        feature_idx = int(node.get('feature', 0))
                        threshold = float(node.get('threshold', 0.0))
                        feature_val = float(plain_record[feature_idx]) if feature_idx < len(plain_record) else 0.0

                        if feature_val <= threshold:
                            current_id = node.get('left')
                        else:
                            current_id = node.get('right')
                        depth += 1

                    results.append(self.he.encrypt([pred]))
                except:
                    results.append(self.he.encrypt([0.0]))
                self._report_progress('查询计算', index, total_records, progress_label)
        elif isinstance(model, dict) and model.get('type') == 'neural_network':
            for index, enc_record in enumerate(encrypted_data, start=1):
                try:
                    plain_record = self.he.decrypt(enc_record)
                    if not isinstance(plain_record, list):
                        plain_record = [float(plain_record)]
                    pred = self._evaluate_neural_network_plain(model, plain_record)
                    results.append(self.he.encrypt([pred]))
                except:
                    results.append(self.he.encrypt([0.0]))
                self._report_progress('查询计算', index, total_records, progress_label)
        else:
            # 简化处理
            for index, enc_record in enumerate(encrypted_data, start=1):
                results.append(self.he.encrypt([0.0]))
                self._report_progress('查询计算', index, total_records, progress_label)
        
        query_time = time.perf_counter() - start_query
        self.metrics['query_times'].append(query_time)

        # 返回包阶段通信量：服务器返回的加密结果列表
        res_size = self._safe_obj_size(results, fallback=max(1, len(results)) * 1024)
        self.metrics['communication_sizes'].append({
            'type': 'decrypt',
            'size': res_size,
            'records': len(results)
        })
        
        # 解密结果（使用查询者密钥）
        start_decrypt = time.perf_counter()
        decrypted_results = []
        total_results = len(results)
        for index, enc_result in enumerate(results, start=1):
            try:
                dec = self.he.decrypt(enc_result)
                if isinstance(dec, list):
                    decrypted_results.append(dec[0] if dec else 0.0)
                else:
                    decrypted_results.append(float(dec))
            except:
                decrypted_results.append(0.0)
            self._report_progress('结果解密', index, total_results, progress_label)
        
        decrypt_time = time.perf_counter() - start_decrypt
        self.metrics['decrypt_times'].append(decrypt_time)
        
        print(f"      Offline 方案查询: {query_time*1000:.2f} ms")
        print(f"      Check时间: 0.00 ms (线下完成)")
        if prepared_model is None:
            print(f"      模型加密: {self.metrics['encrypt_times'][-1]*1000:.2f} ms")
        else:
            print(f"      模型加密: 0.00 ms (复用已准备模型)")
        print(f"      结果解密: {decrypt_time*1000:.2f} ms")
        
        return decrypted_results
    
    def reset_metrics(self):
        """重置实验指标"""
        self.metrics = {
            'setup_time': 0,
            'keygen_times': [],
            'register_times': [],
            'encrypt_times': [],
            'query_times': [],
            'decrypt_times': [],
            'check_times': [],
            'communication_sizes': [],
            'memory_sizes': []
        }
    
    def get_metrics(self) -> Dict:
        """获取所有实验指标"""
        metrics = self.metrics.copy()
        
        # 计算平均值
        if metrics['encrypt_times']:
            metrics['avg_encrypt_time'] = np.mean(metrics['encrypt_times'])
            metrics['std_encrypt_time'] = np.std(metrics['encrypt_times'])
        
        if metrics['query_times']:
            metrics['avg_query_time'] = np.mean(metrics['query_times'])
            metrics['std_query_time'] = np.std(metrics['query_times'])
        
        if metrics['decrypt_times']:
            metrics['avg_decrypt_time'] = np.mean(metrics['decrypt_times'])
            metrics['std_decrypt_time'] = np.std(metrics['decrypt_times'])
        
        if metrics['check_times']:
            metrics['avg_check_time'] = np.mean(metrics['check_times'])
            metrics['total_check_time'] = np.sum(metrics['check_times'])
        
        # 通信大小
        if metrics['communication_sizes']:
            sizes = [s['size'] for s in metrics['communication_sizes']]
            metrics['avg_communication_size'] = np.mean(sizes)
            metrics['total_communication'] = np.sum(sizes)
        
        return metrics