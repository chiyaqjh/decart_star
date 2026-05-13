# decart/experiments/our_decart/runner.py
"""
DeCart 方案实验运行器
执行完整的对比实验，支持多模型测试
"""

import sys
import os
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 回到 decart 目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 直接导入 wrapper，不通过 __init__
from config import Config
from experiments.our_decart.wrapper import DeCartExperimentWrapper


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 系统参数
    N: int = Config.MAX_USERS  # 最大用户数
    n: int = Config.BLOCK_SIZE  # 每块用户数
    
    # 数据参数
    num_records: int = Config.EXPERIMENT_NUM_RECORDS  # 数据记录数
    record_dim: int = Config.EXPERIMENT_RECORD_DIM   # 记录维度
    
    # 模型参数
    model_types: List[str] = None  # 要测试的模型类型
    
    # 策略参数
    policy_size: int = Config.EXPERIMENT_POLICY_SIZE  # 访问策略中的用户数
    
    # 实验参数
    num_runs: int = Config.EXPERIMENT_NUM_RUNS      # 重复运行次数
    save_results: bool = True
    results_dir: str = "experiments/results/our_decart"
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['dot', 'decision_tree', 'neural_network']


class ExperimentRunner:
    """DeCart 实验运行器（支持多模型）"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {
            'config': asdict(config),
            'models': {},  # 按模型类型存储结果
            'summary': {}
        }
        
        # 为每种模型类型创建结果容器
        for model_type in config.model_types:
            self.results['models'][model_type] = {
                'setup_times': [],
                'encrypt_times': [],
                'query_times': [],
                'decrypt_times': [],
                'communication_sizes': [],
                'comm_upload_sizes': [],
                'comm_query_sizes': [],
                'comm_decrypt_sizes': [],
                'setup_crs_sizes': [],
                'setup_pp_sizes': [],
                'setup_aux_sizes': [],
                'setup_total_auxiliary_sizes': [],
                'register_crs_sizes': [],
                'register_pp_sizes': [],
                'register_aux_sizes': [],
                'register_total_auxiliary_sizes': [],
                'final_crs_sizes': [],
                'final_pp_sizes': [],
                'final_aux_sizes': [],
                'final_total_auxiliary_sizes': [],
                'runs': []
            }
        
        # 创建结果目录
        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)
    
    def generate_test_data(self) -> Tuple[List[List[float]], List[int]]:
        """生成测试数据"""
        data = []
        for _ in range(self.config.num_records):
            record = np.random.randn(self.config.record_dim).tolist()
            # 归一化到 [-1, 1] 范围内
            max_val = max(abs(min(record)), abs(max(record)))
            if max_val > 0:
                record = [x / max_val for x in record]
            data.append(record)
        
        return data, []
    
    def generate_model(self, model_type: str) -> Any:
        """根据模型类型生成模型 - 统一格式确保可加密"""
        
        if model_type == 'dot':
            # 点积模型 - 返回列表，可以直接加密
            print(f"   生成点积模型...")
            model = np.random.randn(self.config.record_dim).tolist()
            # 归一化
            max_val = max(abs(min(model)), abs(max(model))) or 1
            model = [x / max_val for x in model]
            return model
        
        elif model_type == 'decision_tree':
            # 决策树模型 - 返回可加密的字典格式
            print(f"   生成决策树模型...")
            try:
                from schemes.ai_model import DecisionTreeHE, DecisionTreeNode
                tree = DecisionTreeHE()
                
                # 创建简单决策树
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
                
                print(f"     决策树节点数: {len(tree.nodes)}")
                return tree
            except ImportError as e:
                print(f"     导入失败: {e}，使用字典格式")
                # 备用方案：返回字典格式
                return {
                    'type': 'decision_tree',
                    'format': 'dict',
                    'root': 0,
                    'nodes': [
                        {'id': 0, 'feature': 0, 'threshold': 0.5, 'left': 1, 'right': 2, 'is_leaf': False},
                        {'id': 1, 'value': 0.0, 'is_leaf': True},
                        {'id': 2, 'value': 1.0, 'is_leaf': True}
                    ]
                }
        
        elif model_type == 'neural_network':
            # 神经网络模型 - 返回可加密的字典格式
            print(f"   生成神经网络模型...")
            
            # 生成随机权重和偏置
            output_dim = 10
            input_dim = self.config.record_dim
            
            # 生成权重矩阵 (output_dim x input_dim)
            weights_matrix = np.random.randn(output_dim, input_dim) * 0.1
            bias = np.random.randn(output_dim) * 0.1
            
            # 展平权重
            weights = weights_matrix.flatten().tolist()
            
            # 限制权重数量避免过慢
            max_weights = 500  # 最大权重数量
            if len(weights) > max_weights:
                weights = weights[:max_weights]
                print(f"     权重数量: {len(weights)} (已截断)")
            
            # 返回统一格式
            return {
                'type': 'neural_network',
                'format': 'single_layer',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'weights': weights,
                'bias': bias.tolist(),
                'weights_shape': (output_dim, input_dim),
                'bias_shape': (output_dim,)
            }
        
        else:
            raise ValueError(f"未知模型类型: {model_type}")
    
    def register_all_users(self, wrapper: DeCartExperimentWrapper, policy: List[int]):
        """注册所有需要的用户"""
        print(f"\n📝 注册策略中的用户...")
        for uid in policy:
            try:
                wrapper.register_user(uid)
                print(f"   用户 {uid} 注册成功")
            except Exception as e:
                print(f"   用户 {uid} 注册失败: {e}")
    
    def run_single_experiment(self, run_id: int, model_type: str) -> Optional[Dict]:
        """运行单次实验（指定模型类型）"""
        print(f"\n{'='*60}")
        print(f"🔬 运行实验 {run_id+1}/{self.config.num_runs} - 模型: {model_type}")
        print(f"{'='*60}")
        
        try:
            # 初始化实验环境
            wrapper = DeCartExperimentWrapper(N=self.config.N, n=self.config.n)
            setup_time = wrapper.setup()
            setup_auxiliary_sizes = wrapper.get_auxiliary_sizes()
            wrapper.reset_metrics()
            
            # 定义用户ID
            owner_id = 5
            querier_id = 6
            
            # 生成访问策略
            policy = list(range(min(self.config.policy_size, self.config.N - 2)))
            policy.append(owner_id)
            policy.append(querier_id)
            policy = list(set(policy))  # 去重
            
            # 注册所有用户
            self.register_all_users(wrapper, policy)
            register_auxiliary_sizes = wrapper.get_auxiliary_sizes()
            
            # 建立信任关系
            wrapper.curator.add_trust(querier_id, owner_id)
            
            # 生成测试数据
            data, _ = self.generate_test_data()
            
            # 生成模型
            model = self.generate_model(model_type)
            
            # 加密数据集
            print(f"\n📦 加密数据集...")
            C_m, sk_h_s, ds_id = wrapper.encrypt_dataset(owner_id, data, policy)
            
            # 存储数据集
            wrapper.store_dataset(owner_id, ds_id, C_m, sk_h_s)
            
            # 根据不同模型类型执行查询
            if model_type == 'dot':
                # 点积模型
                print(f"\n🔍 执行点积查询...")
                results = wrapper.execute_query(querier_id, owner_id, ds_id, model)
                metrics = wrapper.get_metrics()
                
                query_time = metrics['query_times'][-1] if metrics['query_times'] else 0
                decrypt_time = metrics['decrypt_times'][-1] if metrics['decrypt_times'] else 0
                
            elif model_type == 'neural_network':
                # 神经网络模型
                print(f"\n🧠 执行神经网络查询...")
                results = wrapper.execute_query(querier_id, owner_id, ds_id, model)
                metrics = wrapper.get_metrics()
                
                query_time = metrics['query_times'][-1] if metrics['query_times'] else 0
                decrypt_time = metrics['decrypt_times'][-1] if metrics['decrypt_times'] else 0
                
            elif model_type == 'decision_tree':
                # 决策树特殊处理
                print(f"\n🌲 执行决策树查询...")
                
                # 获取公钥
                pk_h = wrapper.curator.system.he.public_key
                
                # 加密决策树
                if hasattr(wrapper.curator.system, 'encrypt_decision_tree'):
                    print(f"   使用系统方法加密决策树...")
                    encrypted_model = wrapper.curator.system.encrypt_decision_tree(model, pk_h)
                else:
                    # 简化的加密
                    print(f"   使用简化方法加密决策树...")
                    # 获取决策树参数
                    if hasattr(model, 'get_encryptable_params'):
                        params = model.get_encryptable_params()
                        # 加密内部节点
                        encrypted_internal = []
                        for node in params.get('internal_nodes', []):
                            encrypted_node = {
                                'node_id': node['node_id'],
                                'feature_idx': node['feature_idx'],
                                'threshold': wrapper.curator.system.he.encrypt([node['threshold']]),
                                'left': node['left'],
                                'right': node['right']
                            }
                            encrypted_internal.append(encrypted_node)
                        
                        # 加密叶子节点
                        encrypted_leaves = []
                        for node in params.get('leaf_nodes', []):
                            encrypted_node = {
                                'node_id': node['node_id'],
                                'value': wrapper.curator.system.he.encrypt([node['value']])
                            }
                            encrypted_leaves.append(encrypted_node)
                        
                        encrypted_model = {
                            'type': 'decision_tree',
                            'internal_nodes': encrypted_internal,
                            'leaf_nodes': encrypted_leaves,
                            'root_id': params.get('root_id', 0),
                            'node_count': params.get('node_count', 0)
                        }
                        print(f"     加密完成: {len(encrypted_internal)}内部节点, {len(encrypted_leaves)}叶子节点")
                    else:
                        # 使用字典格式
                        encrypted_model = {
                            'type': 'decision_tree',
                            'encrypted': True,
                            'nodes': model.get('nodes', []) if isinstance(model, dict) else []
                        }
                        print(f"     使用字典格式")
                
                # 检查权限
                querier = wrapper.create_querier(querier_id)
                C_M = querier.check_access(C_m)
                
                if C_M is None:
                    print(f"      决策树访问检查失败")
                    return None
                
                # 设置加密模型
                C_M['encrypted_model'] = encrypted_model
                C_M['model_type'] = 'decision_tree'
                C_M['access_granted'] = True

                # 记录查询请求阶段通信量（统一口径）
                req_payload = {
                    'querier_id': querier_id,
                    'owner_id': owner_id,
                    'dataset_id': ds_id,
                    'encrypted_model': C_M.get('encrypted_model'),
                    'model_type': C_M.get('model_type')
                }
                req_size = wrapper._safe_obj_size(req_payload)
                wrapper.metrics['communication_sizes'].append({
                    'type': 'query',
                    'size': req_size,
                    'records': len(C_m.get('c6_i', [])) if isinstance(C_m, dict) else 0
                })
                
                # 确保有 sk_h_u
                if 'sk_h_u' not in C_M:
                    C_M['sk_h_u'] = b'demo_secret_key'
                
                # 执行查询
                print(f"   执行加密决策树查询...")
                start_query = time.perf_counter()
                
                # 调用系统 query 方法
                ER = wrapper.curator.system.query(C_M, C_m, sk_h_s)
                
                query_time = time.perf_counter() - start_query
                print(f"   查询时间: {query_time*1000:.2f} ms")
                
                # 解密结果
                if ER is not None:
                    # 记录返回包阶段通信量（决策树分支）
                    res_size = wrapper._safe_obj_size(ER)
                    wrapper.metrics['communication_sizes'].append({
                        'type': 'decrypt',
                        'size': res_size,
                        'records': len(C_m.get('c6_i', [])) if isinstance(C_m, dict) else 0
                    })

                    start_decrypt = time.perf_counter()
                    results = wrapper.curator.system.decrypt(C_M['sk_h_u'], ER)
                    decrypt_time = time.perf_counter() - start_decrypt
                    print(f"   解密时间: {decrypt_time*1000:.2f} ms")
                    print(f"   结果数量: {len(results) if results else 0}")
                    if results:
                        print(f"   结果示例: {results[:3]}")
                else:
                    results = None
                    decrypt_time = 0
                    print(f"      查询失败，返回空结果")
            
            else:
                print(f"      未知模型类型: {model_type}")
                return None
            
            # 收集该模型类型的指标
            phase_comm = {'upload': 0, 'query': 0, 'decrypt': 0}
            for c in wrapper.metrics['communication_sizes']:
                if isinstance(c, dict):
                    t = c.get('type')
                    s = c.get('size', 0)
                    if t == 'encrypt':
                        phase_comm['upload'] += s
                    elif t == 'query':
                        phase_comm['query'] += s
                    elif t == 'decrypt':
                        phase_comm['decrypt'] += s

            final_auxiliary_sizes = wrapper.get_auxiliary_sizes()

            model_metrics = {
                'setup_time': setup_time,
                'encrypt_times': wrapper.metrics['encrypt_times'].copy(),
                'query_time': query_time,
                'decrypt_time': decrypt_time,
                'setup_auxiliary_sizes': setup_auxiliary_sizes.copy(),
                'register_auxiliary_sizes': register_auxiliary_sizes.copy(),
                'final_auxiliary_sizes': final_auxiliary_sizes.copy(),
                'communication_sizes': [s.copy() if isinstance(s, dict) else s 
                                       for s in wrapper.metrics['communication_sizes']],
                'comm_upload_size': phase_comm['upload'],
                'comm_query_size': phase_comm['query'],
                'comm_decrypt_size': phase_comm['decrypt'],
                'success': results is not None,
                'num_results': len(results) if results else 0
            }
            
            if results:
                print(f"\n   查询成功! 结果数量: {len(results)}")
            
            return model_metrics
            
        except Exception as e:
            print(f"\n   实验运行失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self) -> Dict:
        """运行完整实验（测试所有模型类型）"""
        print("\n" + "="*80)
        print("🚀 开始 DeCart 方案多模型实验")
        print("="*80)
        print(f"\n📋 实验配置:")
        for key, value in asdict(self.config).items():
            print(f"   {key}: {value}")
        
        # 对每种模型类型运行实验
        for model_type in self.config.model_types:
            print(f"\n{'#'*70}")
            print(f"📊 测试模型类型: {model_type}")
            print(f"{'#'*70}")
            
            model_results = self.results['models'][model_type]
            
            for i in range(self.config.num_runs):
                run_result = self.run_single_experiment(i, model_type)
                
                if run_result:
                    # 累加结果
                    model_results['setup_times'].append(run_result.get('setup_time', 0))
                    model_results['encrypt_times'].extend(run_result['encrypt_times'])
                    model_results['query_times'].append(run_result['query_time'])
                    model_results['decrypt_times'].append(run_result['decrypt_time'])
                    model_results['comm_upload_sizes'].append(run_result.get('comm_upload_size', 0))
                    model_results['comm_query_sizes'].append(run_result.get('comm_query_size', 0))
                    model_results['comm_decrypt_sizes'].append(run_result.get('comm_decrypt_size', 0))

                    setup_auxiliary_sizes = run_result.get('setup_auxiliary_sizes', {})
                    register_auxiliary_sizes = run_result.get('register_auxiliary_sizes', {})
                    final_auxiliary_sizes = run_result.get('final_auxiliary_sizes', {})
                    model_results['setup_crs_sizes'].append(setup_auxiliary_sizes.get('crs_size_bytes', 0))
                    model_results['setup_pp_sizes'].append(setup_auxiliary_sizes.get('pp_size_bytes', 0))
                    model_results['setup_aux_sizes'].append(setup_auxiliary_sizes.get('aux_size_bytes', 0))
                    model_results['setup_total_auxiliary_sizes'].append(setup_auxiliary_sizes.get('total_auxiliary_size_bytes', 0))
                    model_results['register_crs_sizes'].append(register_auxiliary_sizes.get('crs_size_bytes', 0))
                    model_results['register_pp_sizes'].append(register_auxiliary_sizes.get('pp_size_bytes', 0))
                    model_results['register_aux_sizes'].append(register_auxiliary_sizes.get('aux_size_bytes', 0))
                    model_results['register_total_auxiliary_sizes'].append(register_auxiliary_sizes.get('total_auxiliary_size_bytes', 0))
                    model_results['final_crs_sizes'].append(final_auxiliary_sizes.get('crs_size_bytes', 0))
                    model_results['final_pp_sizes'].append(final_auxiliary_sizes.get('pp_size_bytes', 0))
                    model_results['final_aux_sizes'].append(final_auxiliary_sizes.get('aux_size_bytes', 0))
                    model_results['final_total_auxiliary_sizes'].append(final_auxiliary_sizes.get('total_auxiliary_size_bytes', 0))
                    
                    for comm in run_result['communication_sizes']:
                        if isinstance(comm, dict):
                            model_results['communication_sizes'].append(comm.get('size', 0))
                        else:
                            model_results['communication_sizes'].append(comm)
                    
                    model_results['runs'].append({
                        'run_id': i,
                        'query_time': run_result['query_time'],
                        'decrypt_time': run_result['decrypt_time'],
                        'setup_auxiliary_sizes': setup_auxiliary_sizes,
                        'register_auxiliary_sizes': register_auxiliary_sizes,
                        'final_auxiliary_sizes': final_auxiliary_sizes,
                        'success': run_result['success'],
                        'num_results': run_result['num_results']
                    })
                    
                    print(f"\n      运行 {i+1} 完成: 查询时间 {run_result['query_time']*1000:.2f} ms")
                else:
                    print(f"\n      运行 {i+1} 失败")
        
        # 计算统计
        self._compute_statistics()
        
        # 保存结果
        if self.config.save_results:
            self.save_results()
        
        return self.results
    
    def _compute_statistics(self):
        """计算统计指标（按模型类型）"""
        summary = {}
        
        for model_type, model_data in self.results['models'].items():
            stats = {}

            # 初始化时间
            if model_data['setup_times']:
                times = model_data['setup_times']
                stats['avg_setup_time'] = float(np.mean(times))
                stats['std_setup_time'] = float(np.std(times))
                stats['min_setup_time'] = float(np.min(times))
                stats['max_setup_time'] = float(np.max(times))
            
            # 加密时间
            if model_data['encrypt_times']:
                times = model_data['encrypt_times']
                stats['avg_encrypt_time'] = float(np.mean(times))
                stats['std_encrypt_time'] = float(np.std(times))
                stats['min_encrypt_time'] = float(np.min(times))
                stats['max_encrypt_time'] = float(np.max(times))
            
            # 查询时间
            if model_data['query_times']:
                times = model_data['query_times']
                stats['avg_query_time'] = float(np.mean(times))
                stats['std_query_time'] = float(np.std(times))
                stats['min_query_time'] = float(np.min(times))
                stats['max_query_time'] = float(np.max(times))
            
            # 解密时间
            if model_data['decrypt_times']:
                times = model_data['decrypt_times']
                stats['avg_decrypt_time'] = float(np.mean(times))
                stats['std_decrypt_time'] = float(np.std(times))
                stats['min_decrypt_time'] = float(np.min(times))
                stats['max_decrypt_time'] = float(np.max(times))
            
            # 通信开销
            if model_data['communication_sizes']:
                sizes = model_data['communication_sizes']
                stats['avg_communication_size'] = float(np.mean(sizes)) / 1024  # KB
                stats['std_communication_size'] = float(np.std(sizes)) / 1024
                stats['total_communication'] = float(np.sum(sizes)) / 1024  # KB

            auxiliary_size_fields = {
                'setup_crs_sizes': 'avg_setup_crs_size_kb',
                'setup_pp_sizes': 'avg_setup_pp_size_kb',
                'setup_aux_sizes': 'avg_setup_aux_size_kb',
                'setup_total_auxiliary_sizes': 'avg_setup_total_auxiliary_size_kb',
                'register_crs_sizes': 'avg_register_crs_size_kb',
                'register_pp_sizes': 'avg_register_pp_size_kb',
                'register_aux_sizes': 'avg_register_aux_size_kb',
                'register_total_auxiliary_sizes': 'avg_register_total_auxiliary_size_kb',
                'final_crs_sizes': 'avg_final_crs_size_kb',
                'final_pp_sizes': 'avg_final_pp_size_kb',
                'final_aux_sizes': 'avg_final_aux_size_kb',
                'final_total_auxiliary_sizes': 'avg_final_total_auxiliary_size_kb'
            }
            for field_name, summary_name in auxiliary_size_fields.items():
                if model_data[field_name]:
                    stats[summary_name] = float(np.mean(model_data[field_name])) / 1024
            
            summary[model_type] = stats
        
        self.results['summary'] = summary
        
        # 打印结果
        print("\n" + "="*80)
        print("📊 多模型实验结果统计")
        print("="*80)
        
        for model_type, stats in summary.items():
            print(f"\n▶ 模型类型: {model_type}")
            for key, value in stats.items():
                if 'time' in key:
                    print(f"   {key}: {value*1000:.2f} ms")
                elif 'size' in key:
                    print(f"   {key}: {value:.2f} KB")
                else:
                    print(f"   {key}: {value}")
    
    def save_results(self):
        """保存实验结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_filename = f"decart_multimodel_exp_{timestamp}.json"
        json_path = os.path.join(self.config.results_dir, json_filename)
        
        def convert_to_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json(v) for v in obj]
            else:
                return obj
        
        json_results = convert_to_json(self.results)
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n   多模型实验结果已保存:")
        print(f"   JSON: {json_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeCart 方案多模型实验")
    parser.add_argument("--N", type=int, default=Config.MAX_USERS, help="最大用户数")
    parser.add_argument("--n", type=int, default=Config.BLOCK_SIZE, help="每块用户数")
    parser.add_argument("--num-records", type=int, default=Config.EXPERIMENT_NUM_RECORDS, help="数据记录数")
    parser.add_argument("--record-dim", type=int, default=Config.EXPERIMENT_RECORD_DIM, help="记录维度")
    parser.add_argument("--policy-size", type=int, default=Config.EXPERIMENT_POLICY_SIZE, help="策略大小")
    parser.add_argument("--num-runs", type=int, default=Config.EXPERIMENT_NUM_RUNS, help="重复运行次数")
    parser.add_argument("--model-types", nargs="+", 
                       default=['dot', 'decision_tree', 'neural_network'],
                       help="模型类型列表")
    parser.add_argument("--no-save", action="store_true", help="不保存结果")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"🚀 启动 DeCart 多模型实验 (规模: N={args.N}, n={args.n})")
    print("="*80)
    print(f"\n📋 命令行参数:")
    print(f"   N: {args.N}")
    print(f"   n: {args.n}")
    print(f"   num-records: {args.num_records}")
    print(f"   record-dim: {args.record_dim}")
    print(f"   policy-size: {args.policy_size}")
    print(f"   num-runs: {args.num_runs}")
    print(f"   model-types: {args.model_types}")
    print(f"   save-results: {not args.no_save}")
    
    config = ExperimentConfig(
        N=args.N,
        n=args.n,
        num_records=args.num_records,
        record_dim=args.record_dim,
        policy_size=args.policy_size,
        model_types=args.model_types,
        num_runs=args.num_runs,
        save_results=not args.no_save,
        results_dir="experiments/results/our_decart"
    )
    
    # 直接运行
    runner = ExperimentRunner(config)
    results = runner.run()
    
    print("\n" + "="*80)
    print(" 实验完成")
    print("="*80)