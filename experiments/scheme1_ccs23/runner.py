# decart/experiments/scheme1_ccs23/runner.py
"""
CCS23 方案实验运行器
作为性能上限基准（中心化明文计算）
"""

import sys
import os
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.scheme1_ccs23.wrapper import CCS23ExperimentWrapper


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 系统参数
    N: int = 64  # 最大用户数（接口兼容）
    n: int = 16  # 每块用户数（接口兼容）
    
    # 数据参数
    num_records: int = 64  # 数据记录数
    record_dim: int = 64   # 记录维度
    
    # 模型参数
    model_types: List[str] = None  # 要测试的模型类型
    
    # 策略参数
    policy_size: int = 10  # 访问策略中的用户数
    
    # 实验参数
    num_runs: int = 5      # 重复运行次数
    save_results: bool = True
    results_dir: str = "experiments/results/scheme1_ccs23"
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['dot', 'decision_tree', 'neural_network']


class ExperimentRunner:
    """CCS23 实验运行器（作为性能上限基准）"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {
            'config': asdict(config),
            'models': {},
            'summary': {}
        }
        
        # 为每种模型类型创建结果容器
        for model_type in config.model_types:
            self.results['models'][model_type] = {
                'encrypt_times': [],
                'query_times': [],
                'decrypt_times': [],
                'communication_sizes': [],
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
        """根据模型类型生成模型"""
        
        if model_type == 'dot':
            # 点积模型
            print(f"   生成点积模型...")
            model = np.random.randn(self.config.record_dim).tolist()
            max_val = max(abs(min(model)), abs(max(model))) or 1
            model = [x / max_val for x in model]
            return model
        
        elif model_type == 'decision_tree':
            # 决策树模型（CCS23 中作为基准，简化处理）
            print(f"   生成决策树模型...")
            return {
                'type': 'decision_tree',
                'root': 0,
                'nodes': [
                    {'id': 0, 'feature': 0, 'threshold': 0.5, 'left': 1, 'right': 2},
                    {'id': 1, 'value': 0.0},
                    {'id': 2, 'value': 1.0}
                ]
            }
        
        elif model_type == 'neural_network':
            # 神经网络模型
            print(f"   生成神经网络模型...")
            
            output_dim = 10
            input_dim = self.config.record_dim
            
            # 生成权重矩阵
            weights_matrix = np.random.randn(output_dim, input_dim) * 0.1
            bias = np.random.randn(output_dim) * 0.1
            weights = weights_matrix.flatten().tolist()
            
            # 限制权重数量
            max_weights = 500
            if len(weights) > max_weights:
                weights = weights[:max_weights]
                print(f"     权重数量: {len(weights)} (已截断)")
            
            return {
                'type': 'neural_network',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'weights': weights,
                'bias': bias.tolist()
            }
        
        else:
            raise ValueError(f"未知模型类型: {model_type}")
    
    def register_all_users(self, wrapper: CCS23ExperimentWrapper, policy: List[int]):
        """注册所有需要的用户（CCS23 中只是记录）"""
        print(f"\n 注册用户...")
        for uid in policy:
            wrapper.register_user(uid)
            print(f"   用户 {uid} 注册成功")
    
    def run_single_experiment(self, run_id: int, model_type: str) -> Optional[Dict]:
        """运行单次实验"""
        print(f"\n{'='*60}")
        print(f" [CCS23] 运行实验 {run_id+1}/{self.config.num_runs} - 模型: {model_type}")
        print(f"{'='*60}")
        
        try:
            # 初始化实验环境
            wrapper = CCS23ExperimentWrapper(N=self.config.N, n=self.config.n)
            wrapper.setup()
            wrapper.reset_metrics()
            
            # 定义用户ID
            owner_id = 5
            querier_id = 6
            
            # 生成访问策略
            policy = list(range(min(self.config.policy_size, self.config.N - 2)))
            policy.append(owner_id)
            policy.append(querier_id)
            policy = list(set(policy))
            
            # 注册所有用户
            self.register_all_users(wrapper, policy)
            
            # 生成测试数据
            data, _ = self.generate_test_data()
            
            # 生成模型
            model = self.generate_model(model_type)
            
            # "加密"数据集（实际存储明文）
            print(f"\n 存储数据...")
            C_m, sk_h_s, ds_id = wrapper.encrypt_dataset(owner_id, data, policy)
            
            # 存储数据集（CCS23 中不需要额外操作）
            wrapper.store_dataset(owner_id, ds_id, C_m, sk_h_s)
            
            # 执行查询
            print(f"\n 执行查询...")
            results = wrapper.execute_query(querier_id, owner_id, ds_id, model)
            
            # 收集指标
            metrics = wrapper.get_metrics()
            
            query_time = metrics['query_times'][-1] if metrics['query_times'] else 0
            decrypt_time = metrics['decrypt_times'][-1] if metrics['decrypt_times'] else 0
            
            # 收集该模型类型的指标
            model_metrics = {
                'encrypt_times': wrapper.metrics['encrypt_times'].copy(),
                'query_time': query_time,
                'decrypt_time': decrypt_time,
                'communication_sizes': [s.copy() if isinstance(s, dict) else s 
                                       for s in wrapper.metrics['communication_sizes']],
                'success': results is not None,
                'num_results': len(results) if results else 0
            }
            
            if results:
                print(f"\n 查询成功! 结果数量: {len(results)}")
                print(f"   结果示例: {results[:3]}")
            
            return model_metrics
            
        except Exception as e:
            print(f"\n   实验运行失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self) -> Dict:
        """运行完整实验"""
        print("\n" + "="*80)
        print(" 开始 CCS23 方案实验 (性能上限基准)")
        print("="*80)
        print(f"\n 实验配置:")
        for key, value in asdict(self.config).items():
            print(f"   {key}: {value}")
        
        # 对每种模型类型运行实验
        for model_type in self.config.model_types:
            print(f"\n{'#'*70}")
            print(f" 测试模型类型: {model_type}")
            print(f"{'#'*70}")
            
            model_results = self.results['models'][model_type]
            
            for i in range(self.config.num_runs):
                run_result = self.run_single_experiment(i, model_type)
                
                if run_result:
                    # 累加结果
                    model_results['encrypt_times'].extend(run_result['encrypt_times'])
                    model_results['query_times'].append(run_result['query_time'])
                    model_results['decrypt_times'].append(run_result['decrypt_time'])
                    
                    for comm in run_result['communication_sizes']:
                        if isinstance(comm, dict):
                            model_results['communication_sizes'].append(comm.get('size', 0))
                        else:
                            model_results['communication_sizes'].append(comm)
                    
                    model_results['runs'].append({
                        'run_id': i,
                        'query_time': run_result['query_time'],
                        'decrypt_time': run_result['decrypt_time'],
                        'success': run_result['success'],
                        'num_results': run_result['num_results']
                    })
                    
                    print(f"\n    运行 {i+1} 完成: 查询时间 {run_result['query_time']*1000:.2f} ms")
                else:
                    print(f"\n    运行 {i+1} 失败")
        
        # 计算统计
        self._compute_statistics()
        
        # 保存结果
        if self.config.save_results:
            self.save_results()
        
        return self.results
    
    def _compute_statistics(self):
        """计算统计指标"""
        summary = {}
        
        for model_type, model_data in self.results['models'].items():
            stats = {}
            
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
            
            summary[model_type] = stats
        
        self.results['summary'] = summary
        
        # 打印结果
        print("\n" + "="*80)
        print(" CCS23 实验结果统计 ")
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
        filename = f"ccs23_exp_{timestamp}.pkl"
        filepath = os.path.join(self.config.results_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        # 保存JSON版本
        json_filename = f"ccs23_exp_{timestamp}.json"
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
        
        print(f"\n CCS23 实验结果已保存:")
        print(f"   Pickle: {filepath}")
        print(f"   JSON: {json_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CCS23 方案实验")
    parser.add_argument("--num-records", type=int, default=32, help="数据记录数")
    parser.add_argument("--record-dim", type=int, default=32, help="记录维度")
    parser.add_argument("--policy-size", type=int, default=8, help="策略大小")
    parser.add_argument("--num-runs", type=int, default=3, help="重复运行次数")
    parser.add_argument("--model-types", nargs="+", 
                       default=['dot', 'decision_tree', 'neural_network'],
                       help="模型类型列表")
    parser.add_argument("--no-save", action="store_true", help="不保存结果")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" 启动 CCS23 实验 (性能上限基准)")
    print("="*80)
    print(f"\n 命令行参数:")
    print(f"   num-records: {args.num_records}")
    print(f"   record-dim: {args.record_dim}")
    print(f"   policy-size: {args.policy_size}")
    print(f"   num-runs: {args.num_runs}")
    print(f"   model-types: {args.model_types}")
    print(f"   save-results: {not args.no_save}")
    
    config = ExperimentConfig(
        N=64,
        n=16,
        num_records=args.num_records,
        record_dim=args.record_dim,
        policy_size=args.policy_size,
        model_types=args.model_types,
        num_runs=args.num_runs,
        save_results=not args.no_save,
        results_dir="experiments/results/scheme1_ccs23"
    )
    
    runner = ExperimentRunner(config)
    results = runner.run()
    
    print("\n" + "="*80)
    print("  CCS23 实验完成")
    print("="*80)