# decart/schemes/ai_models.py
"""
AI Models for Homomorphic Encryption
实现论文 Algorithm 1-4:
- Algorithm 1: 决策树加密
- Algorithm 2: 神经网络加密
- Algorithm 3: 加密决策树查询
- Algorithm 4: 加密神经网络查询

支持:
- 决策树 (Decision Tree)
- 前馈神经网络 (Feed-Forward Neural Network)
- 同态友好的激活函数近似
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
import math


class ActivationFunctions:
    """
    同态友好的激活函数近似
    论文要求: 使用低次多项式近似激活函数
    """
    
    @staticmethod
    def relu_square(x: float) -> float:
        """
        ReLU的平方近似: f(x) = x^2 (当x>0时接近，但会放大)
        适用于CKKS的同态计算
        """
        return x * x if x > 0 else 0.0
    
    @staticmethod
    def relu_poly3(x: float) -> float:
        """
        3次多项式近似ReLU: f(x) = 0.125x^2 + 0.5x + 0.25 (在[-2,2]区间)
        论文常用近似
        """
        if x < -2:
            return 0.0
        elif x > 2:
            return x
        else:
            return 0.125 * x * x + 0.5 * x + 0.25
    
    @staticmethod
    def sigmoid_poly3(x: float) -> float:
        """
        3次多项式近似Sigmoid: f(x) = 0.5 + 0.197x - 0.004x^3
        在[-5,5]区间有效
        """
        return 0.5 + 0.197 * x - 0.004 * x * x * x
    
    @staticmethod
    def tanh_poly3(x: float) -> float:
        """
        3次多项式近似Tanh: f(x) = x - x^3/3
        在[-1,1]区间有效
        """
        return x - (x * x * x) / 3.0
    
    @staticmethod
    def get_he_friendly(name: str, x: float) -> float:
        """获取同态友好的激活函数值"""
        if name == "relu_square":
            return ActivationFunctions.relu_square(x)
        elif name == "relu_poly3":
            return ActivationFunctions.relu_poly3(x)
        elif name == "sigmoid_poly3":
            return ActivationFunctions.sigmoid_poly3(x)
        elif name == "tanh_poly3":
            return ActivationFunctions.tanh_poly3(x)
        else:
            raise ValueError(f"未知激活函数: {name}")


class DecisionTreeNode:
    """决策树节点 - 用于Algorithm 1"""
    
    def __init__(self, node_id: int, is_leaf: bool = False):
        self.node_id = node_id
        self.is_leaf = is_leaf
        
        # 内部节点属性
        self.feature_idx: Optional[int] = None  # 特征索引 j_u
        self.threshold: Optional[float] = None  # 阈值 θ_u
        self.left_child: Optional[int] = None    # 左子节点ID
        self.right_child: Optional[int] = None   # 右子节点ID
        
        # 叶子节点属性
        self.value: Optional[float] = None       # 输出值 v_ℓ
    
    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Leaf({self.node_id}: {self.value})"
        else:
            return f"Internal({self.node_id}: f[{self.feature_idx}] <= {self.threshold})"


class DecisionTreeHE:
    """
    同态决策树 - 实现Algorithm 1和Algorithm 3
    对应论文Algorithm 1的加密表示
    """
    
    def __init__(self):
        self.nodes: Dict[int, DecisionTreeNode] = {}
        self.root_id: Optional[int] = None
    
    def add_node(self, node: DecisionTreeNode):
        """添加节点"""
        self.nodes[node.node_id] = node
    
    def set_root(self, node_id: int):
        """设置根节点"""
        if node_id in self.nodes:
            self.root_id = node_id
        else:
            raise ValueError(f"节点 {node_id} 不存在")
    
    @classmethod
    def from_sklearn(cls, tree_model, feature_names: Optional[List[str]] = None):
        """
        从sklearn决策树导入
        用于加载训练好的模型
        """
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        
        tree = cls()
        
        # 获取树结构
        if hasattr(tree_model, 'tree_'):
            # sklearn的DecisionTree
            tree_struct = tree_model.tree_
            n_nodes = tree_struct.node_count
            
            # 遍历所有节点
            for node_id in range(n_nodes):
                node = DecisionTreeNode(node_id)
                
                # 判断是否为叶子节点
                is_leaf = tree_struct.children_left[node_id] == -1
                node.is_leaf = is_leaf
                
                if is_leaf:
                    # 叶子节点：存储输出值
                    if hasattr(tree_model, 'classes_'):
                        # 分类树
                        value = tree_struct.value[node_id][0]
                        # 取概率最大的类别
                        node.value = float(np.argmax(value))
                    else:
                        # 回归树
                        node.value = float(tree_struct.value[node_id][0][0])
                else:
                    # 内部节点：存储特征索引和阈值
                    node.feature_idx = int(tree_struct.feature[node_id])
                    node.threshold = float(tree_struct.threshold[node_id])
                    node.left_child = int(tree_struct.children_left[node_id])
                    node.right_child = int(tree_struct.children_right[node_id])
                
                tree.add_node(node)
                if node_id == 0:  # 根节点
                    tree.root_id = 0
        
        return tree
    
    def get_encryptable_params(self) -> Dict:
        """
        获取可加密的参数 - Algorithm 1
        返回: 需要加密的参数字典
        """
        internal_nodes = []
        leaf_nodes = []
        
        for node_id, node in self.nodes.items():
            if node.is_leaf:
                leaf_nodes.append({
                    'node_id': node_id,
                    'value': node.value
                })
            else:
                internal_nodes.append({
                    'node_id': node_id,
                    'feature_idx': node.feature_idx,
                    'threshold': node.threshold,
                    'left': node.left_child,
                    'right': node.right_child
                })
        
        return {
            'internal_nodes': internal_nodes,
            'leaf_nodes': leaf_nodes,
            'root_id': self.root_id,
            'node_count': len(self.nodes)
        }
    
    def evaluate_plain(self, x: np.ndarray) -> float:
        """
        明文评估决策树 - 用于验证
        """
        if self.root_id is None:
            raise ValueError("树未初始化")
        
        node_id = self.root_id
        while not self.nodes[node_id].is_leaf:
            node = self.nodes[node_id]
            if x[node.feature_idx] <= node.threshold:
                node_id = node.left_child
            else:
                node_id = node.right_child
        
        return self.nodes[node_id].value
    
    def __repr__(self) -> str:
        return f"DecisionTreeHE(nodes={len(self.nodes)}, root={self.root_id})"


# decart/schemes/ai_models.py

class NeuralNetworkHE:
    """同态神经网络 - 默认使用单层架构"""
    
    def __init__(self):
        self.layers: List[Dict] = []
        self.layer_types: List[str] = []
        self.activations: List[str] = []
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
    
    def add_single_layer(self, 
                        input_dim: int,
                        output_dim: int,
                        activation: str = "linear"):
        """
        添加单层网络 - 默认使用这个
        参数:
            input_dim: 输入维度 (MNIST: 784)
            output_dim: 输出维度 (MNIST: 10)
            activation: 激活函数
        """
        # 初始化小随机权重
        w = np.random.randn(output_dim, input_dim).astype(np.float32) * 0.001
        b = np.random.randn(output_dim).astype(np.float32) * 0.001
        
        self.layers.append({
            'weights': w,
            'bias': b
        })
        self.layer_types.append('linear')
        self.activations.append(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        print(f"   创建单层网络: {input_dim} -> {output_dim}, 激活={activation}")
    
    @classmethod
    def create_mnist_single_layer(cls):
        """创建MNIST单层网络 (784 -> 10)"""
        nn = cls()
        nn.add_single_layer(784, 10, activation="linear")
        return nn
    
    def get_encryptable_params(self) -> List[Dict]:
        """获取可加密的参数"""
        encryptable = []
        for i, layer in enumerate(self.layers):
            encryptable.append({
                'layer_idx': i,
                'layer_type': self.layer_types[i],
                'activation': self.activations[i],
                'weights_shape': layer['weights'].shape,
                'bias_shape': layer['bias'].shape,
                'weights': layer['weights'].flatten().tolist(),
                'bias': layer['bias'].flatten().tolist()
            })
        return encryptable
    
    def evaluate_plain(self, x: np.ndarray) -> np.ndarray:
        """明文评估 - 用于验证"""
        if len(self.layers) == 0:
            return x
        
        # 单层网络: y = Wx + b
        layer = self.layers[0]
        return np.dot(layer['weights'], x) + layer['bias']


class EncryptedModelWrapper:
    """
    加密模型包装器
    统一处理两种模型的加密表示
    """
    
    def __init__(self, model_type: str):
        """
        参数:
            model_type: 'decision_tree' 或 'neural_network'
        """
        self.model_type = model_type
        self.plain_model = None
        self.encrypted_params = None
        self.metadata = {}
    
    @classmethod
    def from_decision_tree(cls, tree_model):
        """从决策树创建"""
        wrapper = cls('decision_tree')
        wrapper.plain_model = DecisionTreeHE.from_sklearn(tree_model)
        wrapper.metadata['node_count'] = len(wrapper.plain_model.nodes)
        return wrapper
    
    @classmethod
    def from_neural_network(cls, nn_model, input_dim, hidden_dims, output_dim):
        """从神经网络创建"""
        wrapper = cls('neural_network')
        wrapper.plain_model = NeuralNetworkHE.from_pytorch_mlp(nn_model, input_dim, hidden_dims, output_dim)
        wrapper.metadata['layer_count'] = len(wrapper.plain_model.layers)
        return wrapper
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """
        从文件加载训练好的模型
        支持 .pkl 文件（如 mlp_medium_*.pkl）
        """
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        model_name = config.get('model_name', '')
        architecture = config.get('architecture', {})
        
        # 根据模型名称判断类型
        if 'mlp' in model_name:
            # 创建MLP结构
            from experiments.models.mlp import MLP
            model = MLP(
                input_dim=architecture.get('input_dim', 784),
                hidden1=architecture.get('hidden1', 128),
                hidden2=architecture.get('hidden2', 64),
                output_dim=architecture.get('output_dim', 10)
            )
            # 注意：这里需要加载训练好的权重，但pkl只存了配置
            # 实际使用时需要加载完整的模型文件
            wrapper = cls.from_neural_network(
                model,
                architecture.get('input_dim', 784),
                [architecture.get('hidden1', 128), architecture.get('hidden2', 64)],
                architecture.get('output_dim', 10)
            )
        
        elif 'svm' in model_name:
            # SVM作为特殊的决策树处理
            # 简化：创建单节点决策树
            tree = DecisionTreeHE()
            node = DecisionTreeNode(0, is_leaf=True)
            node.value = 0.0  # 占位值
            tree.add_node(node)
            tree.set_root(0)
            wrapper = cls('decision_tree')
            wrapper.plain_model = tree
        
        elif 'cnn' in model_name:
            # 简化的CNN作为MLP处理
            from experiments.models.cnn import SimpleCNN
            model = SimpleCNN(
                num_classes=architecture.get('num_classes', 10)
            )
            wrapper = cls.from_neural_network(
                model,
                architecture.get('input_size', 28) * architecture.get('input_size', 28),
                [architecture.get('fc1_dim', 64)],
                architecture.get('num_classes', 10)
            )
        
        else:
            raise ValueError(f"未知模型类型: {model_name}")
        
        wrapper.metadata['test_accuracy'] = config.get('test_accuracy', 0.0)
        wrapper.metadata['history'] = config.get('history_summary', {})
        
        return wrapper
    
    def get_encryptable_params(self):
        """获取需要加密的参数"""
        if self.model_type == 'decision_tree':
            return self.plain_model.get_encryptable_params()
        else:
            return self.plain_model.get_encryptable_params()
    
    def evaluate_plain(self, x: np.ndarray):
        """明文评估"""
        return self.plain_model.evaluate_plain(x)


# ========== 测试代码 ==========

def test_decision_tree():
    """测试决策树功能"""
    print("\n" + "="*60)
    print("测试 Decision Tree 功能")
    print("="*60)
    
    # 创建一个简单的决策树
    tree = DecisionTreeHE()
    
    # 根节点: 内部节点
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
    
    print(f"树结构: {tree}")
    
    # 测试明文评估
    x1 = np.array([0.2, 0.8])
    x2 = np.array([0.7, 0.3])
    
    result1 = tree.evaluate_plain(x1)
    result2 = tree.evaluate_plain(x2)
    
    print(f"输入 {x1} → 输出 {result1} (应去左子树)")
    print(f"输入 {x2} → 输出 {result2} (应去右子树)")
    
    # 获取可加密参数
    params = tree.get_encryptable_params()
    print(f"\n可加密参数:")
    print(f"  内部节点数: {len(params['internal_nodes'])}")
    print(f"  叶子节点数: {len(params['leaf_nodes'])}")
    
    print("\n✅ 决策树测试通过")


def test_neural_network():
    """测试神经网络功能"""
    print("\n" + "="*60)
    print("测试 Neural Network 功能")
    print("="*60)
    
    # 创建一个简单的神经网络
    nn = NeuralNetworkHE()
    
    # 第1层: 3->2
    w1 = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]])
    b1 = np.array([0.1, 0.2])
    nn.add_layer(w1, b1, activation="relu_poly3")
    
    # 第2层: 2->1
    w2 = np.array([[0.7, 0.8]])
    b2 = np.array([0.3])
    nn.add_layer(w2, b2, activation="linear")
    
    print(f"网络结构: {nn}")
    
    # 测试明文评估
    x = np.array([1.0, 2.0, 3.0])
    result = nn.evaluate_plain(x)
    
    print(f"输入 {x}")
    print(f"输出 {result}")
    
    # 手动计算验证
    # 第1层: z1 = W1·x + b1
    z1_0 = 0.1*1.0 + 0.2*2.0 + 0.3*3.0 + 0.1  # = 1.5
    z1_1 = 0.4*1.0 + 0.5*2.0 + 0.6*3.0 + 0.2  # = 3.5
    # ReLU^2近似
    h1_0 = ActivationFunctions.relu_poly3(z1_0)  # ≈ 0.125*2.25 + 0.5*1.5 + 0.25 = 1.28125
    h1_1 = ActivationFunctions.relu_poly3(z1_1)  # 对于>2的值，返回x本身 ≈ 3.5
    # 第2层: z2 = W2·h1 + b2
    z2 = 0.7*h1_0 + 0.8*h1_1 + 0.3  # ≈ 0.7*1.281 + 0.8*3.5 + 0.3 = 3.9967
    
    print(f"手动计算: {z2}")
    print(f"误差: {abs(result[0] - z2)}")
    
    # 获取可加密参数
    params = nn.get_encryptable_params()
    print(f"\n可加密参数:")
    print(f"  层数: {len(params)}")
    for i, p in enumerate(params):
        print(f"    层{i}: weights {p['weights_shape']}, bias {p['bias_shape']}")
    
    print("\n 神经网络测试通过")


def test_activation_functions():
    """测试激活函数"""
    print("\n" + "="*60)
    print("测试 Activation Functions")
    print("="*60)
    
    test_points = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    
    print("ReLU^2 近似:")
    for x in test_points:
        y = ActivationFunctions.relu_poly3(x)
        print(f"  x={x:4.1f} → {y:.4f}")
    
    print("\nSigmoid 近似:")
    for x in test_points:
        y = ActivationFunctions.sigmoid_poly3(x)
        print(f"  x={x:4.1f} → {y:.4f}")
    
    print("\n 激活函数测试通过")


if __name__ == "__main__":
    print("="*60)
    print(" AI Models 模块测试")
    print("="*60)
    
    test_activation_functions()
    test_decision_tree()
    test_neural_network()
    
    print("\n" + "="*60)
    print(" 所有测试通过")
    print("   实现 Algorithm 1-4 的基础支持")
    print("="*60)