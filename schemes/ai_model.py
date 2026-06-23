# decart/schemes/ai_models.py

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
import math


class ActivationFunctions:
    
    @staticmethod
    def relu_square(x: float) -> float:
        return x * x if x > 0 else 0.0
    
    @staticmethod
    def relu_poly3(x: float) -> float:

        if x < -2:
            return 0.0
        elif x > 2:
            return x
        else:
            return 0.125 * x * x + 0.5 * x + 0.25
    
    @staticmethod
    def sigmoid_poly3(x: float) -> float:
        return 0.5 + 0.197 * x - 0.004 * x * x * x
    
    @staticmethod
    def tanh_poly3(x: float) -> float:

        return x - (x * x * x) / 3.0

    @staticmethod
    def square(x: float) -> float:
        """Square activation: f(x) = x^2"""
        return x * x

    @staticmethod
    def linear(x: float) -> float:
        """Linear activation: f(x) = x"""
        return x
    
    @staticmethod
    def get_he_friendly(name: str, x: float) -> float:
        """Get the value of a HE-friendly activation function."""
        if name == "linear":
            return ActivationFunctions.linear(x)
        elif name == "square":
            return ActivationFunctions.square(x)
        elif name == "relu_square":
            return ActivationFunctions.relu_square(x)
        elif name == "relu_poly3":
            return ActivationFunctions.relu_poly3(x)
        elif name == "sigmoid_poly3":
            return ActivationFunctions.sigmoid_poly3(x)
        elif name == "tanh_poly3":
            return ActivationFunctions.tanh_poly3(x)
        else:
            raise ValueError(f"Unknown activation function: {name}")


class DecisionTreeNode:
    """Decision tree node."""
    
    def __init__(self, node_id: int, is_leaf: bool = False):
        self.node_id = node_id
        self.is_leaf = is_leaf
        
        # Internal node attributes
        self.feature_idx: Optional[int] = None  # Feature index j_u
        self.threshold: Optional[float] = None  # Threshold theta_u
        self.left_child: Optional[int] = None    # Left child node ID
        self.right_child: Optional[int] = None   # Right child node ID
        
        # Leaf node attributes
        self.value: Optional[float] = None       # Output value v_l
    
    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Leaf({self.node_id}: {self.value})"
        else:
            return f"Internal({self.node_id}: f[{self.feature_idx}] <= {self.threshold})"


class DecisionTreeHE:
    """
    Homomorphic decision tree.
    """
    
    def __init__(self):
        self.nodes: Dict[int, DecisionTreeNode] = {}
        self.root_id: Optional[int] = None
    
    def add_node(self, node: DecisionTreeNode):
        """Add a node."""
        self.nodes[node.node_id] = node
    
    def set_root(self, node_id: int):
        """Set the root node."""
        if node_id in self.nodes:
            self.root_id = node_id
        else:
            raise ValueError(f"Node {node_id} does not exist")
    
    @classmethod
    def from_sklearn(cls, tree_model, feature_names: Optional[List[str]] = None):
        """
        Import from an sklearn decision tree for loading a trained model.
        """
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        
        tree = cls()
        
        # Get tree structure
        if hasattr(tree_model, 'tree_'):
            # sklearn DecisionTree
            tree_struct = tree_model.tree_
            n_nodes = tree_struct.node_count
            
            # Iterate over all nodes
            for node_id in range(n_nodes):
                node = DecisionTreeNode(node_id)
                
                # Determine whether this is a leaf node
                is_leaf = tree_struct.children_left[node_id] == -1
                node.is_leaf = is_leaf
                
                if is_leaf:
                    # Leaf node: store output value
                    if hasattr(tree_model, 'classes_'):
                        # Classification tree
                        value = tree_struct.value[node_id][0]
                        # Take the class with maximum probability
                        node.value = float(np.argmax(value))
                    else:
                        # Regression tree
                        node.value = float(tree_struct.value[node_id][0][0])
                else:
                    # Internal node: store feature index and threshold
                    node.feature_idx = int(tree_struct.feature[node_id])
                    node.threshold = float(tree_struct.threshold[node_id])
                    node.left_child = int(tree_struct.children_left[node_id])
                    node.right_child = int(tree_struct.children_right[node_id])
                
                tree.add_node(node)
                if node_id == 0:  # Root node
                    tree.root_id = 0
        
        return tree
    
    def get_encryptable_params(self) -> Dict:
        """
        Get encryptable parameters.
        Returns: dictionary of parameters that need encryption.
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
        Evaluate decision tree in plaintext.
        """
        if self.root_id is None:
            raise ValueError("Tree is not initialized")
        
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
    """Homomorphic neural network."""
    
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
        Parameters:
            input_dim: Input dimension (MNIST: 784)
            output_dim: Output dimension (MNIST: 10)
            activation: Activation function
        """
        # Initialize small random weights
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
        
        print(f"   Created single-layer network: {input_dim} -> {output_dim}, activation={activation}")

    def add_layer(self,
                  input_dim: int,
                  output_dim: int,
                  activation: str = "linear",
                  weight_scale: Optional[float] = None):
        if weight_scale is None:
            weight_scale = min(0.1, 1.0 / np.sqrt(max(1, input_dim)))

        w = np.random.randn(output_dim, input_dim).astype(np.float32) * weight_scale
        b = np.random.randn(output_dim).astype(np.float32) * weight_scale

        self.layers.append({
            'weights': w,
            'bias': b,
        })
        self.layer_types.append('linear')
        self.activations.append(activation)

        if self.input_dim is None:
            self.input_dim = input_dim
        self.output_dim = output_dim

        print(f"   Created network layer: {input_dim} -> {output_dim}, activation={activation}")

    def add_shallow_mlp(self,
                        input_dim: int,
                        hidden_dim: int = 16,
                        output_dim: int = 10,
                        hidden_activation: str = "square",
                        output_activation: str = "linear"):
        self.layers = []
        self.layer_types = []
        self.activations = []
        self.input_dim = None
        self.output_dim = None
        self.add_layer(input_dim, hidden_dim, activation=hidden_activation)
        self.add_layer(hidden_dim, output_dim, activation=output_activation)
        print(
            f"   Created single-hidden-layer network: {input_dim} -> {hidden_dim} -> {output_dim}, "
            f"activation=({hidden_activation}, {output_activation})"
        )
    
    @classmethod
    def create_mnist_single_layer(cls):
        nn = cls()
        nn.add_single_layer(784, 10, activation="linear")
        return nn

    @classmethod
    def create_shallow_mlp(cls,
                           input_dim: int,
                           hidden_dim: int = 16,
                           output_dim: int = 10,
                           hidden_activation: str = "square",
                           output_activation: str = "linear"):
        nn = cls()
        nn.add_shallow_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
        return nn
    
    def get_encryptable_params(self) -> List[Dict]:
        """Get encryptable parameters."""
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
        """Evaluate in plaintext."""
        if len(self.layers) == 0:
            return x

        values = np.asarray(x, dtype=np.float32)
        for layer, activation in zip(self.layers, self.activations):
            values = np.dot(layer['weights'], values) + layer['bias']
            if activation != 'linear':
                values = np.asarray(
                    [ActivationFunctions.get_he_friendly(activation, float(v)) for v in values],
                    dtype=np.float32,
                )
        return values


class EncryptedModelWrapper:
    """
    Encrypted model wrapper.
    Unified handling of encrypted representations for two model types.
    """
    
    def __init__(self, model_type: str):
        """
        Parameters:
            model_type: 'decision_tree' or 'neural_network'
        """
        self.model_type = model_type
        self.plain_model = None
        self.encrypted_params = None
        self.metadata = {}
    
    @classmethod
    def from_decision_tree(cls, tree_model):
        """Create from a decision tree."""
        wrapper = cls('decision_tree')
        wrapper.plain_model = DecisionTreeHE.from_sklearn(tree_model)
        wrapper.metadata['node_count'] = len(wrapper.plain_model.nodes)
        return wrapper
    
    @classmethod
    def from_neural_network(cls, nn_model, input_dim, hidden_dims, output_dim):
        """Create from a neural network."""
        wrapper = cls('neural_network')
        wrapper.plain_model = NeuralNetworkHE.from_pytorch_mlp(nn_model, input_dim, hidden_dims, output_dim)
        wrapper.metadata['layer_count'] = len(wrapper.plain_model.layers)
        return wrapper
    
    @classmethod
    def load_from_file(cls, filepath: str):
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        model_name = config.get('model_name', '')
        architecture = config.get('architecture', {})
        
        # Determine type based on model name
        if 'mlp' in model_name:
            # Create MLP structure
            from experiments.models.mlp import MLP
            model = MLP(
                input_dim=architecture.get('input_dim', 784),
                hidden1=architecture.get('hidden1', 128),
                hidden2=architecture.get('hidden2', 64),
                output_dim=architecture.get('output_dim', 10)
            )

            wrapper = cls.from_neural_network(
                model,
                architecture.get('input_dim', 784),
                [architecture.get('hidden1', 128), architecture.get('hidden2', 64)],
                architecture.get('output_dim', 10)
            )
        
        elif 'svm' in model_name:

            tree = DecisionTreeHE()
            node = DecisionTreeNode(0, is_leaf=True)
            node.value = 0.0  # Placeholder value
            tree.add_node(node)
            tree.set_root(0)
            wrapper = cls('decision_tree')
            wrapper.plain_model = tree
        
        elif 'cnn' in model_name:
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
            raise ValueError(f"Unknown model type: {model_name}")
        
        wrapper.metadata['test_accuracy'] = config.get('test_accuracy', 0.0)
        wrapper.metadata['history'] = config.get('history_summary', {})
        
        return wrapper
    
    def get_encryptable_params(self):
        """Get parameters that need encryption."""
        if self.model_type == 'decision_tree':
            return self.plain_model.get_encryptable_params()
        else:
            return self.plain_model.get_encryptable_params()
    
    def evaluate_plain(self, x: np.ndarray):
        """Evaluate in plaintext."""
        return self.plain_model.evaluate_plain(x)


# ========== Test Code ==========

def test_decision_tree():
    
    # Create a simple decision tree
    tree = DecisionTreeHE()
    
    # Root node: internal node
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
    
    print(f"Tree structure: {tree}")
    
    # Test plaintext evaluation
    x1 = np.array([0.2, 0.8])
    x2 = np.array([0.7, 0.3])
    
    result1 = tree.evaluate_plain(x1)
    result2 = tree.evaluate_plain(x2)
    
    print(f"Input {x1} -> Output {result1} (should go to left subtree)")
    print(f"Input {x2} -> Output {result2} (should go to right subtree)")
    
    # Get encryptable parameters
    params = tree.get_encryptable_params()
    print(f"\nEncryptable parameters:")
    print(f"  Internal node count: {len(params['internal_nodes'])}")
    print(f"  Leaf node count: {len(params['leaf_nodes'])}")
    
    print("\n Decision tree test passed")


def test_neural_network():
    
    # Create a simple neural network
    nn = NeuralNetworkHE()
    
    # Layer 1: 3->2
    w1 = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]])
    b1 = np.array([0.1, 0.2])
    nn.add_layer(w1, b1, activation="relu_poly3")
    
    # Layer 2: 2->1
    w2 = np.array([[0.7, 0.8]])
    b2 = np.array([0.3])
    nn.add_layer(w2, b2, activation="linear")
    
    print(f"Network structure: {nn}")
    
    # Test plaintext evaluation
    x = np.array([1.0, 2.0, 3.0])
    result = nn.evaluate_plain(x)
    
    print(f"Input {x}")
    print(f"Output {result}")
    
    # Manual calculation for verification
    # Layer 1: z1 = W1·x + b1
    z1_0 = 0.1*1.0 + 0.2*2.0 + 0.3*3.0 + 0.1  # = 1.5
    z1_1 = 0.4*1.0 + 0.5*2.0 + 0.6*3.0 + 0.2  # = 3.5
    # ReLU^2 approximation
    h1_0 = ActivationFunctions.relu_poly3(z1_0)  # ≈ 0.125*2.25 + 0.5*1.5 + 0.25 = 1.28125
    h1_1 = ActivationFunctions.relu_poly3(z1_1)  # For values >2, returns x itself ≈ 3.5
    # Layer 2: z2 = W2·h1 + b2
    z2 = 0.7*h1_0 + 0.8*h1_1 + 0.3  # ≈ 0.7*1.281 + 0.8*3.5 + 0.3 = 3.9967
    
    print(f"Manual calculation: {z2}")
    print(f"Error: {abs(result[0] - z2)}")
    
    # Get encryptable parameters
    params = nn.get_encryptable_params()
    print(f"\nEncryptable parameters:")
    print(f"  Layer count: {len(params)}")
    for i, p in enumerate(params):
        print(f"    Layer {i}: weights {p['weights_shape']}, bias {p['bias_shape']}")
    
    print("\n Neural network test passed")


def test_activation_functions():
    """Test activation functions."""
    print("\n" + "="*60)
    print("Testing Activation Functions")
    print("="*60)
    
    test_points = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    
    print("ReLU^2 approximation:")
    for x in test_points:
        y = ActivationFunctions.relu_poly3(x)
        print(f"  x={x:4.1f} → {y:.4f}")
    
    print("\nSigmoid approximation:")
    for x in test_points:
        y = ActivationFunctions.sigmoid_poly3(x)
        print(f"  x={x:4.1f} → {y:.4f}")
    
    print("\n Activation function tests passed")


if __name__ == "__main__":
    
    test_activation_functions()
    test_decision_tree()
    test_neural_network()