# DeCart / DeCart* 实验项目

本项目实现了 DeCart 与 DeCart* 两种方案，并提供多模型实验与对比基准测试脚本。

## 1. 环境要求

- 操作系统: Windows / Linux / macOS
- Python: 建议 3.10 - 3.11
- 建议使用虚拟环境

## 2. 项目依赖

### 2.1 核心依赖

以下依赖来自 requirements.txt:

- blspy==2.0.3
- bn256>=0.5.0
- cryptography>=42.0.0
- numpy>=1.24.0
- phe>=1.5.0
- pycryptodome>=3.19.0
- sympy>=1.12.0
- tenseal>=0.3.14
- loguru>=0.7.0

### 2.2 实验可选依赖

部分实验脚本还会用到以下库:

- matplotlib
- torch
- torchvision

如果你要运行 experiments/compare 或 MNIST 相关训练，建议安装这些可选库。

## 3. 安装步骤

在项目根目录执行:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install matplotlib torch torchvision
```

## 4. 快速开始

### 4.1 运行方案对比测试

```bash
python tests/test_schemes_comparison.py
```

### 4.2 运行 DeCart 多模型实验

```bash
python -m experiments.our_decart.runner --num-records 32 --record-dim 32 --policy-size 8 --num-runs 3
```

### 4.3 运行 DeCart* 多模型实验

```bash
python -m experiments.our_decart_star.runner --num-records 32 --record-dim 32 --policy-size 8 --num-runs 3
```

说明:

- 默认模型类型为: dot decision_tree neural_network
- 结果默认保存到 experiments/results/ 对应子目录
- 如不想保存结果，添加参数 --no-save

示例:

```bash
python -m experiments.our_decart.runner --model-types dot neural_network --no-save
```

## 5. 训练或生成模型文件

```bash
python -m experiments.models.train_models
```

输出目录:

- experiments/models/trained/

## 6. 通信与存储开销基准

### 6.1 完整通信开销

```bash
python -m experiments.compare.communication_benchmark --max-N 256
```

### 6.2 用户存储开销

```bash
python -m experiments.compare.user_storage_benchmark --max-N 512
```

## 7. 数据与结果文件说明

### 7.1 MNIST 数据

- 代码支持自动下载 MNIST 原始数据
- 通常不需要将 data/MNIST/raw 上传到仓库

### 7.2 实验结果与模型文件

当前仓库允许上传:

- experiments/results/
- experiments/models/trained/*.pkl

这样做的好处:

- 克隆后可直接查看已有实验结果
- 同时保留重新运行能力

## 8. 让项目在新机器可跑的建议流程

1. 克隆仓库
2. 创建并激活虚拟环境
3. 安装依赖
4. 先运行 tests/test_schemes_comparison.py 做快速健康检查
5. 再运行 experiments.our_decart 或 experiments.our_decart_star 的 runner

## 9. 常见问题

### Q1: 缺少某些库怎么办?

先执行:

```bash
pip install -r requirements.txt
```

如果仍报错，再按报错补装，例如:

```bash
pip install matplotlib torch torchvision
```

### Q2: 结果目录没有文件?

确认没有使用 --no-save 参数，且脚本运行完成。

### Q3: 运行较慢?

先减小参数:

- --num-records 16
- --record-dim 16
- --num-runs 1



