# DeCart / DeCart* Experimental Repository

This repository includes implementations of DeCart, DeCart*, and multiple baseline schemes (CCS23/Server/Offline/SecPQ), along with experiment runners and paper plotting scripts.

## 1. Repository Overview

- `schemes/`: Core scheme implementations (DeCart, DeCart*, AI model abstraction)
- `entities/`: Participant entities (owner, querier, server, curator)
- `core/`: Finite field, bilinear pairing, homomorphic encryption, and other core modules
- `experiments/`: Runners for each scheme, dataset loaders, model training, and comparison plotting
- `tests/`: Basic functional and scheme comparison tests
- `data/`: MNIST/UCI HAR dataset directory

## 2. Environment Requirements

- Python 3.10 or 3.11 (recommended)
- Windows / Linux / macOS
- A virtual environment is recommended

## 3. Install Dependencies

Run in the project root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy sympy bn256 tenseal phe pycryptodome cryptography loguru matplotlib torch torchvision scikit-learn scipy
```

Notes:

- If you only run core schemes (no training or plotting), you may skip `torch/torchvision/matplotlib/scikit-learn/scipy`.
- `tenseal` may require a compatible build environment. Check Python and platform compatibility if installation fails.

## 4. Quick Start

### 4.1 Minimal Health Check

```bash
python tests/test_schemes_comparison.py
```

### 4.2 Run DeCart

```bash
python -m experiments.our_decart.runner --N 10000 --n 32 --num-records 32 --record-dim 32 --policy-size 32 --num-runs 3
```

### 4.3 Run DeCart*

```bash
python -m experiments.our_decart_star.runner --N 10000 --n 32 --num-records 32 --record-dim 32 --policy-size 32 --num-runs 3
```

Common arguments:

- `--dataset {synthetic,mnist,uci_har}`
- `--model-source {synthetic,trained}`
- `--model-types dot decision_tree neural_network`
- `--no-save` (do not persist results)
- `--results-dir <path>` (custom output directory)

## 5. Dataset and Dimension Constraints

- `synthetic`: `record_dim` is configurable
- `mnist`: `record_dim` must be `784`
- `uci_har`: `record_dim` must be `561`

It is recommended to start with the synthetic dataset.

```bash
python -m experiments.our_decart.runner --dataset synthetic --record-dim 32 --num-records 32 --policy-size 32 --num-runs 3
python -m experiments.our_decart_star.runner --dataset synthetic --record-dim 32 --num-records 32 --policy-size 32 --num-runs 3
```

To switch to real datasets:

```bash
python -m experiments.our_decart.runner --dataset mnist --record-dim 784 --num-records 128 --mnist-data-dir data
python -m experiments.our_decart_star.runner --dataset uci_har --record-dim 561 --num-records 128 --mnist-data-dir data
```

## 6. Train Real Models

```bash
python -m experiments.models.train_models --dataset mnist --data-dir data
python -m experiments.models.train_models --dataset uci_har --data-dir data
```

Trained models are saved to:

- `experiments/models/trained/`

Use them with the runner:

```bash
python -m experiments.our_decart.runner --dataset mnist --record-dim 784 --model-source trained --trained-models-dir experiments/models/trained
```

## 7. Baseline Comparison Workflow

For fair comparison, use a unified parameter set across schemes:

- `N=10000`
- `n=32`
- `num_records=32`
- `record_dim=32`
- `policy_size=32`
- `num_runs=3`
- `dataset=synthetic`

### 7.1 Run DeCart and DeCart* First

```bash
python -m experiments.our_decart.runner --dataset synthetic --N 10000 --n 32 --num-records 32 --record-dim 32 --policy-size 32 --num-runs 3
python -m experiments.our_decart_star.runner --dataset synthetic --N 10000 --n 32 --num-records 32 --record-dim 32 --policy-size 32 --num-runs 3
```

### 7.2 Run Baseline Schemes

```bash
python -m experiments.scheme1_ccs23.runner --dataset synthetic --N 10000 --n 32 --num-records 32 --record-dim 32 --policy-size 32 --num-runs 3
python -m experiments.scheme2_server.runner --dataset synthetic --N 10000 --n 32 --num-records 32 --record-dim 32 --policy-size 32 --num-runs 3
python -m experiments.scheme3_offline.runner --dataset synthetic --N 10000 --n 32 --num-records 32 --record-dim 32 --policy-size 32 --num-runs 3
python -m experiments.secpq.runner --dataset synthetic --N 10000 --n 32 --num-records 32 --record-dim 32 --policy-size 32 --num-runs 3 --model-types decision_tree
```

## 8. Revoke Experiments

```bash
python -m experiments.revoke.runner --scheme decart --model-type decision_tree --N 10000 --n 32 --num-records 10 --record-dim 10 --policy-size 32 --num-runs 3
python -m experiments.revoke.runner --scheme decart_star --model-type decision_tree --N 10000 --n 32 --num-records 10 --record-dim 10 --policy-size 32 --num-runs 3
```

## 9. Result Output Directories

By default, experiment results are stored in scheme-specific subdirectories under `experiments/results/` (or override with `--results-dir`).

Typical directories:

- `experiments/results/our_decart/`
- `experiments/results/our_decart_star/`
- `experiments/results/scheme1_ccs23/`
- `experiments/results/scheme2_server/`
- `experiments/results/scheme3_offline/`
- `experiments/results/secpq/`
- `experiments/results/revoke/`

Plot-generation scripts are located in the image output directory `experiments/results/pic_new/` and can be run directly from the project root:

```bash
python experiments/results/pic_new/communication/generate_communication_charts.py
python experiments/results/pic_new/computation/generate_computation_charts.py
python experiments/results/pic_new/size/generate_size_charts.py
```

Corresponding output directories:

- `experiments/results/pic_new/communication/`
- `experiments/results/pic_new/computation/`
- `experiments/results/pic_new/size/`

## 10. Windows Notes

- It is recommended to always run experiments with Python from your virtual environment.
- If PowerShell output redirection causes encoding errors, set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## 11. Suggested Reproduction Flow

1. Create a virtual environment and install dependencies.
2. Run `tests/test_schemes_comparison.py` as a basic sanity check.
3. Run small-scale checks for `our_decart` and `our_decart_star` first.
4. Run other baseline schemes and revoke experiments.
5. Finally, run the plotting scripts to generate paper figures.

