<h1 align="center"><a href="https://ieeexplore.ieee.org/abstract/document/11072185" style="color:#9C276A">
Cross-Domain Hyperspectral Image Classification Based on Bi-Directional Domain Adaptation (BiDA)</a></h1>
<h4 align="center">Official PyTorch Implementation & Systematic Tokenization Framework</h4>

<div align="center">

[[Paper 📰]](https://ieeexplore.ieee.org/abstract/document/11072185) [[Datasets 🤗]](https://github.com/YuxiangZhang-BIT/Data-CSHSI)

</div>

## Overview

This repository provides an aligned, enhanced PyTorch implementation of **Bi-directional Domain Adaptation (BiDA)** for cross-temporal and cross-scene hyperspectral image (HSI) classification (published in *IEEE Transactions on Circuits and Systems for Video Technology*, 2025).

![Method](assets/BiDA.png "BiDA Method Overview")

### Key Paper Alignments & Codebase Audit Upgrades
1. **Re-enabled and Formally Aligned Maximum Mean Discrepancy ($\mathcal{L}_{MMD}$)**:
   - Calculates distribution discrepancy across full feature token spaces ($\tilde{\mathbf{T}}_s$ vs. $\tilde{\mathbf{T}}_t$ on source/target branches, and $\tilde{\mathbf{T}}_{s \to t}$ vs. $\tilde{\mathbf{T}}_{t \to s}$ on the coupled branch), strictly adhering to Equations 13 & 14 in the paper:
   $$\mathcal{L}_{MMD} = \text{MMD}(\tilde{\mathbf{T}}_s, \tilde{\mathbf{T}}_t) + \text{MMD}(\tilde{\mathbf{T}}_{s \to t}, \tilde{\mathbf{T}}_{t \to s})$$
2. **Strict Unified Total Loss Formulation**:
   - The unified objective function strictly matches Equation 18:
   $$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_1 (\mathcal{L}_{MMD} + \mathcal{L}_{Bi-distill}) + \lambda_2 \mathcal{L}_{con}$$
   with paper-optimal hyperparameter weights $\lambda_1 = 0.1$ and $\lambda_2 = 1.0$.
3. **ARS Noise Perturbation Pipelines**:
   - Implemented Gaussian noise ($\sigma = 0.05$), RandomRC (random rotation & spatial reflection/cropping), and Radiation noise according to Section II-C and Table IX.
4. **Modular Tokenizer Architecture**:
   - Retains the default **Spatial-Spectral Semantic Tokenizer** and provides three alternative tokenizers to systematically evaluate tokenization strategies:
     - `semantic`: Spatial-spectral semantic tokenizer ($L=4$ semantic tokens + $1$ classification token $T_{cls}$).
     - `patch`: Standard spatial patch tokenizer (flattening 2D spatial patches across all spectral channels).
     - `band_grouping`: Band-grouping spectral tokenizer (splitting continuous spectral channels into contiguous sub-band group tokens).
     - `3d_cube`: 3D-cube tokenizer (extracting spatial-spectral sub-cubes via strided 3D convolutions).
5. **Per-Epoch Training Dynamics & Curve Logging**:
   - Automatically tracks training dynamics and outputs loss curves ($\mathcal{L}_{total}, \mathcal{L}_{cls}, \mathcal{L}_{MMD}, \mathcal{L}_{Bi-distill}, \mathcal{L}_{con}$) and test metrics (Overall Accuracy, Average Accuracy, Kappa) per epoch into `training_dynamics.png` and `history.json`.

---

## Environment Setup

### 1. Create a Python environment

Python 3.9+ is recommended.

```bash
conda create -n bida python=3.10 -y
conda activate bida
```

### 2. Install PyTorch

Please install PyTorch according to your CUDA version from the official guide:
https://pytorch.org/get-started/locally/

Example (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation

The default entry script runs the Houston cross-temporal task (`Houston13` $\to$ `Houston18`). Organize the dataset folder as follows:

```text
./Houston/
├── Houston13.mat
├── Houston13_7gt.mat
├── Houston18.mat
└── Houston18_7gt.mat
```

The `.mat` files should contain:
- `Houston13.mat` / `Houston18.mat`: `ori_data`
- `Houston13_7gt.mat` / `Houston18_7gt.mat`: `map`

For MFF tree species classification:
- `MFF_SD.mat`, `MFF_TD1.mat`, `MFF_TD2.mat`

For HyRANK satellite cross-scene:
- `Dioni.mat`, `Dioni_gt_out68.mat`, `Loukia.mat`, `Loukia_gt_out68.mat`

---

## Training and Evaluation

### Default Run (Semantic Tokenizer, Paper Defaults)
```bash
python main.py \
    --model BiDA \
    --tokenizer_type semantic \
    --source_name Houston13 \
    --target_name Houston18 \
    --dataset_dir ./Houston/ \
    --patch_size 13 \
    --num_tokens 4 \
    --dim 64 \
    --depth 3 \
    --bs 128 \
    --epoch 200 \
    --lr 1e-2 \
    --lambda1 0.1 \
    --lambda2 1.0 \
    --ars_noise randomrc_gauss \
    --seed 2100
```

### Evaluating Alternative Tokenization Strategies
Easily toggle `--tokenizer_type` between `semantic`, `patch`, `band_grouping`, and `3d_cube`:

```bash
# 1. Standard Spatial Patch Tokenizer
python main.py --tokenizer_type patch

# 2. Band-Grouping Spectral Tokenizer
python main.py --tokenizer_type band_grouping

# 3. 3D-Cube Tokenizer
python main.py --tokenizer_type 3d_cube
```

---

## Key Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--tokenizer_type` | `str` | `semantic` | Tokenizer: `['semantic', 'patch', 'band_grouping', '3d_cube']` |
| `--lambda1` | `float` | `0.1` | Regularization weight for $\mathcal{L}_{MMD} + \mathcal{L}_{Bi-distill}$ (Eq. 18) |
| `--lambda2` | `float` | `1.0` | Regularization weight for ARS consistency $\mathcal{L}_{con}$ (Eq. 18) |
| `--ars_noise` | `str` | `randomrc_gauss` | ARS perturbation pipeline: `['randomrc_gauss', 'gauss', 'randomrc', 'radiation', 'all', 'none']` |
| `--num_tokens` | `int` | `4` | Number of tokens $L$ (paper default: 4) |
| `--patch_size` | `int` | `13` | Input spatial patch size $w \times h$ (paper default: 13) |
| `--dim` | `int` | `64` | Token feature embedding dimension |
| `--depth` | `int` | `3` | Number of Transformer layers |
| `--lr` | `float` | `1e-2` | Base learning rate for SGD |
| `--momentum` | `float` | `0.9` | SGD momentum |
| `--weight_decay` | `float` | `1e-4` | SGD weight decay |
| `--scheduler` | `str` | `none` | Learning rate scheduler (`['none', 'cosine', 'step']`) |
| `--ema_decay` | `float` | `0.999` | EMA decay factor for teacher model |

---

## Artifacts Generated

Each training run produces artifacts in `./checkpoints/{model}_{tokenizer_type}/{source}to{target}/`:
- `model_best.pth`: Student model checkpoint achieving peak Target Domain Overall Accuracy (OA).
- `training_dynamics.png`: High-resolution dual-panel plot of training loss components and test accuracy / Kappa curves per epoch.
- `history.json`: Machine-readable training and evaluation metrics recorded every epoch.
- `final_results_{seed}.mat`: MATLAB `.mat` archive containing run parameters and full trajectory history.

---

## Running Test Suite

Verify all tokenizers, loss calculations, ARS noise pipelines, and training dynamics:

```bash
python -m unittest tests.test_bida_paper_alignment
```

---

## Citation

```bibtex
@ARTICLE{11072185,
    author={Zhang, Yuxiang and Li, Wei and Jia, Wen and Zhang, Mengmeng and Tao, Ran and Liang, Shunlin},
    journal={IEEE Transactions on Circuits and Systems for Video Technology},
    title={Cross-Domain Hyperspectral Image Classification Based on Bi-Directional Domain Adaptation},
    year={2025},
    volume={35},
    number={12},
    pages={12038-12051},
    doi={10.1109/TCSVT.2025.3586282}
}
```