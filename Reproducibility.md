# Experiment Reproducibility

This document outlines how to reproduce the experiments presented in the paper. It covers the file structure, how to configure each experimental setting, and how to run diagnostics and evaluations.

---

## Overview

There are two main files relevant to reproducing the experiments:

1. **`train.py`** – Located at the root of the repository. This is the main script used to train models.
2. **`contrastive_model.py`** – Located in `quantum_contrastive/src/quantum_contrastive/models/`. This file defines the `ContrastiveModel` class, which includes the encoder and projection head logic, as well as the quantum feature map setup.

The experiments are divided into two main tracks:

- **Experiment 1** – Uses different **projection heads** (MLP, Bottleneck Linear, or VQC).
- **Experiment 2** – Uses a **quantum feature map** (QFM) as a similarity kernel in place of a projection head.

Each experiment is explained below with instructions on how to configure and run them.

---

## Experiment 1: Projection Heads

In `contrastive_model.py`, the `ContrastiveModel` class is defined starting around line 218. The section beginning at line 233 allows you to set which projection head is used.

### Selecting a Projection Head
Uncomment the relevant lines to activate the desired projection head:

- **MLP Head**: Uncomment line 236.
- **Bottleneck Linear Head**: Uncomment line 237.
- **VQC Head**: Uncomment lines 241–246.

### VQC Hyperparameters
Within the VQC section:

- **Line 241** – Sets the number of qubits (`n_qubits`).
- **Line 242** – Defines the number of observables used for readout. The specific observables (Z-only or X/Y/Z) can be adjusted in lines 165–170.
- **Line 243** – Sets the number of circuit layers (`n_layers`).

### Argument Parser Options
Additional hyperparameters such as `--epochs`, `--temperature`, and `--lr` can be adjusted in the `main()` function, starting around line 473 in `train.py`.

### Device Configuration
At line 512 in `train.py`, the training device is set:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

If using the VQC head, it is recommended to force CPU usage for compatibility with PennyLane:

`device = torch.device("cpu")`


For classical heads (MLP or Bottleneck), MPS can remain enabled if available for faster training.

## Saving and Visualizing the Model

Once training completes, the model is saved to contrastive_model.pth. You can generate a t-SNE visualization of the learned embeddings by running:

python visualize_tsne.py


This script will use the saved model to generate a 2D plot of the embedding space.

## Experiment 2: Quantum Feature Maps (QFM)

To run experiments using a quantum feature map:

Run train.py with the flag:

--use_feature_map


Additional options (e.g., --n_qubits, --n_layers, --kernel, --loss, --temperature) can be set via command-line arguments. These options are defined starting around line 473 in train.py.

In practice, it may be easier to directly modify the script for these values before execution.

## Supported Kernels and Losses

Kernels:

- fidelity: raw inner product squared

- fsrbf: RBF applied to Fubini–Study distance

Losses:

- infonce: standard NT-Xent objective

- mmd: unbiased maximum mean discrepancy

## Metrics and Diagnostics

Each training run prints and/or logs the following:

- Loss value per epoch

- Time per batch and total epoch time

- Accuracy metrics (Linear Probe and k-NN accuracy)

## Additional Diagnostics

Two training loops are defined in train.py:

- Projection heads: train_one_epoch() — starts around line 412.

- Quantum feature maps: train_one_epoch_qfm() — starts around line 278.

These functions include optional metrics that can be enabled for additional logging:

Kernel diagnostics:

- Mean and standard deviation of off-diagonal kernel entries

- Δ-gap between positive and negative pairs

- Bandwidth σ² used in FS-RBF kernel

These are typically printed to the console and can also be saved for reproducibility.

To enable additional logging, uncomment or insert print(...) or logging.info(...) calls within the training functions as needed.

# Final Notes

Be sure to set random seeds for reproducibility.

Use drop_last=True in dataloaders to ensure correct NT-Xent pairing.

If using PennyLane with a QNode or VQC, use torch.device("cpu") to avoid instability.

Consider saving kernel diagnostics to CSV files (e.g., qfm_metrics.csv) for easier table creation and analysis.