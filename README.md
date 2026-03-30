# FMC-SVDD: Federated Multi-Center Deep SVDD with Lyapunov Convergence Guarantees

Official implementation for the paper:

> **Federated Multi-Center Deep SVDD with Lyapunov Convergence Guarantees for Anomaly Detection in IIoT Networks**

## Abstract

Federated learning enables collaborative anomaly detection across Industrial Internet of Things (IIoT) sites without sharing raw data. However, applying Deep Support Vector Data Description (SVDD) in federated settings causes *hypersphere boundary inflation* — aggregating heterogeneous local models inflates the normal-region radius, degrading detection precision. We propose **FMC-SVDD**, a federated multi-center Deep SVDD framework that provably controls this inflation through three contributions: (1) a **Radius-Aware Aggregation (RAA)** strategy that upweights clients exhibiting boundary inflation; (2) a **Distributionally Robust Optimization (DRO)** module that adaptively calibrates RAA sensitivity via Wasserstein ambiguity sets; and (3) a **two-phase training protocol** with a cluster-stabilization warmup followed by boundary optimization. We prove that FMC-SVDD converges to a bounded steady-state Lyapunov function at rate O(ηE), where η is the learning rate and E is local epochs. Experiments on SWaT and WADI benchmarks demonstrate that FMC-SVDD reduces the inflation ratio from 1.35× (FedAvg) to 1.05× while improving AUC by 3.8 percentage points over the best baseline.

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torch-geometric ≥ 2.0
- scikit-learn
- scipy
- numpy
- matplotlib

```bash
pip install torch torch-geometric scikit-learn scipy numpy matplotlib pyyaml
```

## Datasets

This work uses two public ICS security datasets. **Due to usage agreements, we do not distribute the data.** Please apply through the official channels:

- **SWaT** (Secure Water Treatment): Apply at [iTrust Labs](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
- **WADI** (Water Distribution): Apply at [iTrust Labs](https://itrust.sutd.edu.sg/itrust-labs_datasets/)

After obtaining the datasets, place them as follows:

```
data/
├── SWaT/
│   ├── SWaT_Dataset_Normal_v1.xlsx
│   └── SWaT_Dataset_Attack_v0.xlsx
└── WADI/
    ├── WADI_14days_new.csv
    └── WADI_attackdataLABLE.csv
```

## Quick Start

```bash
# Train FMC-SVDD on SWaT with default parameters
python src/train.py --config configs/default.yaml --dataset swat

# Train with specific settings
python src/train.py --config configs/default.yaml --dataset swat \
    --num_clients 3 --partition severe --rounds 50

# Reproduce all experiments
bash scripts/run_experiments.sh
```

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── encoder.py          # GAT + Transformer spatio-temporal encoder
│   │   └── svdd.py             # Multi-center Deep SVDD (k-Medoids, radii, scoring)
│   ├── federated/
│   │   ├── server.py           # Server: aggregation orchestration
│   │   ├── client.py           # Client: local training & inflation signals
│   │   ├── raa.py              # Radius-Aware Aggregation (Eq. 13-15)
│   │   └── dro.py              # Wasserstein DRO calibration (Eq. 24-27)
│   ├── utils/
│   │   ├── metrics.py          # Evaluation metrics (AUC, F1, IR)
│   │   └── data_partition.py   # Federated data partitioning
│   └── train.py                # Main training script (two-phase protocol)
├── configs/
│   └── default.yaml            # Default hyperparameters
├── scripts/
│   └── run_experiments.sh      # Experiment reproduction scripts
└── supplementary/              # Appendix materials
```

## Code Availability

> **Note:** Full implementation is under active development and will be released soon. The current repository provides the model architecture, algorithm skeleton, and configuration files.

## Citation

```bibtex
@article{fmc_svdd_2026,
  title={Federated Multi-Center Deep {SVDD} with {Lyapunov} Convergence Guarantees for Anomaly Detection in {IIoT} Networks},
  author={},
  journal={IEEE Transactions on Network Science and Engineering},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
