#!/bin/bash
# Reproduce all experiments from the paper
# Usage: bash scripts/run_experiments.sh [GPU_ID]

GPU=${1:-0}
DEVICE="cuda:${GPU}"
CONFIG="configs/default.yaml"

echo "============================================"
echo "FMC-SVDD: Reproducing Paper Experiments"
echo "Device: ${DEVICE}"
echo "============================================"

# Experiment 1: Inflation Factor Decomposition (Table II, Fig. 3)
echo "[Exp 1] Inflation Factor Decomposition..."
python src/train.py --config ${CONFIG} --experiment exp1_inflation --device ${DEVICE}

# Experiment 2: Negative Feedback Dynamics (Fig. 4)
echo "[Exp 2] RAA Feedback Dynamics..."
python src/train.py --config ${CONFIG} --experiment exp2_dynamics --device ${DEVICE}

# Experiment 3: Convergence Scaling (Fig. 5)
echo "[Exp 3] Convergence Scaling Verification..."
python src/train.py --config ${CONFIG} --experiment exp3_scaling --device ${DEVICE}

# Experiment 4: DRO Robustness (Table III, Fig. 6)
echo "[Exp 4] DRO Robustness..."
python src/train.py --config ${CONFIG} --experiment exp4_dro --device ${DEVICE}

# Experiment 5: Performance Comparison (Table IV)
echo "[Exp 5] Performance Comparison..."
for DATASET in swat wadi; do
    python src/train.py --config ${CONFIG} --experiment exp5_comparison \
        --dataset ${DATASET} --device ${DEVICE}
done

# Experiment 6: Ablation Study (Table V)
echo "[Exp 6] Ablation Study..."
python src/train.py --config ${CONFIG} --experiment exp6_ablation --device ${DEVICE}

echo "============================================"
echo "All experiments complete."
echo "Results saved to outputs/"
echo "============================================"
