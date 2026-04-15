# DGMF: Dynamic Gaussian Mixture Fusion for Semi-Supervised Multi-View Classification

This project provides the training entry point for **DGMF**, a semi-supervised multi-view classification framework based on graph-enhanced representation learning and Gaussian-mixture-based reliability modeling.

### Python packages

- `python>=3.8`
- `torch>=1.12`

## Suggested project structure

A minimal project structure consistent with the script is:

```text
project_root/
├── main-semi-classification.py
├── Utils.py
├── trainers/
│   └── DGMF_trainer_time_final.py
└── exp/
    └── result/
```

## How to run

### Basic command

```bash
python main-semi-classification.py DGMF --dataset HW --device 0 --dir_h exp1
```

### Example with custom hyperparameters

```bash
python main-semi-classification.py DGMF \
  --dataset 100leaves \
  --device 0 \
  --dir_h dgmf_run \
  --learning_rate 0.003 \
  --weight_decay 0.001 \
  --num_epoch 190 \
  --ratio 0.1 \
  --knns 30 \
  --K 3 \
  --l1 1 \
  --l2 0.001 \
  --l3 0.0001
```
