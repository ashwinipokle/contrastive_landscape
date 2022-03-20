#!/bin/bash
python main.py --cfg config/overpara_pre_alter.yaml --logDir results/simplified-pred --normalize_repr True --model simplified-pred --lr 0.005 --batch_size 512 --log_metrics --use_bn True