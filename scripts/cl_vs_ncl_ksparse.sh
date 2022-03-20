#!/bin/bash
python main_simclr.py --cfg config/overpara_nopre_alter.yaml --logDir results/simclr --normalize_repr True --model simclr --lr 0.005 --batch_size 512 --temperature 0.05 --log_metrics --use_bn True
python main.py --cfg config/overpara_nopre_alter.yaml --logDir results/simplified --normalize_repr True --model simplified --lr 0.005 --batch_size 512 --log_metrics --use_bn True