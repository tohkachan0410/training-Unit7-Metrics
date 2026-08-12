#!/usr/bin/env bash

set -euo pipefail

cd /home/u5450760/train_6

echo "======================================"
echo "Training ResNet18"
echo "======================================"

python src/train_ranking.py \
  --config configs/ranking_resnet18.yaml \
  --wandb-mode online \
  --run-name ranking_resnet18_regression_lr1e-4_bs32


echo "======================================"
echo "Training VGG16"
echo "======================================"

python src/train_ranking.py \
  --config configs/ranking_vgg16.yaml \
  --wandb-mode online \
  --run-name ranking_vgg16_regression_lr1e-4_bs32


echo "======================================"
echo "Training ResNet50"
echo "======================================"

python src/train_ranking.py \
  --config configs/ranking_resnet50.yaml \
  --wandb-mode online \
  --run-name ranking_resnet50_regression_lr1e-4_bs32


echo "======================================"
echo "ALL MODELS FINISHED"
echo "======================================"
