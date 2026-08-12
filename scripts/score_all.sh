#!/usr/bin/env bash

set -euo pipefail

cd /home/u5450760/train_6


echo "======================================"
echo "Scoring ResNet18"
echo "======================================"

python src/score_images.py \
  --model resnet18 \
  --checkpoint outputs/checkpoints/resnet18_best.pt


echo "======================================"
echo "Scoring VGG16"
echo "======================================"

python src/score_images.py \
  --model vgg16 \
  --checkpoint outputs/checkpoints/vgg16_best.pt


echo "======================================"
echo "Scoring ResNet50"
echo "======================================"

python src/score_images.py \
  --model resnet50 \
  --checkpoint outputs/checkpoints/resnet50_best.pt


echo "======================================"
echo "Building confidence + ensemble"
echo "======================================"

python src/build_indicators.py


echo "======================================"
echo "Evaluating Top100"
echo "======================================"

python src/evaluate_top100.py


echo "======================================"
echo "ALL SCORING FINISHED"
echo "======================================"
