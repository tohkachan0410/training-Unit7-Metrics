#!/usr/bin/env bash

set -euo pipefail

cd /home/u5450760/train_6


echo ""
echo "======================================"
echo "TRAINING UNIT7 RANKING PIPELINE"
echo "======================================"
echo ""


echo "[1/2] Training all models"

bash scripts/train_all.sh


echo ""
echo "[2/2] Scoring and evaluation"

bash scripts/score_all.sh


echo ""
echo "======================================"
echo "EVERYTHING FINISHED"
echo "======================================"

echo ""
echo "Checkpoints:"
ls -lh outputs/checkpoints/

echo ""
echo "Scores:"
ls -lh outputs/scores/

echo ""
echo "Rankings:"
ls -lh outputs/rankings/

echo ""
echo "Metrics:"
cat outputs/metrics/top100_metrics.csv
