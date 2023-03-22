#!/bin/bash

touch results/metrics/ois_poker.txt

FILES="saves/converted/*sequential*poker*.pth"
for MODEL_PATH in $FILES
do
  MODEL_NAME=$(basename $MODEL_PATH .pth)
  DATA_ROOT=dataset/playing-cards
  DATA_POST=splits/three_card_poker

  echo "Starting OIS with model $MODEL_NAME"

  python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt ./results/metrics/ois_poker.txt
done

FILES="saves/converted/*independent*poker*.pth"
for MODEL_PATH in $FILES
do
  MODEL_NAME=$(basename $MODEL_PATH .pth)
  DATA_ROOT=dataset/playing-cards
  DATA_POST=splits/three_card_poker

  echo "Starting OIS with model $MODEL_NAME"

  python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt ./results/metrics/ois_poker.txt
done

touch results/metrics/ois_three.txt

FILES="saves/converted/*three*.pth"
for MODEL_PATH in $FILES
do
  MODEL_NAME=$(basename $MODEL_PATH .pth)
  DATA_ROOT=dataset/playing-cards
  DATA_POST=splits/three_card_poker

  echo "Starting OIS with model $MODEL_NAME"

  python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt ./results/metrics/ois_three.txt
done
