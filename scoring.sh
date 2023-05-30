#!/bin/bash

touch results/metrics/ois_poker.txt
touch results/metrics/nis_poker.txt
touch results/metrics/ois_three.txt
touch results/metrics/nis_three.txt


for MODE in niche oracle
do
  FILES="saves/converted/*sequential*poker*.pth"
  for MODEL_PATH in $FILES
  do
    for i in 1 2 3
    do
      MODEL_NAME=$(basename $MODEL_PATH .pth)
      DATA_ROOT=dataset/playing-cards
      DATA_POST=splits/three_card_poker

      echo "Starting scoring with model $MODEL_NAME"

      python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt ./results/metrics/nis_poker.txt --mode $MODE  --dataset_name cards
    done
  done

  FILES="saves/converted/*independent*poker*.pth"
  for MODEL_PATH in $FILES
  do
    for i in 1 2 3
    do
      MODEL_NAME=$(basename $MODEL_PATH .pth)
      DATA_ROOT=dataset/playing-cards
      DATA_POST=splits/three_card_poker

      echo "Starting scoring with model $MODEL_NAME"

      python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt ./results/metrics/nis_poker.txt --mode $MODE  --dataset_name cards
    done
  done

  FILES="saves/converted/*sequential*three*.pth"
  for MODEL_PATH in $FILES
  do
    for i in 1 2 3
    do
      MODEL_NAME=$(basename $MODEL_PATH .pth)
      DATA_ROOT=dataset/playing-cards
      DATA_POST=splits/three_card_poker

      echo "Starting scoring with model $MODEL_NAME"

      python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt ./results/metrics/nis_three.txt --mode $MODE  --dataset_name cards
    done
  done

  FILES="saves/converted/*independent*three*.pth"
  for MODEL_PATH in $FILES
  do
    for i in 1 2 3
    do
      MODEL_NAME=$(basename $MODEL_PATH .pth)
      DATA_ROOT=dataset/playing-cards
      DATA_POST=splits/three_card_poker

      echo "Starting scoring with model $MODEL_NAME"

      python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt ./results/metrics/nis_three.txt --mode $MODE  --dataset_name cards
    done
  done
done
