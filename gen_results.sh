#!/bin/bash

FILES="./saves/converted/*.pth"
for f in $FILES
do
  echo "Generating results for model $f"

  if grep -q "$SUB" <<< "$f"; then
    python experiments/relevancy.py --pretrained_weights $f --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 52 --n_concepts 17 --use_cpu
  else
    python experiments/relevancy.py --pretrained_weights $f --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --use_cpu
  fi
done
