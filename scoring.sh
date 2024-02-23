#!/bin/bash

#touch results/metrics/oracle_poker.txt
#touch results/metrics/niche_poker.txt
#touch results/metrics/oracle_three.txt
#touch results/metrics/niche_three.txt
touch results/metrics/oracle_chexpert_f1.txt
touch results/metrics/niche_chexpert_f1.txt
#touch results/metrics/oracle_class_poker.txt
#touch results/metrics/niche_class_poker.txt


#for MODE in niche oracle
for MODE in oracle
do
  """
  FILES="saves/converted/*sequential*poker*.pth"
  for MODEL_PATH in $FILES
  do
    SUB="class"
    if grep -q "$SUB" <<< "$MODEL_PATH"; then
      for i in 1 2 3
      do
        MODEL_NAME=$(basename $MODEL_PATH .pth)
        DATA_ROOT=dataset/playing-cards
        DATA_POST=splits/three_card_poker_class_level

        echo "Starting $MODE scoring with model $MODEL_NAME"

        #python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 11 --results_txt "./results/metrics/${MODE}_class_poker.txt" --mode $MODE --dataset_name cards
      done
    else
      for i in 1 2 3
      do
        MODEL_NAME=$(basename $MODEL_PATH .pth)
        DATA_ROOT=dataset/playing-cards
        DATA_POST=splits/three_card_poker

        echo "Starting $MODE scoring with model $MODEL_NAME"

        #python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt "./results/metrics/${MODE}_poker.txt" --mode $MODE --dataset_name cards
      done
    fi
  done

  FILES="saves/converted/*independent*poker*.pth"
  for MODEL_PATH in $FILES
  do
    # We do not need to run for class-level poker as it uses the same x to c model part as sequential
    for i in 1 2 3
    do
      MODEL_NAME=$(basename $MODEL_PATH .pth)
      DATA_ROOT=dataset/playing-cards
      DATA_POST=splits/three_card_poker

      echo "Starting $MODE scoring with model $MODEL_NAME"

      #python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt "./results/metrics/${MODE}_poker.txt" --mode $MODE  --dataset_name cards
    done
  done

  FILES="saves/converted/*joint_sig*poker*.pth"
  for MODEL_PATH in $FILES
  do
    SUB="class"
    if grep -q "$SUB" <<< "$MODEL_PATH"; then
      for i in 1 2 3
      do
        MODEL_NAME=$(basename $MODEL_PATH .pth)
        DATA_ROOT=dataset/playing-cards
        DATA_POST=splits/three_card_poker_class_level

        echo "Starting $MODE scoring with model $MODEL_NAME"

        #python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 11 --results_txt "./results/metrics/${MODE}_class_poker.txt" --mode $MODE --dataset_name cards
      done
    else
      for i in 1 2 3
      do
        MODEL_NAME=$(basename $MODEL_PATH .pth)
        DATA_ROOT=dataset/playing-cards
        DATA_POST=splits/three_card_poker

        echo "Starting $MODE scoring with model $MODEL_NAME"

        #python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt "./results/metrics/${MODE}_poker.txt" --mode $MODE --dataset_name cards
      done
    fi
  done

  FILES="saves/converted/*sequential*three*.pth"
  for MODEL_PATH in $FILES
  do
    for i in 1 2 3
    do
      MODEL_NAME=$(basename $MODEL_PATH .pth)
      DATA_ROOT=dataset/playing-cards
      DATA_POST=splits/three_card_poker

      echo "Starting $MODE scoring with model $MODEL_NAME"

      #python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt "./results/metrics/${MODE}_three.txt" --mode $MODE  --dataset_name cards
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

      echo "Starting $MODE scoring with model $MODEL_NAME"

      #python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 6 --n_concepts 52 --results_txt "./results/metrics/${MODE}_three.txt" --mode $MODE  --dataset_name cards
    done
  done
  """

  FILES="saves/chexpert/joint_?.ckpt"
  for MODEL_PATH in $FILES
  do
    for i in 1 2 3
    do
      MODEL_NAME=$(basename $MODEL_PATH .ckpt)
      DATA_ROOT=dataset/chexpert
      DATA_POST=splits/cheXbert_train_val_test

      echo "Starting $MODE scoring with model $MODEL_NAME"

      python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 2 --n_concepts 13 --results_txt "./results/metrics/${MODE}_chexpert_f1.txt" --mode $MODE  --dataset_name chexpert --set_splits
    done
  done

  FILES="saves/chexpert/sequential_?.ckpt"
  for MODEL_PATH in $FILES
  do
    for i in 1 2 3
    do
      MODEL_NAME=$(basename $MODEL_PATH .ckpt)
      DATA_ROOT=dataset/chexpert
      DATA_POST=splits/cheXbert

      echo "Starting $MODE scoring with model $MODEL_NAME"

      python experiments/purity/scoring.py --pretrained_weights $MODEL_PATH --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --n_classes 2 --n_concepts 13 --results_txt "./results/metrics/${MODE}_chexpert_f1.txt" --mode $MODE  --dataset_name chexpert --set_splits
    done
  done
done
