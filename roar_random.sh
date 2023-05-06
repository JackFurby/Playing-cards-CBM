#!/bin/bash

for j in 1 2 3 4 5
do
  DATA_SPLIT=three
  DATA_POST=splits/three

  for i in 1 3 5 7 9
  do
    RAN_NUM=$(shuf -i 0-1000 -n1)
    ## random
    python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/random/0.$i --dataset_postfix $DATA_POST --save_name "XtoC-$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.066 --epochs 200 --weighted_loss  --seed $RAN_NUM
    python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/roar/random/0.$i --dataset_postfix splits/three_card_poker --save_name "CtoY-$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 10 --lr 0.01 --epochs 200 --pretrained_weights "./saves/XtoC-$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM.pth" --freeze --use_sig --seed $RAN_NUM
    python concept_to_standard_model_converter.py --model_path "./saves/CtoY-$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM.pth" --model_out_path "./saves/roar/$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM.pth" --num_concepts 52 --num_classes 6 --end_to_end
    echo "Completed random"

    ## random (concepts)
    python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/concept_random/0.$i --dataset_postfix $DATA_POST --save_name "XtoC-$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.066 --epochs 200 --weighted_loss  --seed $RAN_NUM
    python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/roar/concept_random/0.$i --dataset_postfix splits/three_card_poker --save_name "CtoY-$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 10 --lr 0.01 --epochs 200 --pretrained_weights "./saves/XtoC-$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM.pth" --freeze --use_sig  --seed $RAN_NUM
    python concept_to_standard_model_converter.py --model_path "./saves/CtoY-$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM.pth" --model_out_path "./saves/roar/$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM.pth" --num_concepts 52 --num_classes 6 --end_to_end
    echo "Completed concepts random"
  done

  DATA_SPLIT=three_card_poker
  DATA_POST=splits/three_card_poker

  for i in 1 3 5 7 9
  do
    RAN_NUM=$(shuf -i 0-1000 -n1)
    ## random
    python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/random/0.$i --dataset_postfix $DATA_POST --save_name "XtoC-$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.053 --epochs 200 --weighted_loss  --seed $RAN_NUM
    python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/roar/random/0.$i --dataset_postfix splits/three_card_poker --save_name "CtoY-$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 15 --lr 0.002 --epochs 200 --pretrained_weights "./saves/XtoC-$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM.pth" --freeze --use_sig --seed $RAN_NUM
    python concept_to_standard_model_converter.py --model_path "./saves/CtoY-$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM.pth" --model_out_path "./saves/roar/$DATA_SPLIT-roar-random-0_$i-seed_$RAN_NUM.pth" --num_concepts 52 --num_classes 6 --end_to_end
    echo "Completed random"

    ## random (concepts)
    python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/concept_random/0.$i --dataset_postfix $DATA_POST --save_name "XtoC-$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.053 --epochs 200 --weighted_loss  --seed $RAN_NUM
    python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/roar/concept_random/0.$i --dataset_postfix splits/three_card_poker --save_name "CtoY-$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 15 --lr 0.002 --epochs 200 --pretrained_weights "./saves/XtoC-$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM.pth" --freeze --use_sig  --seed $RAN_NUM
    python concept_to_standard_model_converter.py --model_path "./saves/CtoY-$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM.pth" --model_out_path "./saves/roar/$DATA_SPLIT-roar-concepts-random-0_$i-seed_$RAN_NUM.pth" --num_concepts 52 --num_classes 6 --end_to_end
    echo "Completed concepts random"
  done
done
