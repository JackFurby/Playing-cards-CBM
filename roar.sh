#!/bin/bash

#FILES="./saves/converted/*.pth"
#for f in $FILES
#do
for MODEL_PATH in ./saves/converted/sequential-poker-seed-625.pth ./saves/converted/independent-three-seed-950.pth
do
  SUB="poker"

  MODEL_NAME=$(basename $MODEL_PATH .pth)
  DATA_ROOT=dataset/playing-cards

  if [[ "$MODEL_PATH" == *"$SUB"* ]]; then
    DATA_POST=splits/three_card_poker
    DATA_SPLIT=three_card_poker
  else
    MODEL_NAME=independent-three-seed-950
    DATA_POST=splits/three
    DATA_SPLIT=three
  fi

  echo "Starting ROAR with model $MODEL_NAME"

  # generate dataset
  python experiments/roar/data_gen.py --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --dataset_split $DATA_SPLIT --n_classes 6 --n_concepts 52 --pretrained_weights $MODEL_PATH

  for i in 1 3 5 7 9
  do
    echo "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-random-0_$i.pth"
    if test -f "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-random-0_$i.pth"; then
      ## random
      python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/random/0.$i --dataset_postfix $DATA_POST --save_name "$MODEL_NAME-XtoC-$DATA_SPLIT-roar-random-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
      python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/roar/random/0.$i --dataset_postfix splits/three_card_poker --save_name "$MODEL_NAME-CtoY-$DATA_SPLIT-roar-random-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/$MODEL_NAME-XtoC-$DATA_SPLIT-roar-random-0_$i.pth" --freeze --use_sig
      python concept_to_standard_model_converter.py --model_path "./saves/$MODEL_NAME-CtoY-$DATA_SPLIT-roar-random-0_$i.pth" --model_out_path "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-random-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
    fi
    echo "Completed random"
    #if test -f "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-lrp-0_$i.pth"; then
      ##lrp
    #  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/lrp/$MODEL_NAME/0.$i --dataset_postfix $DATA_POST --save_name "$MODEL_NAME-XtoC-$DATA_SPLIT-roar-lrp-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
    #  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "$MODEL_NAME-CtoY-$DATA_SPLIT-roar-lrp-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/$MODEL_NAME-XtoC-$DATA_SPLIT-roar-lrp-0_$i.pth" --freeze --use_sig
    #  python concept_to_standard_model_converter.py --model_path "./saves/$MODEL_NAME-CtoY-$DATA_SPLIT-lrp-0_$i.pth" --model_out_path "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-lrp-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
    #fi
    #echo "Completed LRP"
    #if test -f "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-lrp-positive-0_$i.pth"; then
      #lrp (only positive relevance)
    #  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/lrp_positive/$MODEL_NAME/0.$i --dataset_postfix $DATA_POST --save_name "$MODEL_NAME-XtoC-$DATA_SPLIT-roar-lrp-positive-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
    #  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "$MODEL_NAME-CtoY-$DATA_SPLIT-roar-lrp-positive-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/$MODEL_NAME-XtoC-$DATA_SPLIT-roar-lrp-positive-0_$i.pth" --freeze --use_sig
    #  python concept_to_standard_model_converter.py --model_path "./saves/$MODEL_NAME-CtoY-$DATA_SPLIT-roar-lrp-positive-0_$i.pth" --model_out_path "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-lrp-positive-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
    #fi
    #echo "Completed LRP positive"
    if test -f "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-concepts-random-0_$i.pth"; then
      ## random (concepts)
      python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/concept_random/0.$i --dataset_postfix $DATA_POST --save_name "$MODEL_NAME-XtoC-$DATA_SPLIT-roar-concepts-random-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
      python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/roar/concept_random/0.$i --dataset_postfix splits/three_card_poker --save_name "$MODEL_NAME-CtoY-$DATA_SPLIT-roar-concepts-random-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/$MODEL_NAME-XtoC-$DATA_SPLIT-roar-concepts-random-0_$i.pth" --freeze --use_sig
      python concept_to_standard_model_converter.py --model_path "./saves/$MODEL_NAME-CtoY-$DATA_SPLIT-roar-concepts-random-0_$i.pth" --model_out_path "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-concepts-random-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
    fi
    echo "Completed concepts random"
    #if test -f "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-concepts-lrp-0_$i.pth"; then
      ##lrp (concepts)
    #  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/concept_lrp/$MODEL_NAME/0.$i --dataset_postfix $DATA_POST --save_name "$MODEL_NAME-XtoC-$DATA_SPLIT-roar-concepts-lrp-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
    #  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "$MODEL_NAME-CtoY-$DATA_SPLIT-roar-concepts-lrp-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/$MODEL_NAME-XtoC-$DATA_SPLIT-roar-concepts-lrp-0_$i.pth" --freeze --use_sig
    #  python concept_to_standard_model_converter.py --model_path "./saves/$MODEL_NAME-CtoY-$DATA_SPLIT-roar-concepts-lrp-0_$i.pth" --model_out_path "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-concepts-lrp-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
    #fi
    #echo "Completed concepts LRP"
    #if test -f "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-concepts-lrp-positive-0_$i.pth"; then
      #lrp (concepts, only positive relevance)
    #  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/concept_lrp_positive/$MODEL_NAME/0.$i --dataset_postfix $DATA_POST --save_name "$MODEL_NAME-XtoC-$DATA_SPLIT-roar-concepts-lrp-positive-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
    #  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "$MODEL_NAME-CtoY-$DATA_SPLIT-roar-concepts-lrp-positive-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/$MODEL_NAME-XtoC-$DATA_SPLIT-roar-concepts-lrp-positive-0_$i.pth" --freeze --use_sig
    #  python concept_to_standard_model_converter.py --model_path "./saves/$MODEL_NAME-CtoY-$DATA_SPLIT-roar-concepts-lrp-positive-0_$i.pth" --model_out_path "./saves/roar/$MODEL_NAME-$DATA_SPLIT-roar-concepts-lrp-positive-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
    #fi
    #echo "Completed concepts LRP positive"
  done
done
