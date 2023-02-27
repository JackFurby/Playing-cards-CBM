#!/bin/bash

#MODEL_PATH=./saves/converted/independent-three-seed-950.pth
#MODEL_NAME=independent-three-seed-950
#DATA_ROOT=dataset/playing-cards
#DATA_POST=splits/three
#DATA_SPLIT=three

MODEL_PATH=./saves/converted/sequential-poker-seed-625.pth
MODEL_NAME=sequential-poker-seed-625
DATA_ROOT=dataset/playing-cards
DATA_POST=splits/three_card_poker
DATA_SPLIT=three_card_poker

# generate dataset

python experiments/roar/data_gen.py --dataset_root $DATA_ROOT --dataset_postfix $DATA_POST --dataset_split $DATA_SPLIT --n_classes 6 --n_concepts 52 --pretrained_weights $MODEL_PATH

# # Sequential training

for i in 1 3 5 7 9
do
  ## random
  #python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/random/0.$i --dataset_postfix $DATA_POST --save_name "sequential-XtoC-three-roar-random-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
  #python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-three-roar-random-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-three-roar-random-0_$i.pth" --freeze --use_sig
  #python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-three-roar-random-0_$i.pth" --model_out_path "./saves/roar/sequential-XtoC-three-roar-random-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
  ##lrp
  #python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/lrp/$MODEL_NAME/0.$i --dataset_postfix $DATA_POST --save_name "sequential-XtoC-three-roar-lrp-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
  #python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-three-roar-lrp-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-three-roar-lrp-0_$i.pth" --freeze --use_sig
  #python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-three-roar-lrp-0_$i.pth" --model_out_path "./saves/roar/sequential-XtoC-three-roar-lrp-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
  ## random (concepts)
  #python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/concept_random/0.$i --dataset_postfix $DATA_POST --save_name "sequential-XtoC-three-roar-concepts-random-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
  #python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-three-roar-concepts-random-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-three-roar-concepts-random-0_$i.pth" --freeze --use_sig
  #python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-three-roar-concepts-random-0_$i.pth" --model_out_path "./saves/roar/sequential-XtoC-three-roar-concepts-random-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
  ##lrp (concepts)
  #python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/concept_lrp/$MODEL_NAME/0.$i --dataset_postfix $DATA_POST --save_name "sequential-XtoC-three-roar-concepts-lrp-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
  #python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-three-roar-concepts-lrp-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-three-roar-concepts-lrp-0_$i.pth" --freeze --use_sig
  #python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-three-roar-concepts-lrp-0_$i.pth" --model_out_path "./saves/roar/sequential-XtoC-three-roar-concepts-lrp-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end


  # random
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/random/0.$i --dataset_postfix $DATA_POST --save_name "sequential-XtoC-poker-roar-random-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-poker-roar-random-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.046 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-poker-roar-random-0_$i.pth" --freeze --use_sig
  python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-poker-roar-random-0_$i.pth" --model_out_path "./saves/roar/sequential-XtoC-poker-roar-random-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
  #lrp
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/lrp/$MODEL_NAME/0.$i --dataset_postfix $DATA_POST --save_name "sequential-XtoC-poker-roar-lrp-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-poker-roar-lrp-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-poker-roar-lrp-0_$i.pth" --freeze --use_sig
  python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-poker-roar-lrp-0_$i.pth" --model_out_path "./saves/roar/sequential-XtoC-poker-roar-lrp-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
  # random (concepts)
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/concept_random/0.$i --dataset_postfix $DATA_POST --save_name "sequential-XtoC-poker-roar-concepts-random-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-poker-roar-concepts-random-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-poker-roar-concepts-random-0_$i.pth" --freeze --use_sig
  python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-poker-roar-concepts-random-0_$i.pth" --model_out_path "./saves/roar/sequential-XtoC-poker-roar-concepts-random-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
  #lrp (concepts)
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/roar/concept_lrp/$MODEL_NAME/0.$i --dataset_postfix $DATA_POST --save_name "sequential-XtoC-poker-roar-concepts-lrp-0_$i" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss
  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-poker-roar-concepts-lrp-0_$i"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-poker-roar-concepts-lrp-0_$i.pth" --freeze --use_sig
  python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-poker-roar-concepts-lrp-0_$i.pth" --model_out_path "./saves/roar/sequential-XtoC-poker-roar-concepts-lrp-0_$i.pth" --num_concepts 52 --num_classes 6 --end_to_end
done
