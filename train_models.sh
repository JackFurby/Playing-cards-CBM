# single playing card images

# # Standard training (same architecture as CBMs)

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode joint --n_classes 52 --n_concepts 17 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "joint-single-seed-$RAN_NUM" --lr 0.05 --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --weighted_loss --attr_loss_weight 0 --epochs 200 --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/joint-single-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/standard-as-joint-single-seed-$RAN_NUM.pth" --num_concepts 17 --num_classes 52 --end_to_end
done

# # Independent training and sequential training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 17 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "XtoC-single-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 10 --lr 0.02 --epochs 200 --weighted_loss --seed $RAN_NUM

  python main.py --mode CtoY --n_concepts 17 --n_classes 52 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "independent-CtoY-single-seed-$RAN_NUM" --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.05 --epochs 200 --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/XtoC-single-seed-$RAN_NUM.pth" --model_2_path "./saves/independent-CtoY-single-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/independent-single-seed-$RAN_NUM.pth" --num_concepts 17 --num_classes 52

  python main.py --mode sequential --n_concepts 17 --n_classes 52 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "sequential-CtoY-single-seed-$RAN_NUM" --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 3 --lr 0.005 --epochs 100 --pretrained_weights "./saves/XtoC-single-seed-$RAN_NUM.pth" --freeze --use_sig --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-single-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/sequential-single-seed-$RAN_NUM.pth" --num_concepts 17 --num_classes 52 --end_to_end
done

# # Joint training (no sig)

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode joint --n_classes 52 --n_concepts 17 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "joint-single-seed-$RAN_NUM" --lr 0.01 --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --weighted_loss --attr_loss_weight 0.643 --epochs 200 --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/joint-single-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/joint-single-seed-$RAN_NUM.pth" --num_concepts 17 --num_classes 52 --end_to_end
done



# poker playing card images

# # Standard training (same architecture as CBMs)

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode joint --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "joint-poker-seed-$RAN_NUM" --lr 0.088 --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --weighted_loss --attr_loss_weight 0 --epochs 300 --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/joint-poker-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/standard-as-joint-poker-seed-$RAN_NUM.pth" --num_concepts 52 --num_classes 6 --end_to_end
done

# # Independent training and sequential training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "XtoC-poker-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --lr 0.02 --epochs 200 --weighted_loss --seed $RAN_NUM

  python main.py --mode CtoY --n_concepts 52 --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "independent-CtoY-poker-seed-$RAN_NUM" --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.01 --epochs 200 --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/XtoC-poker-seed-$RAN_NUM.pth" --model_2_path "./saves/independent-CtoY-poker-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/independent-poker-seed-$RAN_NUM.pth" --num_concepts 52 --num_classes 6

  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-poker-seed-$RAN_NUM"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 15 --lr 0.046 --epochs 200 --pretrained_weights "./saves/XtoC-poker-seed-$RAN_NUM.pth" --freeze --use_sig --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-poker-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/sequential-poker-seed-$RAN_NUM.pth" --num_concepts 52 --num_classes 6 --end_to_end
done

# # Joint training (no sig)

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode joint --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "joint-poker-seed-$RAN_NUM" --lr 0.025 --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --weighted_loss --attr_loss_weight 0.98 --epochs 300 --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/joint-poker-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/joint-poker-seed-$RAN_NUM.pth" --num_concepts 52 --num_classes 6 --end_to_end
done

# # Joint training (with sig)

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode joint --n_concepts 52 --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "joint_sig-poker-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --lr 0.05 --epochs 300 --attr_loss_weight 0.96 --use_sig --weighted_loss --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/joint_sig-poker-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/joint_sig-poker-seed-$RAN_NUM.pth" --num_concepts 52 --num_classes 6 --end_to_end
done


# three / poker playing card images

# # Independent and sequential training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three --save_name "XtoC-three-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss --seed $RAN_NUM

  python main.py --mode CtoY --n_concepts 52 --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "independent-CtoY-three-seed-$RAN_NUM" --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.01 --epochs 200 --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/XtoC-three-seed-$RAN_NUM.pth" --model_2_path "./saves/independent-CtoY-three-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/independent-three-seed-$RAN_NUM.pth" --num_concepts 52 --num_classes 6

  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-three-seed-$RAN_NUM"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.059 --epochs 200 --pretrained_weights "./saves/XtoC-three-seed-$RAN_NUM.pth" --freeze --use_sig --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-three-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/sequential-three-seed-$RAN_NUM.pth" --num_concepts 52 --num_classes 6 --end_to_end
done


# Class level poker cards

# # Independent and sequential taining

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 11 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker_class_level --save_name "XtoC-class-level-poker-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 3 --lr 0.0825 --epochs 100 --weighted_loss --seed $RAN_NUM

  python main.py --mode CtoY --n_concepts 11 --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker_class_level --save_name "independent-CtoY-class-level-poker-seed-$RAN_NUM" --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.064 --epochs 100 --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/XtoC-class-level-poker-seed-$RAN_NUM.pth" --model_2_path "./saves/independent-CtoY-class-level-poker-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/independent-class-level-poker-seed-$RAN_NUM.pth" --num_concepts 11 --num_classes 6

  python main.py --mode sequential --n_classes 6 --n_concepts 11 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker_class_level --save_name "sequential-CtoY-three-seed-$RAN_NUM"  --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 10 --lr 0.0846 --epochs 100 --pretrained_weights "./saves/XtoC-class-level-poker-seed-$RAN_NUM.pth" --freeze --use_sig --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/sequential-CtoY-three-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/sequential-class-level-poker-seed-$RAN_NUM.pth" --num_concepts 11 --num_classes 6 --end_to_end
done

# # Joint training (with sig)

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode joint --n_concepts 11 --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker_class_level --save_name "joint_sig-class-level-poker-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --lr 0.0398 --epochs 100 --attr_loss_weight 0.867 --use_sig --weighted_loss --seed $RAN_NUM
  python concept_to_standard_model_converter.py --model_path "./saves/joint_sig-class-level-poker-seed-$RAN_NUM.pth" --model_out_path "./saves/converted/joint_sig-class-level-poker-seed-$RAN_NUM.pth" --num_concepts 11 --num_classes 6 --end_to_end
done
