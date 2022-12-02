# single playing card images

# # Standard training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode standard --n_classes 52 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "standard-single-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.05 --epochs 200 --seed $RAN_NUM
done

# # Independent training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 17 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "independent-XtoC-single-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 10 --lr 0.02 --epochs 200 --weighted_loss --seed $RAN_NUM
  python main.py --mode CtoY --n_concepts 17 --n_classes 52 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "independent-CtoY-single-seed-$RAN_NUM" --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.05 --epochs 200 --seed $RAN_NUM
done

# # Sequential training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 17 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "sequential-XtoC-single-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 10 --lr 0.02 --epochs 200 --weighted_loss --seed $RAN_NUM
  python main.py --mode sequential --n_concepts 17 --n_classes 52 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "sequential-CtoY-single-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 3 --lr 0.0025 --epochs 100 --pretrained_weights "./saves/sequential-XtoC-single-seed-$RAN_NUM.pth" --freeze --seed $RAN_NUM
done

# # Joint training (no sig)

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode joint --n_classes 52 --n_concepts 17 --dataset_root dataset/playing-cards --dataset_postfix splits/single --save_name "joint-single-seed-$RAN_NUM" --lr 0.01 --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --weighted_loss --attr_loss_weight 0.643 --epochs 200 --seed $RAN_NUM
done



# poker playing card images

# # Standard training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode standard --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "standard-poker-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 10 --lr 0.05 --epochs 200 --seed $RAN_NUM
done

# # Independent training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "independent-XtoC-poker-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --lr 0.02 --epochs 200 --weighted_loss --seed $RAN_NUM
  python main.py --mode CtoY --n_concepts 52 --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "independent-CtoY-poker-seed-$RAN_NUM" --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.01 --epochs 200 --seed $RAN_NUM
done

# # Sequential training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-XtoC-poker-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss --seed $RAN_NUM
  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-poker-seed-$RAN_NUM"  --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --lr 0.01 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-poker-seed-$RAN_NUM.pth" --freeze --seed $RAN_NUM
done

# # Joint training (no sig)

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode joint --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "joint-poker-seed-$RAN_NUM" --lr 0.025 --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --weighted_loss --attr_loss_weight 0.98 --epochs 300 --seed $RAN_NUM
done

# # Joint training (with sig)

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode joint --n_concepts 52 --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "joint_sig-poker-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 15 --lr 0.05 --epochs 300 --attr_loss_weight 0.96 --use_sig --weighted_loss --seed $RAN_NUM
done


# three / poker playing card images

# # Independent training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three --save_name "independent-XtoC-three-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss --seed $RAN_NUM
  python main.py --mode CtoY --n_concepts 52 --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "independent-CtoY-three-seed-$RAN_NUM" --optimizer adam --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.01 --epochs 200 --seed $RAN_NUM
done

# # Sequential training

for i in 1 2 3 4 5
do
  RAN_NUM=$(shuf -i 0-1000 -n1)
  python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three --save_name "sequential-XtoC-three-seed-$RAN_NUM" --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.03 --epochs 200 --weighted_loss --seed $RAN_NUM
  python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --save_name "sequential-CtoY-three-seed-$RAN_NUM"  --optimizer sgd --num_workers 4 --batch_size 32 --lr_patience 5 --lr 0.01 --epochs 200 --pretrained_weights "./saves/sequential-XtoC-three-seed-$RAN_NUM.pth" --freeze --seed $RAN_NUM
done
