@ECHO OFF

python main.py --mode standard --n_classes 52 --dataset_root dataset/playing_cards

python main.py --mode XtoC --n_concepts 17 --dataset_root dataset/playing_cards  --weighted_loss

python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 16 --attr_loss_weight 0.4
python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 16 --attr_loss_weight 0.5
python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 16 --attr_loss_weight 0.8
python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 16 --attr_loss_weight 0.99

python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 1 --dataset_root dataset/three_card_poker_2 --num_workers 4 --batch_size 32 --attr_loss_weight 0.4
python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 1 --dataset_root dataset/three_card_poker_2 --num_workers 4 --batch_size 32 --attr_loss_weight 0.5`
python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 1 --dataset_root dataset/three_card_poker_2 --num_workers 4 --batch_size 32 --attr_loss_weight 0.8`
python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 1 --dataset_root dataset/three_card_poker_2 --num_workers 4 --batch_size 32 --attr_loss_weight 0.99`

python softmax_CBM/main.py --n_classes 6 --n_concepts 17 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 8 --attr_loss_weight 0.4
python softmax_CBM/main.py --n_classes 6 --n_concepts 17 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 8 --attr_loss_weight 0.5
python softmax_CBM/main.py --n_classes 6 --n_concepts 17 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 8 --attr_loss_weight 0.80
python softmax_CBM/main.py --n_classes 6 --n_concepts 17 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 8 --attr_loss_weight 0.99

PAUSE
