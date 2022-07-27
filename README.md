# neuro-symbolic-Concept-Bottleneck-Models
Concept Bottleneck Models, possibly using neural-symbolic AI


# Single card dataset

`python main.py --mode XtoC --n_concepts 17 --dataset_root dataset/playing_cards  --weighted_loss`

# single card classification

`python main.py --mode standard --n_classes 52 --dataset_root dataset/playing_cards --has_class`

# multi card dataset

`python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/multi_playing_cards --weighted_loss`

# Joint

`python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 32 --attr_loss_weight 0.5`

# softmax Joint

`python softmax_CBM/main.py --n_classes 6 --n_concepts 52 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 8 --attr_loss_weight 0.5`

# C to Y

`python main.py --mode CtoY --n_classes 6 --n_concepts 52 --dataset_root dataset/individual_three_card_poker --num_images_per_output 3`

# concept concept model to regular model architecture (X to C)

`python concept_to_standard_model_converter.py --model_path ./saves/XtoC_best.pth --model_out_path ./saves/XtoC_converted.pth --num_concepts 17`

# concept concept model to regular model architecture (X to C to Y / end to end)

`python concept_to_standard_model_converter.py --model_path ./saves/XtoC_best.pth --model_out_path ./saves/XtoC_converted.pth --num_concepts 17`
