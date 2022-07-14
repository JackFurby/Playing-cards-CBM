# neuro-symbolic-Concept-Bottleneck-Models
Concept Bottleneck Models, possibly using neural-symbolic AI


# Single card dataset

`python main.py --mode XtoC --n_concepts 17 --dataset_root dataset/playing_cards  --weighted_loss`

# single card classification

`python main.py --mode standard --n_classes 52 --dataset_root dataset/playing_cards --has_class`

# multi card dataset

`python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/multi_playing_cards --weighted_loss`

# Joint

`python main.py --mode joint --n_concepts 52 --dataset_root dataset/multi_playing_cards --batch_size 32 num_workers 4`

# concept concept model to regular model architecture

`python concept_to_standard_model_converter.py --model_path ./saves/XtoC_best.pth --model_out_path ./saves/XtoC_converted.pth --num_concepts 17`
