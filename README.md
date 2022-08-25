# neuro-symbolic-Concept-Bottleneck-Models

Concept Bottleneck Models, possibly using neural-symbolic AI

## Setup

Note: These instructions were designed for Ubuntu 22.04. They should however be compatible with other operating systems with minor adjustments.

1. Create Conda environment

    `conda create -n playing_cards python=3.10`  

2. Activate Conda environment

    `conda activate playing_cards`  

3. Install dependencies

    `pip install -r requirements.txt`  

4. Download and configure dataset

    a. Clone the playing cards dataset from https://huggingface.co/datasets/JackFurby/playing-cards into the directory `project_root/datasets/playing-cards`

## Arguments

```
options:
  -h, --help            show this help message and exit
  --lr LR               Starting learning rate (default: 0.001)
  --lr_patience LR_PATIENCE
                        Number of epochs with no improvement after which learning rate will be reduced (default: 10)
  --mode {XtoC,standard,joint,CtoY}
                        Training method to use for training a model (default: XtoC)
  --epochs EPOCHS       Number of epochs to train a model (default: 100)
  --batch_size BATCH_SIZE
                        Number of samples per batch (default: 16)
  --seed SEED           Seed set to Pytorch (default: 0)
  --num_workers NUM_WORKERS
                        Number of workers fetching data for the dataloader (default: 0)
  --optimizer {adam,sgd}
                        Optimizer to use for training (default: adam)
  --pretrained_weights PRETRAINED_WEIGHTS
                        File path to pytorch weight file
  --n_classes N_CLASSES
                        Number of classes in dataset
  --n_concepts N_CONCEPTS
                        Number of concepts/attributes in dataset
  --num_images_per_output NUM_IMAGES_PER_OUTPUT
                        Number of images input for each output (only used for joint training) (default: 1)
  --dataset_root DATASET_ROOT
                        Path to dataset root directory
  --dataset_postfix DATASET_POSTFIX
                        Path to dataset split files from dataset root (default: "")
  --weighted_loss       Whether to use weighted loss for single concept or multiple ones
  --attr_loss_weight ATTR_LOSS_WEIGHT
                        Weight task loss (default: 1.0)
  --freeze              Freeze the input to layer before concept vector model part
  --use_sig             Use sigmoid between the XtoC and the CtoY model parts
  --use_cpu             Only use the system CPU, even if CUDA is available
```

## Model training

This repository includes training methods for the main CBM training methods.

### Example commands

#### Standard (non-CBM training)

`python main.py --mode standard --n_classes 52 --dataset_root dataset/playing-cards --dataset_postfix splits/single`

`python main.py --mode standard --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker_card_class`


#### XtoC training

`python main.py --mode XtoC --n_concepts 17 --dataset_root dataset/playing_cards  --dataset_postfix splits/single  --weighted_loss`

`python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/playing_cards  --dataset_postfix splits/three_card_class  --weighted_loss`


#### Independent CtoY training

`python main.py --mode CtoY --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker_card_class`


#### Sequential CtoY training

TO DO


#### Joint training

`python main.py --mode joint --n_classes 6 --n_concepts 52 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 32 --attr_loss_weight 0.5`  Note: This training configuration is not supported with the current dataset

`python main.py --mode joint --n_classes 52 --n_concepts 17 --num_images_per_output 1 --dataset_root dataset/playing-cards --dataset_postfix splits/single  --num_workers 4 --batch_size 32 --attr_loss_weight 0.99`


#### softmax Joint

`python softmax_CBM/main.py --n_classes 6 --n_concepts 52 --num_images_per_output 3 --dataset_root dataset/individual_three_card_poker --num_workers 4 --batch_size 8 --attr_loss_weight 0.5`  Note: This training configuration is not supported with the current dataset


## Model conversion

With a trained model the architecture will contain a modlule list layer containing linear layers. This can be converted to a single linear layer which will make it easier to index the concept vector.

### Example commands

#### XtoC

`python concept_to_standard_model_converter.py --model_path ./saves/XtoC_best.pth --model_out_path ./saves/XtoC_converted.pth --num_concepts 17`


#### XtoC and CtoY (independent)

TO DO


#### XtoCtoY / end to end / XtoC and CtoY (sequential)

`python concept_to_standard_model_converter.py --model_path ./saves/XtoC_best.pth --model_out_path ./saves/XtoC_converted.pth --num_concepts 17`
