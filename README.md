# Playing cards CBM
## Setup

Note: These instructions were designed for Ubuntu 22.04. They should however be compatible with other operating systems with minor adjustments.

1. Create Conda environment

    `conda create -n playing_cards python=3.10`  

2. Activate Conda environment

    `conda activate playing_cards`  

3. Install dependencies

    `pip install -r requirements.txt`  

4. Install repo as a package

    `pip install -e .`  

5. Download and configure dataset

    a. Clone the playing cards dataset from https://huggingface.co/datasets/JackFurby/playing-cards into the directory `project_root/datasets/playing-cards`

## Arguments

```
options:
  -h, --help            show this help message and exit
  --lr LR               Starting learning rate (default: 0.001)
  --lr_patience LR_PATIENCE
                        Number of epochs with no improvement after which learning rate will be reduced (default: 10)
  --mode {XtoC,standard,joint,CtoY,sequential}
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
  --save_name SAVE_NAME
                        Name to save model file. If left as default a systematic file name will be used (default: "")

```

## Model training

This repository includes training methods for the main CBM training methods.

### Example commands

#### Standard (non-CBM training)

`python main.py --mode standard --n_classes 52 --dataset_root dataset/playing-cards --dataset_postfix splits/single`

`python main.py --mode standard --n_classes 6 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker`


#### XtoC training

`python main.py --mode XtoC --n_concepts 17 --dataset_root dataset/playing-cards  --dataset_postfix splits/single  --weighted_loss --optimizer sgd`

`python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/playing-cards  --dataset_postfix splits/three_card_class  --weighted_loss --optimizer sgd`

`python main.py --mode XtoC --n_concepts 36 --dataset_root dataset/celebA  --dataset_postfix splits/cbm  --weighted_loss --dataset_name faces --x_resize 178 --y_resize 218 --optimizer sgd --pretrained_weights saves/vgg11_bn-imagenet.pth --freeze`

`python main.py --mode XtoC --n_concepts 13 --dataset_root dataset/chexpert  --dataset_postfix splits/CBM  --weighted_loss --optimizer sgd --dataset_name chexpert --num_workers 6 --batch_size 32 --save_name chexpert_x_to_c --vgg_version 16_bn --lr 0.01 --x_resize 320 --y_resize 320`


#### Independent CtoY training

`python main.py --mode CtoY --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker`


#### Sequential CtoY training

`python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --num_workers 4 --batch_size 32 --pretrained_weights saves/XtoC_best_three.pth --freeze`

*Note: This training method requires pretrained_weights to be passes an XtoC model with a modulelist concept vector*

#### Joint training

`python main.py --mode joint --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --num_workers 4 --batch_size 32 --attr_loss_weight 0.5`

`python main.py --mode joint --n_classes 52 --n_concepts 17 --dataset_root dataset/playing-cards --dataset_postfix splits/single  --num_workers 4 --batch_size 32 --attr_loss_weight 0.99`

## Model conversion

With a trained model the architecture will contain a modlule list layer containing linear layers. This can be converted to a single linear layer which will make it easier to index the concept vector.

### Example commands

#### XtoC

`python concept_to_standard_model_converter.py --model_path ./saves/XtoC_best.pth --model_out_path ./saves/XtoC_converted.pth --num_concepts 17`


#### XtoC and CtoY (independent)

`python concept_to_standard_model_converter.py --model_path ./saves/independent-XtoC-single.pth --model_2_path ./saves/independent-CtoY-single.pth --model_out_path ./saves/independent-single.pth --num_concepts 17 --num_classes 52`


#### XtoCtoY / end to end / XtoC and CtoY (sequential)

`python concept_to_standard_model_converter.py --model_path ./saves/XtoCtoY_best.pth --model_out_path ./saves/XtoCtoY_converted.pth --num_concepts 52 --num_classes 6 --end_to_end`

''

## Experiments

### Model accuracies

`python experiments/model_accuracy.py --weights_dir ./saves/paper_weights --dataset_root dataset/playing-cards`

### saliency maps (samples and averages)

`python experiments/relevancy.py --pretrained_weights ./saves/paper_weights/independent-single-seed-557.pth --dataset_root dataset/playing-cards --dataset_postfix splits/single --n_classes 52 --n_concepts 17`

`python experiments/relevancy.py --pretrained_weights ./saves/converted/independent-poker-seed-782.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52`

### ROAR


**Random:** `python experiments/roar/data_gen.py --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --dataset_split three_card_poker --n_classes 6 --n_concepts 52 --pretrained_weights saves/converted/independent-three-seed-950.pth`

`sh roar.sh`

'python experiments/roar/plot.py --weights_dir ./saves/roar --dataset_root dataset/roar/random/0.1'


### Purity (OIS and NIS)

#### setup

1. Clone `https://github.com/mateoespinosa/concept-quality`

2. Update `requiremetns.txt`

  ```
  matplotlib==3.6.*
  matplotlib-inline==0.1.*
  numpy==1.23.*
  scanpy==1.9.*
  scikit-learn==1.2.*
  scipy==1.10.*
  Pillow==9.4.*
  pandas==1.5.*
  h5py==3.8.*
  PyYAML==6.0.*
  sklearn==0.0
  setuptools==67.5.*
  tensorboard==2.12.*
  tensorboard-data-server==0.7.*
  tensorboard-plugin-wit==1.8.*
  tensorflow==2.12.*
  tensorflow-datasets==4.8.*
  tensorflow-estimator==2.12.*
  tensorflow-io-gcs-filesystem==0.32.*
  tensorflow-metadata==1.12.*
  tensorflow-probability==0.19.*
  ```

3. Update `setup.py`

  ```py
  from setuptools import setup, find_packages

  with open("README.md", "r") as f:
      long_description = f.read()

  with open('requirements.txt') as f:
      required = f.read().splitlines()

  version = '0.1.0'
  setup(
      name='concepts_xai',
      version=version,
      packages=find_packages(),
      install_requires=required,
      description='Concept Extraction Comparison',
      long_description=long_description,
      long_description_content_type="text/markdown",
      python_requires='>=3.6',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache-2.0 License",
          "Operating System :: OS Independent",
      ],
  )
  ```

4. Install `concepts_xai` from concept-quality directory: `python setup.py install`

#### Run experiments

`python experiments/purity/scoring.py --pretrained_weights saves/converted/sequential-poker-seed-625.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --results_txt ./results/metrics/ois_poker.txt --mode oracle`

`./scoring.sh`
