# Playing cards CBM

This repository is for the IJCAI review process. This code will train models on the playing cards dataset and test models on the playing cards dataset and CheXpert. The playing cards dataset is not included for the review process but will be made available for the camera-ready paper.

If you do not have enough memory on your GPU (some experiments will need 24GB), the flag `--use_cpu` can be added to run experiments on the CPU. This will increase time required to run experiments significantly.

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

5. Setup datasets

    Datasets should be in the following file structure

    root
    |
    |--datasets
    |  |
    |  |--chexpert
    |  |  |
    |  |  |--CheXlocalize
    |  |  |  |
    |  |  |  |--gradcam_maps_val
    |  |  |  |  |
    |  |  |  |  |--patient*_study*_view*_*.pkl
    |  |  |  |  |
    |  |  |  |  ...
    |  |  |  |
    |  |  |  |--*.json
    |  |  |
    |  |  |--CheXpert-v1.0
    |  |  |  |
    |  |  |  |--val
    |  |  |  |  |
    |  |  |  |  |--patient*/study*/*.jpeg
    |  |  |  |  |
    |  |  |  |--test
    |  |  |  |  |
    |  |  |  |  |--patient*/study*/*.jpeg
    |  |  |  |  |
    |  |  |  |--train
    |  |  |  |  |
    |  |  |  |  |--patient*/study*/*.jpeg
    |  |  |  |  |
    |  |  |  |  ...
    |  |  |
    |  |  |--splits
    |  |  |  |
    |  |  |  |--cbm
    |  |  |  |  |
    |  |  |  |  |--train.pkl
    |  |  |  |  |
    |  |  |  |  |--test.pkl
    |  |  |  |  |
    |  |  |  |  |--val.pkl
    |  |  |  |  |
    |  |  |  |  |--classes.txt
    |  |  |  |  |
    |  |  |  |  |--concepts.txt
    |  |
    |  |--playing-cards
    |  |  |
    |  |  |--imgs
    |  |  |  |
    |  |  |  |--three
    |  |  |  |  |
    |  |  |  |  |--*.png
    |  |  |  |  |
    |  |  |  |  ...
    |  |  |  |
    |  |  |  |--three_card_poker
    |  |  |  |  |
    |  |  |  |  |--*.png
    |  |  |  |  |
    |  |  |  |  ...
    |  |  |  |
    |  |  |  |--three_card_poker_class_level
    |  |  |  |  |
    |  |  |  |  |--*.png
    |  |  |  |  |
    |  |  |  |  ...
    |  |  |  |
    |  |  |--splits
    |  |  |  |--three
    |  |  |  |  |
    |  |  |  |  |--train.pkl
    |  |  |  |  |
    |  |  |  |  |--val.pkl
    |  |  |  |  |
    |  |  |  |  |--classes.txt
    |  |  |  |  |
    |  |  |  |  |--concepts.txt
    |  |  |  |
    |  |  |  |--three_card_poker
    |  |  |  |  |
    |  |  |  |  |--train.pkl
    |  |  |  |  |
    |  |  |  |  |--val.pkl
    |  |  |  |  |
    |  |  |  |  |--classes.txt
    |  |  |  |  |
    |  |  |  |  |--concepts.txt
    |  |  |  |
    |  |  |  |--three_card_poker_class_level
    |  |  |  |  |
    |  |  |  |  |--train.pkl
    |  |  |  |  |
    |  |  |  |  |--val.pkl
    |  |  |  |  |
    |  |  |  |  |--classes.txt
    |  |  |  |  |
    |  |  |  |  |--concepts.txt

## Model training (optional)

Train all models: `sh train_models.sh`

### Example commands

#### XtoC training

`python main.py --mode XtoC --n_concepts 52 --dataset_root dataset/playing-cards  --dataset_postfix splits/three  --weighted_loss --optimizer sgd`

#### Independent CtoY training

`python main.py --mode CtoY --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker`


#### Sequential CtoY training

`python main.py --mode sequential --n_classes 6 --n_concepts 52 --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --num_workers 4 --batch_size 32 --pretrained_weights saves/XtoC_best_three.pth --freeze`

#### Joint training

`python main.py --mode joint --n_classes 52 --n_concepts 17 --dataset_root dataset/playing-cards --dataset_postfix splits/single  --num_workers 4 --batch_size 32 --attr_loss_weight 0.99`

## Model conversion

With a trained model the architecture will contain a module list layer containing linear layers. This can be converted to a single linear layer which will make it easier to index the concept vector. This is only required for newly trained playing card models.

### Example commands

#### XtoC

`python concept_to_standard_model_converter.py --model_path ./saves/XtoC_best.pth --model_out_path ./saves/XtoC_converted.pth --num_concepts 17 --vgg_version 11_bn`

#### XtoC and CtoY (independent)

`python concept_to_standard_model_converter.py --model_path ./saves/independent-XtoC-single.pth --model_2_path ./saves/independent-CtoY-single.pth --model_out_path ./saves/independent-single.pth --num_concepts 17 --num_classes 52 --vgg_version 11_bn`


#### XtoCtoY / end to end / XtoC and CtoY (sequential)

`python concept_to_standard_model_converter.py --model_path ./saves/XtoCtoY_best.pth --model_out_path ./saves/XtoCtoY_converted.pth --num_concepts 52 --num_classes 6 --end_to_end --vgg_version 11_bn`


## Experiments

Note: The commands listed below are examples. Some parts of the commands may need changeing e.g. model file names

### Model accuracies (playing card models only)

`python experiments/model_accuracy.py --weights_dir ./saves/converted --dataset_root dataset/playing-cards`

Note: CheXpert models can be tested with the notebook `chexpert_test.ipynb`

### saliency maps (samples and averages)

`python experiments/relevancy.py --pretrained_weights ./saves/converted/independent-poker-seed-782.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --saliency_mode lrp`

`python experiments/relevancy.py --pretrained_weights ./saves/converted/joint_sig-poker-seed-738.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --saliency_mode lrp`

`python experiments/relevancy.py --pretrained_weights ./saves/converted/sequential-three-seed-46.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --saliency_mode lrp`

`python experiments/relevancy.py --pretrained_weights ./saves/converted/independent-poker-seed-782.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --saliency_mode ig_sg`

`python experiments/relevancy.py --pretrained_weights ./saves/converted/joint_sig-poker-seed-738.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --saliency_mode ig_sg`

`python experiments/relevancy.py --pretrained_weights ./saves/converted/sequential-three-seed-46.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --saliency_mode ig_sg`

`python experiments/relevancy.py --pretrained_weights ./saves/converted/independent-poker-seed-782.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --saliency_mode ig_sg_sq`

`python experiments/relevancy.py --pretrained_weights ./saves/converted/joint_sig-poker-seed-738.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --saliency_mode ig_sg_sq`

`python experiments/relevancy.py --pretrained_weights ./saves/converted/sequential-three-seed-46.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --saliency_mode ig_sg_sq`

`python experiments/relevancy.py --pretrained_weights ./saves/chexpert/joint_4.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode gradCAM --dataset_file test --dataset_name chexpert --samples_per_class -1`

`python experiments/relevancy.py --pretrained_weights ./saves/chexpert/joint_4.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg --dataset_file test --dataset_name chexpert --samples_per_class -1`

`python experiments/relevancy.py --pretrained_weights ./saves/chexpert/joint_4.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq --dataset_file test --dataset_name chexpert --samples_per_class -1`

'python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/joint_4.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq'

`python experiments/relevancy.py --pretrained_weights ./saves/chexpert/sequential_class_0_3.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/class_level_0 --n_classes 2 --n_concepts 13 --saliency_mode gradCAM --dataset_file test --dataset_name chexpert --samples_per_class -1`

### ROAR


**Random:** `python experiments/roar/data_gen.py --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --dataset_split three_card_poker --n_classes 6 --n_concepts 52 --pretrained_weights saves/converted/independent-three-seed-950.pth`

`sh roar.sh`

'python experiments/roar/plot.py --weights_dir ./saves/roar'


### Purity (OIS)

#### setup

1. Clone `https://github.com/mateoespinosa/concept-quality`

2. Update `requiremetns.txt`

  ```
  matplotlib==3.7.1
  matplotlib-inline==0.1.6
  numpy==1.24.2
  scanpy==1.9.3
  scikit-learn==1.2.0
  scipy==1.9.1
  Pillow==10.2.0
  pandas==1.5.3
  h5py==3.8.0
  PyYAML==6.0
  sklearn==0.0
  setuptools==67.5.*
  tensorboard==2.11.2
  tensorboard-data-server==0.6.1
  tensorboard-plugin-wit==1.8.1
  tensorflow==2.11.0
  tensorflow-datasets==4.8.3
  tensorflow-estimator==2.11.0
  tensorflow-io-gcs-filesystem==0.31.0
  tensorflow-metadata==1.12.0
  tensorflow-probability==0.19.0
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

4. Update `requirements.txt`

  Some pip packages will need updating. This list may not be complete

  ```
  Pillow==10.1.*
  ```

5. Install `concepts_xai` from concept-quality directory: `python setup.py install`

#### Run experiments

Run OIS and NIS scoring: `./scoring.sh`

This will create a number of txt files containing ois results. To plot these open `plot.py` in `experiments/purity` and copy the values over.

##### Example commands

`python experiments/purity/scoring.py --pretrained_weights saves/converted/sequential-poker-seed-625.pth --dataset_root dataset/playing-cards --dataset_postfix splits/three_card_poker --n_classes 6 --n_concepts 52 --results_txt ./results/metrics/ois_poker.txt --mode oracle --dataset_name cards`

`python experiments/purity/scoring.py --pretrained_weights ./saves/chexpert/chexpert_0.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/CBM_5_obs --n_classes 2 --n_concepts 5 --results_txt ./results/metrics/ois_chexpert.txt --mode oracle --dataset_name chexpert`
