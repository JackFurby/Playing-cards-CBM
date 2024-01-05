#!/bin/bash

python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/joint_0.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/joint_1.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/joint_2.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/joint_3.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq

python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/joint_0.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/joint_1.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/joint_2.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/joint_3.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg

python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_0.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_1.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_2.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_3.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_4.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg_sq

python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_0.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_1.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_2.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_3.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg
python experiments/chexpert_prop_relevance.py --pretrained_weights ./saves/chexpert/sequential_4.ckpt --dataset_root dataset/chexpert --dataset_postfix splits/cheXbert --n_classes 2 --n_concepts 13 --saliency_mode ig_sg
