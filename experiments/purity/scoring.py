import torch
from torch import nn
import numpy as np
import argparse
from torchmetrics import F1Score
from dataset.dataset import load_data, find_class_imbalance
from models.model import End2EndModel
import models.chexpert.classifier as classifier
from easydict import EasyDict as edict
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
import glob
import gc

import concepts_xai.evaluation.metrics.oracle as oracle
import concepts_xai.evaluation.metrics.niching as niche


def ImputityScore(x_to_c_model, data_loader, mode, dataset):
	x_to_c_model.eval()

	c_soft = []
	c_true = []
	with torch.set_grad_enabled(False):
		for images, labels, ground_concepts in tqdm(data_loader, unit="batch"):


			concept_representations = x_to_c_model(images)

			sigmoid_outputs = torch.nn.Sigmoid()(concept_representations.view(concept_representations.size(0), -1))

			#print(concept_representations)
			#print(sigmoid_outputs)
			#print(ground_concepts)

			#c_soft.append(concept_representations)
			c_soft.append(sigmoid_outputs)
			c_true.append(ground_concepts)

	if mode == "oracle":
		score, pred_matrix, oracle_matrix = oracle.oracle_impurity_score(
			c_soft=np.stack(np.concatenate(c_soft, axis=0)),
			c_true=np.stack(np.concatenate(c_true, axis=0)),
			output_matrices=True
		)
		return score, pred_matrix, oracle_matrix
	elif mode == "niche":
		score = niche.niche_impurity_score(
			c_soft=np.stack(np.concatenate(c_soft, axis=0)),
			c_true=np.stack(np.concatenate(c_true, axis=0))
		)
		return score, None, None
	else:
		raise Exception(f"Mode has to have the value 'oracle' or 'niche'")


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Evaluate a Concept Bottleneck Model')
	parser.add_argument(
		'--pretrained_weights',
		type=str,
		help='File path to pytorch weight file'
	)
	parser.add_argument(
		'--dataset_root',
		type=str,
		help='Path to dataset root directory'
	)
	parser.add_argument(
		'--dataset_postfix',
		type=str,
		default="",
		help='Path to dataset split files from dataset root (default: "")'
	)
	parser.add_argument(
		'--dataset_name',
		type=str,
		default='cards',
		choices=["cards", "chexpert", "CUB"],
		help='Name of dataset (default: "cards")'
	)
	parser.add_argument(
		'--n_classes',
		type=int,
		help='Number of classes in dataset'
	)
	parser.add_argument(
		'--n_concepts',
		type=int,
		help='Number of concepts/attributes in dataset'
	)
	parser.add_argument(
		'--use_sig',
		action='store_true',
		help='Use sigmoid between the XtoC and the CtoY model parts'
	)
	parser.add_argument(
		'--results_txt',
		type=str,
		default="",
		help='Path to txt file to append results to (default: "")'
	)
	parser.add_argument(
		'--mode',
		type=str,
		choices=["oracle", "niche"],
		help='Type of score to generate'
	)
	args = parser.parse_args()

	if "CUB" in args.dataset_root:
		val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", 16, dataset_root=args.dataset_root, dataset_name="CUB", num_workers=1, x_resize=299, y_resize=299)
	elif "chexpert" in args.dataset_root:
		val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/test.pkl", 16, dataset_root=args.dataset_root, dataset_name="chexpert", num_workers=1, x_resize=512, y_resize=512)
	else:
		val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", 16, dataset_root=args.dataset_root, dataset_name="cards", num_workers=1, x_resize=299, y_resize=299)

	model_name = args.pretrained_weights.split("/")[-1][:-4]
	if "CUB" in args.dataset_root:
		model = End2EndModel(vgg_version="19_bn", num_classes=args.n_classes, num_concepts=args.n_concepts, use_sigmoid=args.use_sig, use_modulelist=False)
		model.load_state_dict(torch.load(args.pretrained_weights, map_location=torch.device('cpu')))
	elif "chexpert" in args.dataset_root:
		cfg = edict({
			"backbone": "densenet121",
			"norm_type": "BatchNorm",
			"pretrained": True,
			"lse_gamma": 0.5,
			"global_pool": "AVG_MAX",
			"num_classes": [1,1,1,1,1],
			"attention_map": "FPA",
			"fc_bn": True,
			"fc_drop": 0,
		})

		model = classifier.Classifier(cfg)
		model.load_state_dict(torch.load(args.pretrained_weights, map_location=torch.device('cpu'))['state_dict'], strict=False)
	else:
		if "sig" in args.pretrained_weights:  # joint with sig
			use_sig = True
		elif "standard" in args.pretrained_weights:  # standard as joint
			use_sig = False
		elif "joint" not in args.pretrained_weights:  # independent and sequential
			use_sig = True
		else:  # joint without sig
			use_sig = False

		model = End2EndModel(model_version="11_bn", num_classes=args.n_classes, num_concepts=args.n_concepts, use_sigmoid=use_sig, use_modulelist=False)
		model.load_state_dict(torch.load(args.pretrained_weights, map_location=torch.device('cpu')))

	if "CUB" in args.dataset_root:
		x_to_c_model = model.first_model
	elif "chexpert" in args.dataset_root:
		x_to_c_model = model
	else:
		x_to_c_model = model.x_to_c_model

	score, pred_matrix, oracle_matrix = ImputityScore(x_to_c_model, val_loader, mode=args.mode, dataset=args.dataset_name)
	with open(args.results_txt, "a") as f:
		f.write(f"{model_name} {score}\n")

	save_dir = f"./results/metrics/{model_name}"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	if type(pred_matrix) == np.ndarray:
		for matrix_name in ["pred_matrix", "oracle_matrix"]:
			if matrix_name == "pred_matrix":
				matrix = pred_matrix
			else:
				matrix = oracle_matrix
			fig = plt.figure()
			ax = fig.add_subplot(111)
			cax = ax.matshow(matrix, interpolation='nearest', vmin=0.0, vmax=1.0)
			fig.colorbar(cax)
			plt.savefig(f'{save_dir}/{args.dataset_postfix.split("/")[-1]}_{matrix_name}.png')
