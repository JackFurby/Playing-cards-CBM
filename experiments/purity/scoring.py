import torch
from torch import nn
import numpy as np
import argparse
from torchmetrics import F1Score
from dataset.dataset import load_data, find_class_imbalance
from models.model import End2EndModel
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
from experiments.purity.purity_model import PredictorModel

import concepts_xai.evaluation.metrics.oracle as oracle
import concepts_xai.evaluation.metrics.niching as niche


def ImputityScore(x_to_c_model, data_loader, mode, device="cpu"):
	x_to_c_model.eval()

	c_soft = []
	c_true = []
	with torch.set_grad_enabled(False):
		for images, labels, ground_concepts in tqdm(data_loader, unit="batch"):
			# Transfering images and labels to GPU if available
			images, labels, ground_concepts = images.to(device), labels.to(device), ground_concepts.to(device)

			concept_representations = x_to_c_model(images)

			c_soft.append(concept_representations)
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
		return score
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
		'--use_cpu',
		action='store_true',
		help='Only use the system CPU, even if CUDA is available'
	)
	args = parser.parse_args()

	device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.use_cpu) else "cpu")
	if torch.cuda.is_available() and not args.use_cpu:
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)

	model_name = args.pretrained_weights.split("/")[-1][:-4]

	val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", 16, dataset_root=args.dataset_root, num_workers=1)

	model = End2EndModel(num_classes=args.n_classes, num_concepts=args.n_concepts, use_sigmoid=args.use_sig, use_modulelist=False)
	model.load_state_dict(torch.load(f"{args.pretrained_weights}", map_location=torch.device('cpu')))

	save_dir = f"./results/metrics/{model_name}"

	"""
	ois, pred_matrix, oracle_matrix = ImputityScore(model.x_to_c_model, val_loader, mode="oracle", device=device)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	with open(f"{save_dir}/{args.dataset_postfix.split('/')[-1]}_ois_score.txt", "w") as file:
		file.write(f"{ois}\n")

	for matrix_name in ["pred_matrix", "oracle_matrix"]:
		if matrix_name == "pred_matrix":
			matrix = pred_matrix
		else:
			matrix = oracle_matrix
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(matrix, interpolation='nearest', vmin=0.0, vmax=1.0)
		fig.colorbar(cax)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		plt.savefig(f'{save_dir}/{args.dataset_postfix.split("/")[-1]}_{matrix_name}.png')
	"""

	nis = ImputityScore(model.x_to_c_model, val_loader, mode="niche", device=device)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	with open(f"{save_dir}/{args.dataset_postfix.split('/')[-1]}_nis_score.txt", "w") as file:
		file.write(f"{nis}\n")
