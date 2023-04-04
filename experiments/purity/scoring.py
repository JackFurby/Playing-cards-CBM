import torch
from torch import nn
import numpy as np
import argparse
from torchmetrics import F1Score
from dataset.dataset import load_data, find_class_imbalance
#from models.model import End2EndModel
from models.vgg import End2EndModel
from models.cub_model import ModelXtoCtoY as CUB_End2EndModel
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
import glob
import gc

import concepts_xai.evaluation.metrics.oracle as oracle
import concepts_xai.evaluation.metrics.niching as niche


def ImputityScore(x_to_c_model, data_loader, mode):
	x_to_c_model.eval()

	c_soft = []
	c_true = []
	with torch.set_grad_enabled(False):
		for images, labels, ground_concepts in tqdm(data_loader, unit="batch"):
			# Transfering images and labels to GPU if available
			images, labels, ground_concepts = images, labels, ground_concepts

			concept_representations = x_to_c_model(images)

			sigmoid_outputs = torch.nn.Sigmoid()(concept_representations.view(concept_representations.size(0), -1))

			#c_soft.append(concept_representations)
			c_soft.append(sigmoid_outputs)
			c_true.append(ground_concepts)

	if mode == "oracle":
		score, pred_matrix, oracle_matrix = oracle.oracle_impurity_score(
			c_soft=np.stack(np.concatenate(c_soft, axis=0)),
			c_true=np.stack(np.concatenate(c_true, axis=0)),
			output_matrices=True
		)
		#score = oracle.oracle_impurity_score(
		#	c_soft=np.stack(np.concatenate(c_soft, axis=0)),
		#	c_true=np.stack(np.concatenate(c_true, axis=0))
		#)
		return score, pred_matrix, oracle_matrix
		#return score
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
		'--multi_runs',
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
		val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", 16, dataset_root=args.dataset_root, num_workers=1, dataset_name="cub")
	else:
		val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", 16, dataset_root=args.dataset_root, num_workers=1)

	# generate results for multiple model weights and save to txt file
	if args.multi_runs:

		save_dir = f"./results/metrics/"
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		# get all models trained on poker cards

		model_list = glob.glob('saves/converted/*sequential*poker*.pth')

		ois_scores = []
		for model_file in model_list:
			model = End2EndModel(num_classes=args.n_classes, num_concepts=args.n_concepts, use_sigmoid=args.use_sig, use_modulelist=False)
			model.load_state_dict(torch.load(f"{model_file}", map_location=torch.device('cpu')))

			ois, pred_matrix, oracle_matrix = ImputityScore(model.x_to_c_model, val_loader, mode=args.mode)
			ois_scores.append((model_file, ois))

			if pred_matrix == None:
				pass
			else:
				for matrix_name in ["pred_matrix", "oracle_matrix"]:
					if matrix_name == "pred_matrix":
						matrix = pred_matrix
					else:
						matrix = oracle_matrix
					fig = plt.figure()
					ax = fig.add_subplot(111)
					cax = ax.matshow(matrix, interpolation='nearest', vmin=0.0, vmax=1.0)
					fig.colorbar(cax)

					if not os.path.exists(f"save_dir/{model_name}"):
						os.makedirs(f"save_dir/{model_name}")
					plt.savefig(f'{save_dir}/{model_name}/{args.dataset_postfix.split("/")[-1]}_{matrix_name}.png')

			gc.collect()

		model_list = glob.glob('saves/converted/*independent*poker*.pth')

		for model_file in model_list:
			model = End2EndModel(num_classes=args.n_classes, num_concepts=args.n_concepts, use_sigmoid=args.use_sig, use_modulelist=False)
			model.load_state_dict(torch.load(f"{model_file}", map_location=torch.device('cpu')))

			ois, pred_matrix, oracle_matrix = ImputityScore(model.x_to_c_model, val_loader, mode=args.mode)
			ois_scores.append((model_file, ois))

			if pred_matrix == None:
				pass
			else:
				for matrix_name in ["pred_matrix", "oracle_matrix"]:
					if matrix_name == "pred_matrix":
						matrix = pred_matrix
					else:
						matrix = oracle_matrix
					fig = plt.figure()
					ax = fig.add_subplot(111)
					cax = ax.matshow(matrix, interpolation='nearest', vmin=0.0, vmax=1.0)
					fig.colorbar(cax)

					if not os.path.exists(f"save_dir/{model_name}"):
						os.makedirs(f"save_dir/{model_name}")
					plt.savefig(f'{save_dir}/{model_name}/{args.dataset_postfix.split("/")[-1]}_{matrix_name}.png')

			gc.collect()

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		with open(f"{save_dir}/poker_models_ois_scores.txt", "w") as file:
			for score in ois_scores:
				concepts.write(f"{score[0]} {score[1]}\n")

		# get all models trained on three random cards

		model_list = glob.glob('saves/converted/*three*.pth')

		ois_scores = []
		for model_file in model_list:
			model = End2EndModel(num_classes=args.n_classes, num_concepts=args.n_concepts, use_sigmoid=args.use_sig, use_modulelist=False)
			model.load_state_dict(torch.load(f"{model_file}", map_location=torch.device('cpu')))

			ois, pred_matrix, oracle_matrix = ImputityScore(model.x_to_c_model, val_loader, mode=args.mode, device=device)
			ois_scores.append((model_file, ois))

			if pred_matrix == None:
				pass
			else:
				for matrix_name in ["pred_matrix", "oracle_matrix"]:
					if matrix_name == "pred_matrix":
						matrix = pred_matrix
					else:
						matrix = oracle_matrix
					fig = plt.figure()
					ax = fig.add_subplot(111)
					cax = ax.matshow(matrix, interpolation='nearest', vmin=0.0, vmax=1.0)
					fig.colorbar(cax)

					if not os.path.exists(f"save_dir/{model_name}"):
						os.makedirs(f"save_dir/{model_name}")
					plt.savefig(f'{save_dir}/{model_name}/{args.dataset_postfix.split("/")[-1]}_{matrix_name}.png')

			gc.collect()

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		with open(f"{save_dir}/three_models_ois_scores.txt", "w") as file:
			for score in ois_scores:
				concepts.write(f"{score[0]} {score[1]}\n")

	# generate results for a single weights file and append results to txt file
	else:

		model_name = args.pretrained_weights.split("/")[-1][:-4]
		if "CUB" in args.dataset_root:
			model = CUB_End2EndModel(n_class_attr=2, num_classes=args.n_classes, n_attributes=args.n_concepts, expand_dim=0, use_relu=False, use_sigmoid=args.use_sig, train=True)
		else:
			model = End2EndModel(num_classes=args.n_classes, num_concepts=args.n_concepts, use_sigmoid=args.use_sig, use_modulelist=False)
		model.load_state_dict(torch.load(args.pretrained_weights, map_location=torch.device('cpu')))
		model.eval()

		if "CUB" in args.dataset_root:
			x_to_c_model = model.first_model
		else:
			x_to_c_model = model.x_to_c_model

		score, pred_matrix, oracle_matrix = ImputityScore(x_to_c_model, val_loader, mode=args.mode)
		with open(args.results_txt, "a") as f:
			f.write(f"{model_name} {score}\n")

		save_dir = f"./results/metrics/{model_name}"
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		if pred_matrix == None:
			pass
		else:
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
