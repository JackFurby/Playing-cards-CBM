import torch
from torch import nn
import numpy as np
import argparse
from torchmetrics import F1Score
from dataset.dataset import load_data, find_class_imbalance
from models.model import End2EndModel, VGG
import models.chexpert.classifier as classifier
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
import glob
import gc
from sklearn.model_selection import train_test_split

#import concepts_xai.evaluation.metrics.oracle as oracle
import experiments.purity.oracle as oracle
import concepts_xai.evaluation.metrics.niching as niche


def ImputityScore(x_to_c_model, data_loader, mode, dataset, set_splits=False):
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
			if dataset == "chexpert":
				#print(ground_concepts)
				#sigmoid_outputs = torch.cat([sigmoid_outputs[0:11], sigmoid_outputs[11+1:]], axis=1)
				#ground_concepts = torch.cat([ground_concepts[0:11], ground_concepts[11+1:]], axis=1)
				#print(ground_concepts[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]])

				#sigmoid_outputs = sigmoid_outputs[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]]
				#ground_concepts = ground_concepts[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]]
				pass
			c_soft.append(sigmoid_outputs)
			c_true.append(ground_concepts)

	if mode == "oracle":
		c_soft = np.stack(np.concatenate(c_soft, axis=0))
		c_true = np.stack(np.concatenate(c_true, axis=0))

		score, pred_matrix, oracle_matrix = oracle.oracle_impurity_score(
			c_soft=c_soft,
			c_true=c_true,
			output_matrices=True
		)
		return score, pred_matrix, oracle_matrix
	elif mode == "niche":

		# chexpert has two concepts with a single occurance. This breaks niche_imputiry_score
		# as they use a AUC score. To fix this we have to set the train splits manually and
		# duplicate the problematic samples so they always occure in the test split.
		c_soft=np.stack(np.concatenate(c_soft, axis=0))
		c_true=np.stack(np.concatenate(c_true, axis=0))
		if dataset == "chexpert" and set_splits:
			c_soft_train, c_soft_test, c_true_train, c_true_test = train_test_split(
				c_soft,
				c_true,
				test_size=0.2,
			)
			c_true_test = np.vstack([c_true_test, c_true[109], c_true[113]])
		else:
			c_soft_train=None
			c_true_train=None

		score = niche.niche_impurity_score(
			c_soft=c_soft,
			c_true=c_true,
			c_soft_train=c_soft_train,
			c_true_train=c_true_train
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
		'--set_splits',
		action='store_true',
		help='Use if using the instance-level version oc CheXpert'
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
		vgg_model = cub_model.vgg16_bn(pretrained=False, num_classes=args.n_classes, n_attributes=args.n_concepts, bottleneck=True, expand_dim=False, train=True)
		model1 = cub_model.x_to_c_model(freeze=False, model=vgg_model)
		model2 = cub_model.MLP(input_dim=args.n_concepts, num_classes=args.n_classes, expand_dim=False, train=True)
		model = cub_model.End2EndModel(model1, model2, False, True)

		model.load_state_dict(torch.load(args.pretrained_weights, map_location=torch.device('cpu')))

	elif "chexpert" in args.dataset_root:
		cfg = {
			"backbone": "densenet121",
			"norm_type": "BatchNorm",
			"pretrained": True,
			"num_concepts": args.n_concepts,
			"cbm_mode": "joint",  # can be either xtoc or joint (joint is also used for sequential)
			"xtoc_only": True  # if true then only predict concepts
		}

		model = classifier.Classifier(cfg)
		model.load_state_dict(torch.load(args.pretrained_weights, map_location=torch.device('cpu'))['state_dict'], strict=True)
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

	score, pred_matrix, oracle_matrix = ImputityScore(x_to_c_model, val_loader, mode=args.mode, dataset=args.dataset_name, set_splits=args.set_splits)
	with open(args.results_txt, "a") as f:
		f.write(f"{model_name} {score}\n")

	if type(pred_matrix) == np.ndarray:
		save_dir = f"./results/metrics/{model_name}_{len(os.listdir('./results/metrics/'))}"
		print(save_dir)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		for matrix_name in ["pred_matrix", "oracle_matrix"]:
			if matrix_name == "pred_matrix":
				matrix = pred_matrix
			else:
				matrix = oracle_matrix
			fig = plt.figure()
			ax = fig.add_subplot(111)
			#cax = ax.matshow(matrix, interpolation='nearest', vmin=0.0, vmax=1.0)
			cax = ax.matshow(matrix, interpolation='nearest')
			fig.colorbar(cax)
			plt.savefig(f'{save_dir}/{args.dataset_postfix.split("/")[-1]}_{matrix_name}.png')

		# save matrix
		with open(f'{save_dir}/pred_matrix.npy', 'wb') as f:
			np.save(f, pred_matrix)

		with open(f'{save_dir}/oracle_matrix.npy', 'wb') as f:
			np.save(f, oracle_matrix)
