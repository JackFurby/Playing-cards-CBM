import torch
from torch import nn
import numpy as np
import argparse
from torchmetrics import F1Score
from dataset.dataset import load_data
from models.model import End2EndModel
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
from experiments.purity.purity_model import PurityModel


def train(x_to_c_model, purity_model, data_loader, optimizer, criterion, train=True, device="cpu"):
	"""
	Training and evaluation loop.

		args:
			x_to_c_model (Pytorch model): X to C Pytorch model
			purity_model (Pytorch model): Pytorch model to train / evaluate
			data_loader (Pytorch dataloader): Pytorch dataloader with dataset Initilised
			optimizer (torch.optim): Optimiser for updating model weights
			criterion (pytorch loss / list of Pytorch loss): Loss method
			train (bool): True if model is training, False if model is testing
			device (str): Device top train / test using (e.g. cpu or cuda:0)

		returns:
			average loss: Average loss of model
			average accuracy: Average accuracy of concepts / task (non joint)
	"""

	if train:
		purity_model.train()
	else:
		purity_model.eval()

	losses = []
	accuracy = []
	task_acc = []
	f1_acc = []
	with torch.set_grad_enabled(train):
		for images, labels, ground_concepts in tqdm(data_loader, unit="batch"):
			# Transfering images and labels to GPU if available
			images, labels, ground_concepts = images.to(device), labels.to(device), ground_concepts.to(device)

			if train:
				# Zero the parameter gradients
				optimizer.zero_grad()

			one_hot = nn.functional.one_hot(torch.arange(0, ground_concepts.size(1)), num_classes=ground_concepts.size(1)).to(device)

			pred_concepts = x_to_c_model(images)

			concepts_in = []

			for i in range(pred_concepts.size(0)):
				concepts_in.append(torch.mul(pred_concepts[i], one_hot))

			# x_to_c_model concept predictions are turned into one hot like vectors
			# each concept prediction is in an array of size (0, num_concepts) where all but that concept has the value of 0
			# the new batch size for purity_model training is original batch size * num_concepts
			concepts_in = torch.cat(concepts_in)

			# repeate concepts to use as lables for concepts_in
			target_concepts = torch.repeat_interleave(pred_concepts.detach(), concepts_in.size(1), dim=0)

			outputs = purity_model(concepts_in)

			batch_losses = []
			for i in range(len(criterion)):
				batch_losses.append(criterion[i](outputs[:, i], target_concepts[:, i]))

			loss = sum(batch_losses) / len(batch_losses)
			#sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))

			# accuracy for concepts
			accuracy.append(binary_accuracy(outputs, target_concepts).item())  # NOT SURE IF THIS IS THE RIGHT METRIC TO USE

			losses.append(loss.item())
			if train:
				loss.backward()

				# Optimizing the parameters
				optimizer.step()

		acc = sum(accuracy) / len(accuracy)

	return sum(losses) / len(losses), acc


def binary_accuracy(output, target):
	"""
	Computes the accuracy for multiple binary predictions.

	args:
		output (Pytorch tensor): Model prediction
		target (Pytorch tensor): Ground truth from dataset
	returns:
		acc: binary accuracy
	"""
	pred = output >= 0.5
	target = target >= 0.5
	acc = (pred.int()).eq(target.int()).sum()
	acc = acc * 100 / np.prod(np.array(target.size()))
	return acc


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Evaluate a Concept Bottleneck Model')
	parser.add_argument(
		'--lr',
		type=float,
		default=0.001,
		help='Starting learning rate (default: 0.001)'
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=16,
		help='Number of samples per batch (default: 16)'
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=0,
		help='Seed set to Pytorch (default: 0)'
	)
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
		'--weighted_loss',
		action='store_true',
		help='Whether to use weighted loss for single concept or multiple ones'
	)
	parser.add_argument(
		'--freeze',
		action='store_true',
		help='Freeze the input to layer before concept vector model part'
	)
	parser.add_argument(
		'--mode',
		type=str,
		default="XtoC",
		choices=["OIS", "NIS"],
		help='Evauation metric'
	)
	parser.add_argument(
		'--use_sig',
		action='store_true',
		help='Use sigmoid between the XtoC and the CtoY model parts'
	)
	parser.add_argument(
		'--optimizer',
		type=str,
		default='adam',
		choices=["adam", "sgd"],
		help='Optimizer to use for training (default: adam)'
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

	torch.manual_seed(args.seed)  # reproducibility

	model_name = args.pretrained_weights.split("/")[-1][:-4]
	sig_string = "_sig" if args.use_sig else ""
	save_name = f"{model_name}_{args.mode}_model.pth"

	train_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/train.pkl", args.batch_size, dataset_root=args.dataset_root, resampling=True, num_workers=1)
	val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", args.batch_size, dataset_root=args.dataset_root, num_workers=1)

	model = End2EndModel(num_classes=args.n_classes, num_concepts=args.n_concepts, use_sigmoid=args.use_sig, use_modulelist=False)
	model.load_state_dict(torch.load(f"{args.pretrained_weights}", map_location=torch.device('cpu')))

	if args.optimizer == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	else:
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

	# we do not want to change the CBM weights.
	if args.freeze:
		for name, param in model.named_parameters():
			param.requires_grad = False
		print("model layers frozen")

	model.eval()
	model.to(device)

	purity_model = PurityModel(num_concepts=model.x_to_c_model.out_fc.out_features).to(device)

	attr_criterion = []  # separate criterion (loss function) for each attribute
	if args.weighted_loss:
		imbalance = find_class_imbalance(f"./{args.dataset_root}/{args.dataset_postfix}/train.pkl", True)
		print(imbalance)
		for ratio in imbalance:
			attr_criterion.append(nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).to(device)))
	else:
		for i in range(args.n_concepts):
			attr_criterion.append(nn.CrossEntropyLoss())

	best_loss = None
	best_epoch = 0

	for epoch in range(25):
		train_loss, train_acc = train(model.x_to_c_model, purity_model, train_loader, optimizer, attr_criterion, train=True, device=device)
		test_loss, test_acc = train(model.x_to_c_model, purity_model, val_loader, optimizer, attr_criterion, train=False, device=device)

		if (best_loss is None) or (best_loss > test_loss):
			best_loss = test_loss
			best_epoch = epoch

			if not os.path.exists("./saves"):
				os.makedirs("./saves")
			torch.save(purity_model.state_dict(), f"./saves/{save_name}")
			print("Model saved")

		print("epoch:", epoch)
		print("trainLoss:", train_loss)
		print("trainAcc:", train_acc, "%")
		print("testLoss:", test_loss)
		print("testAcc:", test_acc, "%")

	print(f"Finished training! Best epoch: {bestEpoch}")
