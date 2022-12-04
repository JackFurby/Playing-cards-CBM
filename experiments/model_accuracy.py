import torch
import numpy as np
import argparse
from torchmetrics import F1Score
from dataset.dataset import load_data
from models.model import End2EndModel
from tqdm import tqdm
import os
import pickle


def eval(model, data_loader, f1=None, device="cpu"):
	"""
	Training and evaluation loop.

		args:
			model (Pytorch model): Pytorch model to train / evaluate
			data_loader (Pytorch dataloader): Pytorch dataloader with dataset Initilised
			f1 (Pytorch torchmetrics F1): F1 metrics object
			device (str): Device top train / test using (e.g. cpu or cuda:0)

		returns:
			average accuracy: Average accuracy of concepts / task
			average F1 score: Average F1 score of concepts / task
			average task F1 score: Average task F1 score
			average task accuracy: Average task accuracy
	"""

	model.eval()

	accuracy = []
	task_acc = []
	f1_acc = []
	for images, labels, concepts in tqdm(data_loader, unit="batch"):
		# Transfering images and labels to GPU if available
		images, labels, concepts = images.to(device), labels.to(device), concepts.to(device)

		outputs = model(images)

		# seperate model output into task and concept predictions
		concept_out = outputs[0]
		task_out = outputs[1]

		sigmoid_outputs = torch.nn.Sigmoid()(concept_out.view(concept_out.size(0), -1))

		# accuracy for concepts
		concepts = concepts.view(concepts.size(0), -1)
		accuracy.append(binary_accuracy(sigmoid_outputs, concepts).item())

		# accuracy for task
		_, predicted = task_out.max(1)
		task_acc.extend(predicted.eq(labels))

		if f1 is not None:
			f1_acc.append(f1(sigmoid_outputs, concepts.to(torch.int32)))
			task_f1 = f1(task_out, labels.to(torch.int32))

	acc = sum(accuracy) / len(accuracy)

	if f1 is not None:
		f1_avg_acc = sum(f1_acc) / len(f1_acc)
	else:
		f1_avg_acc = None
		task_f1 = None

	return acc, f1_avg_acc.item(), task_f1.item(), (sum(task_acc).item() / len(task_acc))


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
	acc = (pred.int()).eq(target.int()).sum()
	acc = acc * 100 / np.prod(np.array(target.size()))
	return acc


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Evaluate a Concept Bottleneck Model')
	parser.add_argument(
		'--batch_size',
		type=int,
		default=16,
		help='Number of samples per batch (default: 16)'
	)
	parser.add_argument(
		'--weights_dir',
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
		'--use_cpu',
		action='store_true',
		help='Only use the system CPU, even if CUDA is available'
	)
	args = parser.parse_args()

	if os.path.exists("./experiments/all_acc.pkl"):
		print("Previous accuracies file found. Loading model accuracies from experiments/all_acc.pkl")
		all_acc = pickle.load(open("./experiments/all_acc.pkl", "rb"))
	else:

		device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.use_cpu) else "cpu")
		if torch.cuda.is_available() and not args.use_cpu:
			print("Device:", device, torch.cuda.get_device_name(0))
		else:
			print("Device:", device)

		f1 = F1Score().to(device)

		all_acc = []

		for file in os.listdir(args.weights_dir):
			if "sig" in file or "joint" not in file:
				use_sig = True
			else:
				use_sig = False
			if "single" in file:
				num_concepts = 17
				num_classes = 52
			else:
				num_concepts = 52
				num_classes = 6

			if "single" in file:
				postfix = "splits/single"
			elif "three" in file:
				postfix = "splits/three"
			else:
				postfix = "splits/three_card_poker"

			val_loader = load_data(f"./{args.dataset_root}/{postfix}/val.pkl", args.batch_size, dataset_root=args.dataset_root, num_workers=1)

			model = End2EndModel(num_classes=num_classes, num_concepts=num_concepts, use_sigmoid=use_sig, use_modulelist=False)
			model.load_state_dict(torch.load(f"{args.weights_dir}/{file}", map_location=torch.device('cpu')))

			model.to(device)

			concept_acc, concept_F1, task_F1, task_acc = eval(model, val_loader, f1=f1, device=device)

			all_acc.append((file, concept_acc, concept_F1, task_F1, task_acc))

		pickle.dump(all_acc, open(f"./experiments/all_acc.pkl", "wb"))

	print(all_acc)
