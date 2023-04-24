import torch
import numpy as np
import argparse
from torchmetrics import F1Score
from dataset.dataset import load_data
from models.vgg import End2EndModel
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt


def make_plot(three_dataset_models, poker_dataset_models, var_index, var_name, save_name):
	x = ["0%", "10%", "30%", "50%", "70%", "90%"]

	random_three, random_poker, concept_random_three, concept_random_poker = [], [], [], []

	for idx, removed in enumerate(poker_dataset_models):
		concept_random = [i[var_index] for i in removed if "concepts-random" in i[0]]
		random = [i[var_index] for i in removed if "concepts-random" not in i[0]]
		if idx == 0:  # duplicate no features removed
			random_poker.append(random)
			concept_random_poker.append(random)
		else:
			random_poker.append(random)
			concept_random_poker.append(concept_random)

	for idx, removed in enumerate(three_dataset_models):
		for i in removed:
			print(i[0], i[1], i[4])


		concept_random = [i[var_index] for i in removed if "concepts-random" in i[0]]
		random = [i[var_index] for i in removed if "concepts-random" not in i[0]]
		if idx == 0:
			random_three.append(random)
			concept_random_three.append(random)
		else:
			random_three.append(random)
			concept_random_three.append(concept_random)

	#three_ran_plot = plt.plot(x, [sum(i)/len(i) for i in random_three], label="Three", marker='o')
	#plt.fill_between(x, [min(i) for i in random_three], [max(i) for i in random_three], alpha=0.2)
	#poker_ran_plot = plt.plot(x, [sum(i)/len(i) for i in random_poker], label="Poker", marker='o')
	#plt.fill_between(x, [min(i) for i in random_poker], [max(i) for i in random_poker], alpha=0.2)
	#three_con_ran_plot = plt.plot(x, [sum(i)/len(i) for i in concept_random_three], label="Concepts three", marker='o')
	#plt.fill_between(x, [min(i) for i in concept_random_three], [max(i) for i in concept_random_three], alpha=0.2)
	#poker_con_ran_plot = plt.plot(x, [sum(i)/len(i) for i in concept_random_poker], label="Concepts poker", marker='o')
	#plt.fill_between(x, [min(i) for i in concept_random_poker], [max(i) for i in concept_random_poker], alpha=0.2)

	three_ran_plot = plt.plot(x, [max(i) for i in random_three], label="Three", marker='o')
	poker_ran_plot = plt.plot(x, [max(i) for i in random_poker], label="Poker", marker='o')
	three_con_ran_plot = plt.plot(x, [max(i) for i in concept_random_three], label="Concepts three", marker='o')
	poker_con_ran_plot = plt.plot(x, [max(i) for i in concept_random_poker], label="Concepts poker", marker='o')

	plt.xlabel("Features removed")
	plt.ylabel(f"{var_name} accuracy")
	plt.legend(loc="best")
	#plt.ylim(92, 100)
	plt.xlim(0, 5)
	plt.autoscale(enable=True, axis="y", tight=None)

	save_dir = 'results/roar'
	if not os.path.exists(f"{save_dir}"):
		os.makedirs(f"{save_dir}")
	plt.savefig(f'{save_dir}/{save_name}.png', bbox_inches='tight', pad_inches = 0)

	plt.clf()


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

	return acc, f1_avg_acc.item(), task_f1.item(), (sum(task_acc).item() / len(task_acc)) * 100


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

	if os.path.exists("./experiments/roar_acc.pkl"):
		print("Previous accuracies file found. Loading model accuracies from experiments/all_acc.pkl")
		all_acc = pickle.load(open("./experiments/roar_acc.pkl", "rb"))
	else:

		device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.use_cpu) else "cpu")
		if torch.cuda.is_available() and not args.use_cpu:
			print("Device:", device, torch.cuda.get_device_name(0))
		else:
			print("Device:", device)

		f1 = F1Score().to(device)

		all_acc = []

		for file in os.listdir(args.weights_dir):
			use_sig = True
			if "single" in file:
				num_concepts = 17
				num_classes = 52
			else:
				num_concepts = 52
				num_classes = 6

			if "single" in file:
				postfix = "splits/single"
			elif "three" in file:
				postfix = "splits/three_card_poker"  # the c->y model part is trained on three card poker
			else:
				postfix = "splits/three_card_poker"

			if "0_0" in file:
				dataset_root = f"./dataset/playing-cards"
			elif "0_1" in file:
				dataset_root = f"./dataset/roar/random/0.1"
			elif "0_3" in file:
				dataset_root = f"./dataset/roar/random/0.3"
			elif "0_5" in file:
				dataset_root = f"./dataset/roar/random/0.5"
			elif "0_7" in file:
				dataset_root = f"./dataset/roar/random/0.7"
			else:
				dataset_root = f"./dataset/roar/random/0.9"

			val_loader = load_data(f"./{dataset_root}/{postfix}/val.pkl", args.batch_size, dataset_root=dataset_root, num_workers=1)

			model = End2EndModel(num_classes=num_classes, num_concepts=num_concepts, use_sigmoid=use_sig, use_modulelist=False)
			model.load_state_dict(torch.load(f"{args.weights_dir}/{file}", map_location=torch.device('cpu')))

			model.to(device)

			concept_acc, concept_F1, task_F1, task_acc = eval(model, val_loader, f1=f1, device=device)

			print(file, f"./{dataset_root}/{postfix}/val.pkl", concept_acc)

			all_acc.append((file, concept_acc, concept_F1, task_F1, task_acc))

		pickle.dump(all_acc, open(f"./experiments/roar_acc.pkl", "wb"))

	three_dataset_models = []
	poker_dataset_models = []

	for features_removed in [0, 1, 3, 5, 7, 9]:
		three_dataset_models.append([])
		poker_dataset_models.append([])

	for i in all_acc:
		name_split = i[0].split("-")
		if name_split[-2][-1] == "0":
			f_removed_index = 0
		elif name_split[-2][-1] == "1":
			f_removed_index = 1
		elif name_split[-2][-1] == "3":
			f_removed_index = 2
		elif name_split[-2][-1] == "5":
			f_removed_index = 3
		elif name_split[-2][-1] == "7":
			f_removed_index = 4
		else:
			f_removed_index = 5

		if "three_card_poker" in i[0]:
			poker_dataset_models[f_removed_index].append(i)
		else:
			three_dataset_models[f_removed_index].append(i)

	make_plot(
		three_dataset_models,
		poker_dataset_models,
		1,
		"Concept",
		"three_v_poker_concepts"
	)

	make_plot(
		three_dataset_models,
		poker_dataset_models,
		4,
		"Task",
		"three_v_poker_task"
	)
