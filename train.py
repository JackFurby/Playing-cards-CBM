import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sklearn import metrics
from torchmetrics.functional.classification import multilabel_auroc


def get_loss(output, target, index, pos_weight, device):
	target = target[:, index].view(-1)
	print(output[:, index].view(-1))
	print(torch.nn.Sigmoid()(output[:, index].view(-1)))
	if pos_weight == None:
		"""
		binary_output = torch.nn.Sigmoid()(output[:, index].view(-1)) >= 0.5
		print(binary_output)
		if target.sum() == 0:  # set weight to number of 1's in target if output has no 1's
			weight = torch.tensor(0.).to(device)
		else:  # based on https://arxiv.org/pdf/1705.02315v5.pdf
			print(target.size(0), target.sum(), target.size(0) - target.sum())
			batch_pos = target.size(0) / target.sum()
			batch_neg = target.size(0) / (target.size(0) - target.sum())
			print(batch_pos, batch_neg)
			#wight_pos = batch_pos * (target.sum() - binary_output.sum())
			#weight_neg = batch_neg * ((target.size(0) - target.sum()) - (1 - binary_output.size(0) - binary_output.sum()))
			wight_pos = batch_pos * (target.sum() - binary_output.size(0))
			weight_neg = batch_neg * ((target.size(0) - target.sum()) - (1 - binary_output.size(0)))
			print(wight_pos, weight_neg)
			weight = wight_pos + weight_neg
		print(weight)
		print("===========================")
		loss = F.binary_cross_entropy_with_logits(output[:, index].view(-1), target, pos_weight=weight)
		"""
		if target.sum() == 0:
			loss = torch.tensor(0., requires_grad=True).to(device)
		else:
			weight = (target.size()[0] - target.sum()) / target.sum()
			loss = F.binary_cross_entropy_with_logits(output[:, index].view(-1), target, pos_weight=weight)
	else:
		pos_weight = torch.from_numpy(
			np.array(
				pos_weight,
				dtype=np.float32
			)).to(device).type_as(target)
		loss = F.binary_cross_entropy_with_logits(
			output[:, index].view(-1), target, pos_weight=pos_weight[index])

	return loss


def train(model, data_loader, optimizer, criterion, f1=None, mode="standard", attr_loss_weight=None, train=True, device="cpu"):
	"""
	Training and evaluation loop.

		args:
			model (Pytorch model): Pytorch model to train / evaluate
			data_loader (Pytorch dataloader): Pytorch dataloader with dataset Initilised
			optimizer (torch.optim): Optimiser for updating model weights
			criterion (pytorch loss / list of Pytorch loss): Loss method
			f1 (Pytorch torchmetrics F1): F1 metrics object
			mode (str): Training method
			attr_loss_weight (float): weight between concept and task loss (joint model only)
			train (bool): True if model is training, False if model is testing
			device (str): Device top train / test using (e.g. cpu or cuda:0)

		returns:
			average loss: Average loss of model
			average accuracy: Average accuracy of concepts / task (non joint)
			average F1 score: Average F1 score of concepts / task (non joint)
			average task F1 score (joint only): Average task F1 score
			average task accuracy (joint only): Average task accuracy
	"""

	if train:
		model.train()
	else:
		model.eval()

	losses = []
	accuracy = []
	task_acc = []
	f1_acc = []
	auc_acc = []
	with torch.set_grad_enabled(train):
		for images, labels, concepts in tqdm(data_loader, unit="batch"):
			# Transfering images and labels to GPU if available
			images, labels, concepts = images.to(device), labels.to(device), concepts.to(device)

			if train:
				# Zero the parameter gradients
				optimizer.zero_grad()

			if mode == "CtoY":
				concepts = concepts.view(concepts.size(0), -1)  # make sure input is vector
				outputs = model(concepts)
			else:
				outputs = model(images)

			# discard concept layer output as we are not training the XtoC model part
			if mode == "sequential":
				outputs = outputs[1]

			if mode in ["standard", "CtoY", "sequential"]:
				loss = criterion(outputs, labels)
				_, predicted = outputs.max(1)
				accuracy.extend(predicted.eq(labels))

				if f1 is not None:
					f1_acc.append(f1(predicted, labels.to(torch.int32)))
			elif mode == "joint":
				"""
				Joint training will train both the concept vector and final classification
				together. This works in the following way:
				1. The model output is split into concept and task outputs
				3. The loss is computed for concept and task. attr_loss_weight is
					used to balance between concept and task loss priority
				4. Loss and accuracy measurements are computed and averaged

				note:
				* The model expects an input image in the format
					[batch_size, channels, width, height]

				* The concepts should be in the format
					[batch_size, number_of_concepts]
				"""

				# seperate model output into task and concept predictions
				concept_out = outputs[0]
				task_out = outputs[1]

				# compute concept and task loss (following CBM joint training method)
				batch_losses = []

				task_loss = criterion[0](task_out, labels)
				for i in range(len(criterion) - 1):
					concept_loss = criterion[i + 1](concept_out[:, i].squeeze(), concepts[:, i])
					batch_losses.append((attr_loss_weight * concept_loss) + task_loss)

				loss = sum(batch_losses) / len(batch_losses)
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

				# AUC metric
				auc_acc.append(multilabel_auroc(sigmoid_outputs, concepts, num_labels=concepts.size(1), average="macro", thresholds=None))
			else:  # XtoC
				batch_losses = []
				for i in range(len(criterion)):
					loss = criterion[i](outputs[i].squeeze(), concepts[:, i])
					batch_losses.append(loss)

				loss = sum(batch_losses) / len(concepts[:, i])
				sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))

				# accuracy for concepts
				accuracy.append(binary_accuracy(sigmoid_outputs, concepts).item())

				if f1 is not None:
					f1_acc.append(f1(sigmoid_outputs, concepts.to(torch.int32)))

				# AUC metric
				auc_acc.append(multilabel_auroc(sigmoid_outputs, concepts, num_labels=concepts.size(1), average="macro", thresholds=None))

			losses.append(loss.item())
			if train:
				loss.backward()

				# Optimizing the parameters
				optimizer.step()

		acc = sum(accuracy) / len(accuracy)

		if f1 is not None:
			f1_avg_acc = sum(f1_acc) / len(f1_acc)
		else:
			f1_avg_acc = None
			task_f1 = None

		if len(auc_acc) > 0:
			auc_avg = (sum(auc_acc) / len(auc_acc)).item()
		else:
			auc_avg = None

	if mode == "joint":
		return sum(losses) / len(losses), acc, f1_avg_acc, task_f1, (sum(task_acc) / len(task_acc)).item(), auc_avg
	else:
		return sum(losses) / len(losses), acc, f1_avg_acc, auc_avg


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
