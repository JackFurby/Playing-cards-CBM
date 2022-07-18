import numpy as np
import torch
import torch.nn as nn


# def train(model, data_loader, optimizer, criterion, attr_loss_weight, device="cpu"):
def train(model, data_loader, optimizer, criterion, f1=None, mode="standard", attr_loss_weight=None, train=True, device="cpu"):
	if train:
		model.train()
	else:
		model.eval()

	losses = []
	accuracy = []
	task_acc = []
	f1_acc = []
	with torch.set_grad_enabled(train):
		for images, labels, concepts in data_loader:
			# Transfering images and labels to GPU if available
			images, labels, concepts = images.to(device), labels.to(device), concepts.to(device)

			if train:
				# Zero the parameter gradients
				optimizer.zero_grad()

			outputs = model(images)

			if mode == "standard":
				loss = criterion(outputs, labels)
				_, predicted = outputs.max(1)
				accuracy.extend(predicted.eq(labels))

				if f1 is not None:
					f1_acc.append(f1(predicted, labels.to(torch.int32)))
			elif mode == "joint":
				# turn true concepts into one hot vectors and concat them
				batch_vecs = []
				for batch_item in concepts:
					concept_vecs = []
					for card_id in batch_item:
						card_concept = torch.zeros(len(criterion) - 1)
						card_concept[card_id.item()] += 1
						concept_vecs.append(card_concept)  # should be size of concept vector for each image
					batch_vecs.append(torch.stack(concept_vecs))

				concepts = torch.stack(batch_vecs).to(device)

				concept_out = outputs[0]
				task_out = outputs[1]

				concept_out = concept_out.view(-1, 3, 52)  # changes shape of output for criterion

				batch_losses = []
				for i in range(len(criterion) - 1):
					concept_loss = criterion[i + 1](concept_out[:, :, i].squeeze(), concepts[:, :, i])
					batch_losses.append((attr_loss_weight * concept_loss) + (1 - attr_loss_weight) * (criterion[0](task_out, labels)))

				loss = sum(batch_losses) / len(concepts[:, :, i])
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

			else:
				batch_losses = []
				for i in range(len(criterion)):
					loss = criterion[i](outputs[i].squeeze(), concepts[:, i])
					batch_losses.append(attr_loss_weight * loss)

				loss = sum(batch_losses) / len(concepts[:, i])
				sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))

				# accuracy for concepts
				accuracy.append(binary_accuracy(sigmoid_outputs, concepts).item())

				if f1 is not None:
					f1_acc.append(f1(sigmoid_outputs, concepts.to(torch.int32)))

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

	if mode == "joint":
		sum(losses) / len(losses), acc, f1_avg_acc, task_f1, sum(task_acc) / len(task_acc)
	else:
		return sum(losses) / len(losses), acc, f1_avg_acc


def binary_accuracy(output, target):
	"""
	Computes the accuracy for multiple binary predictions
	output and target are Torch tensors
	"""
	pred = output >= 0.5
	acc = (pred.int()).eq(target.int()).sum()
	acc = acc * 100 / np.prod(np.array(target.size()))
	return acc
