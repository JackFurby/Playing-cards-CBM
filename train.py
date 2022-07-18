import numpy as np
import torch
import torch.nn as nn


# def train(model, data_loader, optimizer, criterion, attr_loss_weight, device="cpu"):
def train(model, data_loader, optimizer, criterion, f1=None, mode="standard", attr_loss_weight=None, device="cpu"):
	model.train()
	losses = []
	accuracy = []
	f1_acc = []
	for images, labels, concepts in data_loader:
		# Transfering images and labels to GPU if available
		images, labels, concepts = images.to(device), labels.to(device), concepts.to(device)

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

			if f1 is not None:
				f1_acc.append(f1(sigmoid_outputs, concepts.to(torch.int32)))
			f1_acc.append(f1(task_out, labels.to(torch.int32)))

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
		loss.backward()

		# Optimizing the parameters
		optimizer.step()

	acc = sum(accuracy) / len(accuracy)

	if f1 is not None:
		f1_avg_acc = sum(f1_acc) / len(f1_acc)
	else:
		f1_avg_acc = None

	# Return the average loss and accuracy
	return sum(losses) / len(losses), acc, f1_avg_acc


def test(model, data_loader, optimizer, criterion, f1=None, mode="standard", attr_loss_weight=None, device="cpu"):
	model.eval()
	losses = []
	accuracy = []
	f1_acc = []
	with torch.no_grad():
		for images, labels, concepts in data_loader:
			# Transfering images and labels to GPU if available
			images, labels, concepts = images.to(device), labels.to(device), concepts.to(device)

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

				if f1 is not None:
					f1_acc.append(f1(sigmoid_outputs, concepts.to(torch.int32)))
				f1_acc.append(f1(task_out, labels.to(torch.int32)))

			else:
				batch_losses = []
				# caculate loss for each concept seperatly
				for i in range(len(criterion)):
					loss = criterion[i](outputs[i].squeeze(), concepts[:, i])
					batch_losses.append(attr_loss_weight * loss)

				loss = sum(batch_losses) / len(concepts[:, i])
				sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))

				accuracy.append(binary_accuracy(sigmoid_outputs, concepts).item())

				if f1 is not None:
					f1_acc.append(f1(sigmoid_outputs, concepts.to(torch.int32)))

			losses.append(loss.item())

		acc = sum(accuracy) / len(accuracy)

		if f1 is not None:
			f1_avg_acc = sum(f1_acc) / len(f1_acc)
		else:
			f1_avg_acc = None

	# Return the average loss
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
