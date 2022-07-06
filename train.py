import numpy as np
import torch
import torch.nn as nn


# def train(model, data_loader, optimizer, criterion, attr_loss_weight, device="cpu"):
def train(model, data_loader, optimizer, criterion, mode="standard", attr_loss_weight=None, device="cpu"):
	model.train()
	losses = []
	accuracy = []
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
		else:
			batch_losses = []
			for i in range(len(criterion)):
				loss = criterion[i](outputs[i].squeeze(), concepts[:, i])
				batch_losses.append(attr_loss_weight * loss)

			loss = sum(batch_losses) / len(concepts[:, i])
			sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))

			# accuracy for concepts
			accuracy.append(binary_accuracy(sigmoid_outputs, concepts).item())

		losses.append(loss.item())
		loss.backward()


		# Optimizing the parameters
		optimizer.step()

	acc = sum(accuracy) / len(accuracy)

	# Return the average loss and accuracy
	return sum(losses) / len(losses), acc


def test(model, data_loader, optimizer, criterion, mode="standard", attr_loss_weight=None, device="cpu"):
	model.eval()
	losses = []
	accuracy = []
	with torch.no_grad():
		for images, labels, concepts in data_loader:
			# Transfering images and labels to GPU if available
			images, labels, concepts = images.to(device), labels.to(device), concepts.to(device)

			outputs = model(images)

			if mode == "standard":
				loss = criterion(outputs, labels)
				_, predicted = outputs.max(1)
				accuracy.extend(predicted.eq(labels))
			else:
				batch_losses = []
				# caculate loss for each concept seperatly
				for i in range(len(criterion)):
					loss = criterion[i](outputs[i].squeeze(), concepts[:, i])
					batch_losses.append(attr_loss_weight * loss)

				loss = sum(batch_losses) / len(concepts[:, i])
				sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))

				accuracy.append(binary_accuracy(sigmoid_outputs, concepts).item())



			losses.append(loss.item())

		acc = sum(accuracy) / len(accuracy)

	# Return the average loss
	return sum(losses) / len(losses), acc


def binary_accuracy(output, target):
	"""
	Computes the accuracy for multiple binary predictions
	output and target are Torch tensors
	"""
	pred = output >= 0.5
	acc = (pred.int()).eq(target.int()).sum()
	acc = acc * 100 / np.prod(np.array(target.size()))
	return acc
