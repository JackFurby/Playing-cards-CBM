import numpy as np
import torch
import torch.nn as nn


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

				concept_out = outputs[0]
				task_out = outputs[1]

				if len(concepts.size()) == 2:
					num_input_images = 1
				else:
					num_input_images = concepts.size(1)
				# turn true concepts into one hot vectors and concat them
				batch_vecs = []
				for batch_item in concepts:
					concept_vecs = []
					if num_input_images > 1:  # if ther are multiple images for sample then len of size will be greater than 1
						for card_id in batch_item:
							card_concept = torch.zeros(len(criterion) - 1)
							card_concept[card_id.item()] += 1
							concept_vecs.append(card_concept)  # should be size of concept vector for each image
					else:
						concept_vecs = batch_item
					if num_input_images > 1:
						batch_vecs.append(torch.stack(concept_vecs))
					else:
						batch_vecs.append(concept_vecs)

				concepts = torch.stack(batch_vecs).to(device)

				# Add in extra dim of len 1 if less than 3 present.
				if len(concepts.size()) == 2:
					concepts = torch.unsqueeze(concepts, 0)
					concepts = torch.permute(concepts, (1, 0, 2))



				# changes shape of output for criterion (concepts and concept_out have same size())
				# [batch size, number of images per sample, concept vector length]
				concept_out = concept_out.view(concepts.size())

				batch_losses = []
				for image_concept in range(concept_out.size(1)):
					individual_concept = concepts[:, image_concept, :]
					individual_concept_out = concept_out[:, image_concept, :]

					task_loss = criterion[0](task_out, labels)
					for i in range(len(criterion) - 1):
						concept_loss = criterion[i + 1](individual_concept_out[:, i].squeeze(), individual_concept[:, i])
						batch_losses.append((attr_loss_weight * concept_loss) + ((1 - attr_loss_weight) * (task_loss)))

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
		return sum(losses) / len(losses), acc, f1_avg_acc, task_f1, sum(task_acc) / len(task_acc)
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
