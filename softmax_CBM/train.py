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

			"""
			Joint training will train both the concept vector and final classification
			together. This training is similar to the standartd CBM but criterion does
			not require the concept vector to be a modulelist. Instead criterion is
			applied to the entire concept vector together This works in the following way:
			1. The model output is split into concept and task outputs
			2. The number of input images are computed
				2a. If only one image is in the input then an extra dimention in added
				    as such the concept and concept_out have a size of
				    [batch_size, number_of_input_images, number_of_concepts]
			3. The loss is computed for concept and task. attr_loss_weight is
			   used to balance between concept and task loss priority
			4. Loss and accuracy measurements are computed and averaged

			note:
			* The model expects an input image in the format
			  [batch_size, number_of_input_images, channels, width, height] or
			  [batch_size, channels, width, height]

			* The concepts should be in the format
			  [batch_size, number_of_input_images, number_of_concepts] or
			  [batch_size, number_of_concepts]
			"""

			outputs = model(images)

			# seperate model output into task and concept predictions
			concept_out = outputs[0]
			task_out = outputs[1]

			# get the number of images per task output
			if len(concepts.size()) == 2:
				num_input_images = 1
			else:
				num_input_images = concepts.size(1)
				concept_out = concept_out.view(concepts.size())

			# Add in extra dim of len 1 if less than 3 present.
			if len(concepts.size()) == 2:
				concepts = torch.unsqueeze(concepts, 0)
				concepts = torch.permute(concepts, (1, 0, 2))


			# changes shape of output for criterion (concepts and concept_out have same size())
			# [batch size, number of images per sample, concept vector length]
			concept_out = concept_out.view(concepts.size())

			# compute concept and task loss (following our own CBM method without per concept loss)
			batch_losses = []
			for image_concept in range(concept_out.size(1)):
				task_loss = criterion[0](task_out, labels)
				for i in range(len(criterion) - 1):
					concept_loss = criterion[i + 1](concept_out[:, image_concept, :], concepts[:, image_concept, :])
					batch_losses.append((attr_loss_weight * concept_loss) + ((1 - attr_loss_weight) * task_loss))

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

	return sum(losses) / len(losses), acc, f1_avg_acc, task_f1, sum(task_acc) / len(task_acc)


def binary_accuracy(output, target):
	"""
	Computes the accuracy for multiple binary predictions
	output and target are Torch tensors
	"""
	pred = output >= 0.5
	acc = (pred.int()).eq(target.int()).sum()
	acc = acc * 100 / np.prod(np.array(target.size()))
	return acc
