import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


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
	with torch.set_grad_enabled(train):
		for images, labels, concepts in tqdm(data_loader, unit="batch"):
			# Transfering images and labels to GPU if available
			images, labels, concepts = images.to(device), labels.to(device), concepts.to(device)

			if train:
				# Zero the parameter gradients
				optimizer.zero_grad()

			if mode == "CtoY":
				concepts = concepts.view(concepts.size(0), -1)  # make sure input is vector (will not be if using multi input images)
				outputs = model(concepts)
			else:
				outputs = model(images)

			if mode in ["standard", "CtoY"]:
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

				# compute concept and task loss (following CBM joint training method)
				batch_losses = []
				for image_concept in range(concept_out.size(1)):
					individual_concept = concepts[:, image_concept, :]
					individual_concept_out = concept_out[:, image_concept, :]

					task_loss = criterion[0](task_out, labels)
					for i in range(len(criterion) - 1):
						concept_loss = criterion[i + 1](individual_concept_out[:, i].squeeze(), individual_concept[:, i])
						#batch_losses.append((attr_loss_weight * concept_loss) + ((1 - attr_loss_weight) * task_loss))
						batch_losses.append(concept_loss + ((1 - attr_loss_weight) * task_loss))
						#batch_losses.append((attr_loss_weight * concept_loss) + task_loss)

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
		return sum(losses) / len(losses), acc, f1_avg_acc, task_f1, (sum(task_acc) / len(task_acc)).item()
	else:
		return sum(losses) / len(losses), acc, f1_avg_acc


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
