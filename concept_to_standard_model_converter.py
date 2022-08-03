import torch
import torch.nn as nn
import torchvision
from models.model import PlayingCardNet, End2EndModel
import argparse


def get_concept_model(model_path, num_concepts):
	"""
	This function will load and return a concept bottleneck model.

	args:
		model_path (string): File path to Pytorch pth file
		num_concepts (int): number of concepts in the model to load
	returns:
		Pytorch model
	"""

	XtoC = PlayingCardNet(num_concepts=num_concepts)
	XtoC.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	return XtoC


def get_end_to_end_model(model_path, num_concepts, num_classes, num_images_per_output):
	"""
	This function will load and return a concept bottleneck model (end to end / joint).

	args:
		model_path (string): File path to Pytorch pth file
		num_concepts (int): number of concepts in the model to load
	returns:
		Pytorch model
	"""
	model = End2EndModel(num_classes=num_classes, num_concepts=num_concepts, num_images_per_output=num_images_per_output, freeze=False, use_sigmoid=True)
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	return model


def get_converted_model(model, num_concepts, end_to_end):
	"""
	Initilises a new model and load in weights form a concept model.
	The ModuleList layer from the concept model is replaced by a linear layer

	Note: This only works if expanded_dim == 0 for the time being

	args:
		model (Pytorch model): Pytorch model which weights are loaded from
		num_concepts (int): Number of concepts (XtoC model output)
		end_to_end (bool): True if model includes both CtoY and XtoC, else False

	returns:
		Pytorch model
	"""
	new_model = PlayingCardNet(num_classes=num_concepts)
	state_dict = model.state_dict() if not end_to_end else model.x_to_c_model.state_dict()
	new_model.load_state_dict(state_dict, strict=False)

	temp_weights = []
	temp_biases = []

	all_fc = model.all_fc if not end_to_end else model.x_to_c_model.all_fc

	for i in all_fc:
		temp_weights.append(i.weight)
		temp_biases.append(i.bias)

	temp_weights = torch.cat(temp_weights, dim=0)
	temp_biases = torch.cat(temp_biases, dim=0)

	new_model.out_fc = nn.Linear(512, num_concepts)

	new_model.out_fc.weight = nn.Parameter(temp_weights)
	new_model.out_fc.bias = nn.Parameter(temp_biases)

	if end_to_end:
		model.x_to_c_model = new_model
		return model

	return new_model


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Replace the ModuleList layer in a concept bottleneck model with a linear layer')

	parser.add_argument(
		'--model_path',
		type=str,
		help='model path'
	)
	parser.add_argument(
		'--model_out_path',
		type=str,
		help='output model save path'
	)
	parser.add_argument(
		'--end_to_end',
		action='store_true',
		help='Whether model includes both CtoY and XtoC'
	)
	parser.add_argument(
		'--num_concepts',
		type=int,
		help='Number of concepts/attributes in dataset'
	)
	parser.add_argument(
		'--num_classes',
		type=int,
		help='Number of classes in dataset'
	)
	parser.add_argument(
		'--num_images_per_output',
		type=int,
		help='Number of images per model ouput'
	)
	args = parser.parse_args()

	if args.end_to_end:
		model = get_end_to_end_model(model_path=args.model_path, num_concepts=args.num_concepts, num_classes=args.num_classes, num_images_per_output=args.num_images_per_output)
		model = get_converted_model(model, num_concepts=len(model.x_to_c_model.all_fc), end_to_end=args.end_to_end)
	else:
		model = get_concept_model(model_path=args.model_path, num_concepts=args.num_concepts)
		model = get_converted_model(model, num_concepts=len(model.all_fc), end_to_end=args.end_to_end)

	print(model)

	torch.save(model.state_dict(), args.model_out_path)
