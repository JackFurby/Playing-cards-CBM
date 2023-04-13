import torch
import torch.nn as nn
import torchvision
import models.vgg as models
import argparse
import os


def get_concept_model(model_path, num_concepts, vgg_version):
	"""
	This function will load and return a concept bottleneck model.

	args:
		model_path (string): File path to Pytorch pth file
		num_concepts (int): number of concepts in the model to load
	returns:
		Pytorch model
	"""

	if args.vgg_version == "19_bn":
		XtoC = models.vgg19_bn(num_concepts=num_concepts)
	else:
		XtoC = models.vgg11_bn(num_concepts=num_concepts)
	XtoC.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	return XtoC


def get_task_model(model_path, num_concepts, num_classes):
	"""
	This function will load and return a concept bottleneck model C to Y part.

	args:
		model_path (string): File path to Pytorch pth file
		num_concepts (int): number of concepts in the model to load
		num_classes (int): number of classes in the model to load
	returns:
		Pytorch model
	"""

	CtoY = models.CtoYNet(num_concept_in=num_concepts, num_classes=num_classes)
	CtoY.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	return CtoY


def get_end_to_end_model(model_path, num_concepts, num_classes, vgg_version):
	"""
	This function will load and return a concept bottleneck model (end to end / joint).

	args:
		model_path (string): File path to Pytorch pth file
		num_concepts (int): number of concepts in the model to load
	returns:
		Pytorch model
	"""
	model = models.End2EndModel(num_classes=num_classes, num_concepts=num_concepts, freeze=False, use_sigmoid=True, model_version=vgg_version)
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	return model


def get_converted_model(model, num_concepts, end_to_end, vgg_version):
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
	if args.vgg_version == "19_bn":
		new_model = models.vgg19_bn(num_classes=num_concepts)
	else:
		new_model = models.vgg11_bn(num_classes=num_concepts)
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


def combine_model_parts(model_1, model_2, num_classes, num_concepts, vgg_version):
	"""
	"""

	end_to_end_model = models.End2EndModel(num_classes=num_classes, num_concepts=num_concepts, freeze=False, use_sigmoid=True, model_version=vgg_version)
	end_to_end_model.x_to_c_model = model_1
	end_to_end_model.c_to_y_model = model_2
	return end_to_end_model


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Replace the ModuleList layer in a concept bottleneck model with a linear layer')

	parser.add_argument(
		'--model_path',
		type=str,
		help='model path for X to C or end to end models'
	)
	parser.add_argument(
		'--model_2_path',
		type=str,
		default=None,
		help='model path for C to Y model'
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
		'--vgg_version',
		type=str,
		default="11_bn",
		choices=["11_bn", "19_bn"],
		help='Specify the type of vgg model to use (default: "11_bn")'
	)
	args = parser.parse_args()

	if args.end_to_end:
		model = get_end_to_end_model(model_path=args.model_path, num_concepts=args.num_concepts, num_classes=args.num_classes, vgg_version=args.vgg_version)
		model = get_converted_model(model, num_concepts=len(model.x_to_c_model.all_fc), end_to_end=args.end_to_end, vgg_version=args.vgg_version)
	else:
		model = get_concept_model(model_path=args.model_path, num_concepts=args.num_concepts, vgg_version=args.vgg_version)
		model = get_converted_model(model, num_concepts=len(model.all_fc), end_to_end=args.end_to_end, vgg_version=args.vgg_version)
		if args.model_2_path is not None:  # X to C and C to Y
			model_2 = get_task_model(model_path=args.model_2_path, num_concepts=args.num_concepts, num_classes=args.num_classes)
			model = combine_model_parts(model, model_2, args.num_classes, args.num_concepts, vgg_version=args.vgg_version)

	print(model)

	if not os.path.exists("/".join(args.model_out_path.split("/")[:-1])):
		os.makedirs("/".join(args.model_out_path.split("/")[:-1]))
	torch.save(model.state_dict(), args.model_out_path, _use_new_zipfile_serialization=True)
