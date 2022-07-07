import torch
import torch.nn as nn
import torchvision
from models.model import PlayingCardNet
import argparse


def get_concept_model(model_path, num_concepts):
	"""
	This function will load and return a concept bottleneck model. If only model_1_path
	is provided it is assumed this is for a joint model. If both model paths are provided
	then model_1_path is assumed to be the XtoC model and model_2_path is for the CtoY model.

	args:
		model_path (string): File path to Pytorch pth file
		num_concepts (int): number of concepts in the model to load
	returns:
		Pytorch model
	"""

	XtoC = PlayingCardNet(num_concepts=num_concepts)
	XtoC.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	return XtoC


def get_converted_model(XtoC_model, num_concepts, expand_dim):
	"""
	Initilises a new model and load in weights form a concept model.
	The ModuleList layer from the concept model is replaced by a linear layer

	Note: This only works if expanded_dim == 0 for the time being

	args:
		XtoC_model (Pytorch model): Pytorch model which weights are loaded from
		num_concepts (int): Number of concepts (XtoC model output)
		expand_dim (bool): Size of an additional linear layer in CtoY model (if layer not required, set to 0)

	returns:
		Pytorch model
	"""
	new_XtoC_model = PlayingCardNet(num_classes=num_concepts)
	new_XtoC_model.load_state_dict(XtoC_model.state_dict(), strict=False)

	# Note: this only works if expand_dim==0 (try to update with an expand_dim value)
	temp_weights = []
	temp_biases = []

	for i in XtoC_model.all_fc:
		temp_weights.append(i.fc.weight)
		temp_biases.append(i.fc.bias)

	temp_weights = torch.cat(temp_weights, dim=0)
	temp_biases = torch.cat(temp_biases, dim=0)

	new_XtoC_model.out_fc = nn.Linear(512, num_concepts)

	new_XtoC_model.out_fc.weight = nn.Parameter(temp_weights)
	new_XtoC_model.out_fc.bias = nn.Parameter(temp_biases)

	return new_XtoC_model


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
		'--expand_dim',
		type=int,
		default=0,
		help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only'
	)
	parser.add_argument(
		'--num_concepts',
		type=int,
		help='Number of concepts/attributes in dataset'
	)
	args = parser.parse_args()

	XtoC_model = get_concept_model(model_path=args.model_path, num_concepts=args.num_concepts)
	XtoC_model = get_converted_model(XtoC_model, num_concepts=len(XtoC_model.all_fc), expand_dim=args.expand_dim)
	torch.save(XtoC_model.state_dict(), args.model_out_path)
