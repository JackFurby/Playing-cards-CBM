"""credit: https://github.com/fhvilshoj/TorchLRP/blob/74253a1be05f0be0b7c535736023408670443b6e/lrp/converter.py#L12"""
import torch
from lrp.conv import Conv2d
from lrp.linear import Linear
from lrp.sequential import Sequential
from lrp.maxpool import MaxPool2d
from lrp.avgpool import AvgPool2d

conversion_table = {
	'Linear': Linear,
	'Conv2d': Conv2d,
	'MaxPool2d': MaxPool2d,
	'AvgPool2d': AvgPool2d
}


def convert(module, modules=None):
	print(module)
	# First time
	if modules is None:
		modules = []
		for m in module.children():
			convert(m, modules=modules)

		sequential = Sequential(*modules)
		return sequential

	# Recursion
	if isinstance(module, torch.nn.Sequential):
		for m in module.children():
			convert(m, modules=modules)

	elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.AvgPool2d):
		class_name = module.__class__.__name__
		lrp_module = conversion_table[class_name].from_torch(module)
		modules.append(lrp_module)
	# maxpool is handled with gradient for the moment

	elif isinstance(module, torch.nn.ReLU):
		# avoid inplace operations. They might ruin PatternNet pattern
		# computations
		modules.append(torch.nn.ReLU())

	elif isinstance(module, torch.nn.Flatten):
		modules.append(torch.nn.Flatten())
	else:
		modules.append(module)
