import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple


__all__ = [
	'DenseNet',
	'densenet121',
	'densenet169',
	'densenet201',
	'densenet161',
	"x_to_c_model",
	"End2EndModel"]

model_urls = {
	'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
	'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
	'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
	'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class End2EndModel(torch.nn.Module):
	def __init__(self, num_classes=6, num_concepts=52, pretrained_weights=None, imagenet_weights=False, freeze=False, use_sigmoid=False, use_modulelist=True):
		super(End2EndModel, self).__init__()
		self.c_to_y_model = CtoYNet(num_concepts, num_classes)
		self.use_sigmoid = use_sigmoid
		self.use_modulelist = use_modulelist  # modulelist used for training but may be converted to linear layer after

		if self.use_modulelist:
			self.x_to_c_model = densenet121(pretrained=imagenet_weights, num_concepts=num_concepts, pretrained_weights=pretrained_weights)
		else:
			self.x_to_c_model = densenet121(pretrained=imagenet_weights, num_classes=num_concepts, pretrained_weights=pretrained_weights)

		# if freeze == true then freeze all layers in x_to_c_model
		if freeze:  # only finetune c_to_y_model
			for name, param in self.x_to_c_model.named_parameters():
				param.requires_grad = False
			print("frozen layers in x_to_c_model")

	def forward_stage2(self, stage_1_out):
		if self.use_modulelist:
			stage_1_out = torch.cat(stage_1_out, dim=1)
		if self.use_sigmoid:
			c_to_y_input = torch.nn.Sigmoid()(stage_1_out)
		else:
			c_to_y_input = stage_1_out

		return stage_1_out, self.c_to_y_model(c_to_y_input)

	def forward(self, x):
		return self.forward_stage2(self.x_to_c_model(x))


class CtoYNet(nn.Module):
	def __init__(self, num_concept_in, num_classes, expand_dim=128):
		super(CtoYNet, self).__init__()
		self.expand_dim = expand_dim
		if self.expand_dim:
			self.classifier = nn.Sequential(
				nn.Linear(num_concept_in, expand_dim),
				nn.ReLU(),
				nn.Linear(expand_dim, num_classes)
			)
		else:
			self.classifier = self.lib.Linear(num_concept_in, num_classes)

	def forward(self, x):
		return self.classifier(x)


class _DenseLayer(nn.Module):
	def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
		super(_DenseLayer, self).__init__()
		self.norm1: nn.BatchNorm2d
		self.add_module('norm1', nn.BatchNorm2d(num_input_features))
		self.relu1: nn.ReLU
		self.add_module('relu1', nn.ReLU(inplace=True))
		self.conv1: nn.Conv2d
		self.add_module(
			'conv1',
			nn.Conv2d(
				num_input_features,
				bn_size * growth_rate,
				kernel_size=1,
				stride=1,
				bias=False
			)
		)
		self.norm2: nn.BatchNorm2d
		self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
		self.relu2: nn.ReLU
		self.add_module('relu2', nn.ReLU(inplace=True))
		self.conv2: nn.Conv2d
		self.add_module(
			'conv2',
			nn.Conv2d(
				bn_size * growth_rate,
				growth_rate,
				kernel_size=3,
				stride=1,
				padding=1,
				bias=False
			)
		)
		self.drop_rate = float(drop_rate)
		self.memory_efficient = memory_efficient

	def bn_function(self, inputs):
		concated_features = torch.cat(inputs, 1)
		bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
		return bottleneck_output

	# todo: rewrite when torchscript supports any
	def any_requires_grad(self, input):
		for tensor in input:
			if tensor.requires_grad:
				return True
		return False

	def call_checkpoint_bottleneck(self, input):
		def closure(*inputs):
			return self.bn_function(inputs)

		return cp.checkpoint(closure, *input)

	def forward(self, input):
		if isinstance(input, Tensor):
			prev_features = [input]
		else:
			prev_features = input

		bottleneck_output = self.bn_function(prev_features)

		new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
		if self.drop_rate > 0:
			new_features = F.dropout(
				new_features,
				p=self.drop_rate,
				training=self.training
			)
		return new_features


class _DenseBlock(nn.ModuleDict):
	_version = 2

	def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
		super(_DenseBlock, self).__init__()
		for i in range(num_layers):
			layer = _DenseLayer(
				num_input_features + i * growth_rate,
				growth_rate=growth_rate,
				bn_size=bn_size,
				drop_rate=drop_rate,
				memory_efficient=memory_efficient,
			)
			self.add_module('denselayer%d' % (i + 1), layer)

	def forward(self, init_features: Tensor) -> Tensor:
		features = [init_features]
		for name, layer in self.items():
			new_features = layer(features)
			features.append(new_features)
		return torch.cat(features, 1)


class _Transition(nn.Sequential):
	def __init__(self, num_input_features, num_output_features):
		super(_Transition, self).__init__()
		self.add_module('norm', nn.BatchNorm2d(num_input_features))
		self.add_module('relu', nn.ReLU(inplace=True))
		self.add_module(
			'conv',
			nn.Conv2d(
				num_input_features,
				num_output_features,
				kernel_size=1,
				stride=1,
				bias=False
			)
		)
		self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
	r"""Densenet-BC model class, based on
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
	Args:
		growth_rate (int) - how many filters to add each layer (`k` in paper)
		block_config (list of 4 ints) - how many layers in each pooling block
		num_init_features (int) - the number of filters to learn in the first convolution layer
		bn_size (int) - multiplicative factor for number of bottle neck layers
		  (i.e. bn_size * k features in the bottleneck layer)
		drop_rate (float) - dropout rate after each dense layer
		num_classes (int) - number of classification classes
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
			but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
	"""

	def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=52, num_concepts=0, memory_efficient=False, freeze=False, pretrained_weights=None):
		super(DenseNet, self).__init__()

		self.num_concepts = num_concepts

		# First convolution
		self.features = nn.Sequential(OrderedDict([
			('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
			('norm0', nn.BatchNorm2d(num_init_features)),
			('relu0', nn.ReLU(inplace=True)),
			('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
		]))

		# Each denseblock
		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			block = _DenseBlock(
				num_layers=num_layers,
				num_input_features=num_features,
				bn_size=bn_size,
				growth_rate=growth_rate,
				drop_rate=drop_rate,
				memory_efficient=memory_efficient
			)
			self.features.add_module('denseblock%d' % (i + 1), block)
			num_features = num_features + num_layers * growth_rate
			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
				self.features.add_module('transition%d' % (i + 1), trans)
				num_features = num_features // 2

		# Final batch norm
		self.features.add_module('norm5', nn.BatchNorm2d(num_features))

		# concept training requires a modulelist for loss
		if self.num_concepts > 0:
			# all_fc replaces fc2 from standard model
			self.classifier = nn.ModuleList()  # separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
			for i in range(self.num_concepts):
				self.classifier.append(nn.Linear(num_features, 1))
		# basic fc for standard training
		else:
			self.classifier = nn.Linear(num_features, num_classes)

		# if weights file provided, load them
		if pretrained_weights is not None:
			self.load_state_dict(torch.load(pretrained_weights, map_location=torch.device('cpu')), strict=False)  # as we are replacing the final fc layer we can set strict to False
			print(f"Loaded weights from {pretrained_weights}")

		# if freeze == true then freeze all but fc layers
		if freeze:  # only finetune fc layers
			for name, param in self.named_parameters():
				if 'fc' not in name:
					param.requires_grad = False
			print("frozen lower model layers")

		# Official init from torch repo.
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.constant_(m.bias, 0)

	def forward(self, x: Tensor) -> Tensor:
		features = self.features(x)
		x = F.relu(features, inplace=True)
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = torch.flatten(x, 1)
		if self.num_concepts > 0:
			out = []
			for fc in self.classifier:
				out.append(fc(x))
		else:
			out = self.classifier(x)
		return out


def _load_state_dict(model, model_url, progress):
	# '.'s are no longer allowed in module names, but previous _DenseLayer
	# has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
	# They are also in the checkpoints in model_urls. This pattern is used
	# to find such keys.
	pattern = re.compile(
		r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

	state_dict = load_state_dict_from_url(model_url, progress=progress)
	for key in list(state_dict.keys()):
		res = pattern.match(key)
		if res:
			new_key = res.group(1) + res.group(2)
			state_dict[new_key] = state_dict[key]
			del state_dict[key]
	model.load_state_dict(state_dict)


def _densenet(
	arch: str,
	growth_rate: int,
	block_config: Tuple[int, int, int, int],
	num_init_features: int,
	pretrained: bool,
	progress: bool,
	**kwargs: Any
) -> DenseNet:
	model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
	if pretrained:
		_load_state_dict(model, model_urls[arch], progress)
	return model


def densenet121(pretrained=False, progress=True, **kwargs):
	r"""Densenet-121 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
	The required minimum input size of the model is 29x29.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
			but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
	"""
	return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress, **kwargs)


def densenet161(pretrained=False, progress=True, **kwargs):
	r"""Densenet-161 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
	The required minimum input size of the model is 29x29.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
			but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
	"""
	return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress, **kwargs)


def densenet169(pretrained=False, progress=True, **kwargs):
	r"""Densenet-169 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
	The required minimum input size of the model is 29x29.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
			but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
	"""
	return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress, **kwargs)


def densenet201(pretrained=False, progress=True, **kwargs):
	r"""Densenet-201 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
	The required minimum input size of the model is 29x29.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
			but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
	"""
	return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress, **kwargs)
