"""
VGG Network modified from https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
New changes: add softmax layer + option for freezing lower layers except fc
"""
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchexplain

from typing import Union, List, Dict, Any, cast

__all__ = [
	"VGG",
	"vgg11",
	"vgg11_bn",
	"vgg13",
	"vgg13_bn",
	"vgg16",
	"vgg16_bn",
	"vgg19_bn",
	"vgg19",
	"x_to_c_model",
	"MLP",
	"End2EndModel"
]

model_urls = {
	"vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
	"vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
	"vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
	"vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
	"vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
	"vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
	"vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
	"vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class End2EndModel(torch.nn.Module):
	def __init__(self, num_classes=6, num_concepts=52, model_version="11_bn", pretrained_weights=None, freeze=False, use_sigmoid=False, use_modulelist=True):
		super(End2EndModel, self).__init__()
		self.c_to_y_model = CtoYNet(num_concepts, num_classes)
		self.use_sigmoid = use_sigmoid
		self.use_modulelist = use_modulelist  # modulelist used for training but may be converted to linear layer after

		if self.use_modulelist:
			if model_version == "16_bn":
				self.x_to_c_model = vgg16_bn(pretrained=False, num_concepts=num_concepts, pretrained_weights=pretrained_weights)
			else:
				self.x_to_c_model = vgg11_bn(pretrained=False, num_concepts=num_concepts, pretrained_weights=pretrained_weights)
		else:
			if model_version == "16_bn":
				self.x_to_c_model = vgg16_bn(pretrained=False, num_classes=num_concepts, pretrained_weights=pretrained_weights)
			else:
				self.x_to_c_model = vgg11_bn(pretrained=False, num_classes=num_concepts, pretrained_weights=pretrained_weights)

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
				nn.Linear(num_concept_in, 128),
				nn.ReLU(),
				nn.Linear(128, num_classes)
			)
		else:
			self.classifier = self.lib.Linear(num_concept_in, num_classes)

	def forward(self, x):
		return self.classifier(x)


class VGG(nn.Module):
	def __init__(self, features=nn.Module, num_classes=52, num_concepts=0, pretrained_weights=None, init_weights=True, freeze=False):
		"""
		Args:
		num_classes: number of main task classes
		n_attributes: number of attributes to predict
		bottleneck: whether to make X -> A model
		expand_dim: if not 0, add an additional fc layer with expand_dim neurons
		"""
		super().__init__()

		self.num_concepts = num_concepts
		self.features = features
		self.avgPool = nn.AdaptiveAvgPool2d((7, 7))
		self.flatten = nn.Flatten()
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(0.5),
		)

		# concept training requires a modulelist for loss
		if self.num_concepts > 0:
			# all_fc replaces fc2 from standard model
			self.all_fc = nn.ModuleList()  # separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
			for i in range(self.num_concepts):
				self.all_fc.append(nn.Linear(4096, 1))
		# basic fc for standard training
		else:
			self.out_fc = nn.Linear(4096, num_classes)

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


		if init_weights:
		    for m in self.modules():
		        if isinstance(m, nn.Conv2d):
		            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
		            if m.bias is not None:
		                nn.init.constant_(m.bias, 0)
		        elif isinstance(m, nn.BatchNorm2d):
		            nn.init.constant_(m.weight, 1)
		            nn.init.constant_(m.bias, 0)
		        elif isinstance(m, nn.Linear):
		            nn.init.normal_(m.weight, 0, 0.01)
		            nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgPool(x)
		x = self.flatten(x)
		x = self.classifier(x)
		if self.num_concepts > 0:
			out = []
			for fc in self.all_fc:
				out.append(fc(x))
		else:
			out = self.out_fc(x)
		return out

	def load_partial_state_dict(self, state_dict):
		"""
		If dimensions of the current model doesn't match the pretrained one (esp for fc layer), load whichever weights that match
		"""
		own_state = self.state_dict()
		for name, param in state_dict.items():
			if name not in own_state or 'fc' in name:
				continue
			if isinstance(param, Parameter):
				param = param.data
			own_state[name].copy_(param)


def make_layers(cfg, batch_norm=False, **kwargs):

	layers: List[nn.Module] = []
	in_channels = 3
	for v in cfg:
		if v == "M":
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			v = cast(int, v)
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfgs = {
	"A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
	"B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
	"D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
	"E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
	if pretrained:
		kwargs["init_weights"] = False
	model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, **kwargs), **kwargs)
	if pretrained:
		model.load_partial_state_dict(model_zoo.load_url(model_urls[arch]))
	return model


def vgg11(pretrained=False, progress=True, **kwargs):
	r"""VGG 11-layer model (configuration "A") from
	`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
	The required minimum input size of the model is 32x32.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
	r"""VGG 11-layer model (configuration "A") with batch normalization
	`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
	The required minimum input size of the model is 32x32.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
	r"""VGG 13-layer model (configuration "B")
	`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
	The required minimum input size of the model is 32x32.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
	r"""VGG 13-layer model (configuration "B") with batch normalization
	`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
	The required minimum input size of the model is 32x32.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg("vgg13_bn", "B", True, pretrained, progress, train=train, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
	r"""VGG 16-layer model (configuration "D")
	`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
	The required minimum input size of the model is 32x32.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
	r"""VGG 16-layer model (configuration "D") with batch normalization
	`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
	The required minimum input size of the model is 32x32.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
	r"""VGG 19-layer model (configuration "E")
	`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
	The required minimum input size of the model is 32x32.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, train=True, **kwargs):
	r"""VGG 19-layer model (configuration 'E') with batch normalization
	`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
	The required minimum input size of the model is 32x32.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)
