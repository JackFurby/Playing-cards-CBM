import torch
from torch import nn


class End2EndModel(torch.nn.Module):
	def __init__(self, num_classes=6, num_concepts=52, pretrained_weights=None, freeze=False, use_sigmoid=False, use_modulelist=True):
		super(End2EndModel, self).__init__()
		self.c_to_y_model = CtoYNet(num_concepts, num_classes)
		self.use_sigmoid = use_sigmoid
		self.use_modulelist = use_modulelist  # modulelist used for training but may be converted to linear layer after

		if self.use_modulelist:
			self.x_to_c_model = PlayingCardNet(num_concepts=num_concepts, pretrained_weights=pretrained_weights)
		else:
			self.x_to_c_model = PlayingCardNet(num_classes=num_concepts, pretrained_weights=pretrained_weights)

		# if freeze == true then freeze all layers in x_to_c_model
		if freeze:  # only finetune c_to_y_model
			for name, param in self.x_to_c_model.named_parameters():
				param.requires_grad = False
			print("frozen layers in x_to_c_model")

	def forward_stage2(self, stage_1_out):
		if self.use_sigmoid:
			c_to_y_input = torch.nn.Sigmoid()(torch.cat(stage_1_out, dim=1))
		else:
			c_to_y_input = torch.cat((stage_1_out), dim=1)

		return (torch.cat(stage_1_out, dim=1), self.c_to_y_model(c_to_y_input))

	def forward(self, x):
		return self.forward_stage2(self.x_to_c_model(x))


class CtoYNet(nn.Module):
	def __init__(self, num_concept_in, num_classes):
		super(CtoYNet, self).__init__()
		self.classifier = nn.Sequential(
			nn.Linear(num_concept_in, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes)
		)

	def forward(self, x):
		return self.classifier(x)


# VGG 11 with Batch Norm
class PlayingCardNet(nn.Module):
	def __init__(self, num_classes=52, num_concepts=0, pretrained_weights=None, freeze=False):
		super(PlayingCardNet, self).__init__()
		self.num_concepts = num_concepts
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
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
