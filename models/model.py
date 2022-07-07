import torch
from torch import nn


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/template_model.py
class FC(nn.Module):

	def __init__(self, input_dim, output_dim, expand_dim, stddev=None):
		"""
		Extend standard Torch Linear layer to include the option of expanding into 2 Linear layers
		"""
		super(FC, self).__init__()
		self.expand_dim = expand_dim
		if self.expand_dim > 0:
			self.relu = nn.ReLU()
			self.fc_new = nn.Linear(input_dim, expand_dim)
			self.fc = nn.Linear(expand_dim, output_dim)
		else:
			self.fc = nn.Linear(input_dim, output_dim)
		if stddev:
			self.fc.stddev = stddev
			if expand_dim > 0:
				self.fc_new.stddev = stddev

	def forward(self, x):
		if self.expand_dim > 0:
			x = self.fc_new(x)
			x = self.relu(x)
		x = self.fc(x)
		return x


class PlayingCardNet(nn.Module):
	def __init__(self, num_classes=52, num_concepts=0, pretrained_weights=None, expand_dim=0, freeze=False):
		super(PlayingCardNet, self).__init__()
		self.num_concepts = num_concepts
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, 3),  # 3 channels; 64 is the output channel size; 3 is the kernel size;
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(64, 64, 3),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(64, 128, 3),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(128, 128, 3),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)
		self.avgPool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(0.5),
			nn.Linear(6272, 512),
			nn.ReLU()
		)
		# concept training requires a module list for loss
		if self.num_concepts > 0:
			# all_fc replaces fc2 from standard model
			self.all_fc = nn.ModuleList()  #separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
			for i in range(self.num_concepts):
				self.all_fc.append(FC(512, 1, expand_dim))
		# basic fc for standdard training
		else:
			self.out_fc = nn.Linear(512, num_classes)

		# if weights file provided, load them
		if pretrained_weights is not None:
			self.load_state_dict(torch.load(pretrained_weights))

		# if freeze == true then freeze all but fc layers
		if freeze:  # only finetune fc layer
			for name, param in self.named_parameters():
				if 'fc' not in name:
					param.requires_grad = False

	def forward(self, x):
		x = self.features(x)
		x = self.avgPool(x)
		x = self.classifier(x)
		if self.num_concepts > 0:
			out = []
			for fc in self.all_fc:
				out.append(fc(x))
		else:
			out = self.out_fc(x)
		return out
