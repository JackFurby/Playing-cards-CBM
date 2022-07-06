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
	def __init__(self, num_classes, pretrained=False, pretrained_weights=None):
		super(PlayingCardNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 3)  # 3 channels; 64 is the output channel size; 3 is the kernel size;
		self.conv2 = nn.Conv2d(64, 64, 3)
		self.conv3 = nn.Conv2d(64, 128, 3)
		self.conv4 = nn.Conv2d(128, 128, 3)
		self.fc1 = nn.Linear(17280, 512)
		self.fc2 = nn.Linear(512, num_classes)
		self.pool = nn.MaxPool2d(2, 2)
		self.flatten = nn.Flatten()
		self.dropout = nn.Dropout(0.5)
		self.ReLU = nn.ReLU()

	def forward(self, x):
		x = self.pool(self.ReLU(self.conv1(x)))
		x = self.pool(self.ReLU(self.conv2(x)))
		x = self.pool(self.ReLU(self.conv3(x)))
		x = self.pool(self.ReLU(self.conv4(x)))
		x = self.flatten(x)
		x = self.dropout(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return x


class ConcpetPlayingCardNet(nn.Module):
	def __init__(self, num_concepts, pretrained=False, pretrained_weights=None, expand_dim=0):
		super(ConcpetPlayingCardNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 3)  # 3 channels; 64 is the output channel size; 3 is the kernel size;
		self.conv2 = nn.Conv2d(64, 64, 3)
		self.conv3 = nn.Conv2d(64, 128, 3)
		self.conv4 = nn.Conv2d(128, 128, 3)
		self.fc1 = nn.Linear(17280, 512)
		self.pool = nn.MaxPool2d(2, 2)
		self.flatten = nn.Flatten()
		self.dropout = nn.Dropout(0.5)
		self.ReLU = nn.ReLU()

		# all_fc replaces fc2 from standard model
		self.all_fc = nn.ModuleList()  #separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
		for i in range(num_concepts):
			self.all_fc.append(FC(512, 1, expand_dim))

	def forward(self, x):
		x = self.pool(self.ReLU(self.conv1(x)))
		x = self.pool(self.ReLU(self.conv2(x)))
		x = self.pool(self.ReLU(self.conv3(x)))
		x = self.pool(self.ReLU(self.conv4(x)))
		x = self.flatten(x)
		x = self.dropout(x)
		x = self.fc1(x)

		out = []
		for fc in self.all_fc:
			out.append(fc(x))
		return out
