import torch
from torch import nn


class PurityModel(nn.Module):
	def __init__(self, num_concepts):
		super(PurityModel, self).__init__()
		self.classifier = nn.Sequential(
			nn.Linear(num_concepts, 128),
			nn.ReLU(),
			nn.Linear(128, num_concepts)
		)

	def forward(self, x):
		return self.classifier(x)
