import torch
from torch import nn

import torch.nn.functional as F
from models.chexpert.backbone.densenet import (densenet121, densenet169, densenet201)


BACKBONES = {
	'densenet121': densenet121,
	'densenet169': densenet169,
	'densenet201': densenet201
}


BACKBONES_TYPES = {
	'densenet121': 'densenet',
	'densenet169': 'densenet',
	'densenet201': 'densenet'
}


class Classifier(nn.Module):

	def __init__(self, cfg):
		super(Classifier, self).__init__()
		self.cfg = cfg
		self.backbone = BACKBONES[cfg["backbone"]](cfg)
		self.classifier = nn.Linear(self.backbone.num_features, self.cfg["num_concepts"])  # concept classifier

		if cfg["cbm_mode"] == "joint":
			self.ctoy = nn.Sequential(
				nn.Linear(self.cfg["num_concepts"], 128),
				nn.ReLU(),
				nn.Linear(128, 2)  # output is either "no findings" True or False
			)

	def cuda(self, device=None):
		return self._apply(lambda t: t.cuda(device))

	def forward(self, x):
		# (N, C, H, W)
		#feat_map = self.backbone(x)
		out = self.backbone(x)
		out = F.adaptive_avg_pool2d(out, (1, 1))
		out = torch.flatten(out, 1)
		out = self.classifier(out)

		if self.cfg["cbm_mode"] == "joint" and self.cfg["xtoc_only"] != True:
			c_to_y_input = torch.nn.Sigmoid()(out)
			return out, self.ctoy(c_to_y_input)

		return out
